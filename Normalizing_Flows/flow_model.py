import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim = 8, hidden_layers = 3):
        super(LinearBlock, self).__init__()
        layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(), )]

        for _ in range(hidden_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(), ))
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_l):

        return self.net(z_l)

class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim = 8, hidden_layers = 3, eps=1e-6):
        super(AffineCoupling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.eps = eps

        self.s_block = LinearBlock(input_dim // 2, hidden_dim, hidden_layers)
        self.b_block = LinearBlock(input_dim // 2, hidden_dim, hidden_layers)

    def forward(self,z):
        # Split input into two halves
        z_l, z_r = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]

        # Predict log(s) and b
        log_s = torch.clamp(self.s_block(z_l), min=-5.0, max=5.0)
        b = self.b_block(z_l)
        s = torch.exp(log_s) + self.eps

        # Affine coupling
        y_r = s * z_r + b
        y = torch.cat([z_l, y_r], dim=1)

        return y, log_s

    def inverse(self, y):
        # Split input into two halves
        y_l, y_r = y[:, :self.input_dim // 2], y[:, y.shape[1] // 2:]

        # Predict log(s) and b
        log_s = torch.clamp(self.s_block(y_l), min=-5.0, max=5.0)
        b = self.b_block(y_l)
        s = torch.exp(log_s) + self.eps

        # Affine coupling
        z_r = (y_r - b) / s
        z = torch.cat([y_l, z_r], dim=1)

        return z, log_s

    def log_det_jacobian(self, log_s):
        # Log determinant of the Jacobin
        return torch.sum(log_s, dim=1)

class Permutation(nn.Module):
    def __init__(self, input_dim):
        super(Permutation, self).__init__()
        self.input_dim = input_dim
        self.register_buffer('perm_indices', torch.randperm(input_dim))
        self.register_buffer('inv_perm_indices', torch.argsort(self.perm_indices))

    def forward(self, z):
        return z[:, self.perm_indices]

    def inverse(self, y):
        return y[:, self.inv_perm_indices]

class NormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim = 8, num_layers = 15, hidden_block_layers = 3):
        super(NormalizingFlowModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_block_layers = hidden_block_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(AffineCoupling(input_dim, hidden_dim, hidden_block_layers))
            self.layers.append(Permutation(input_dim))
        self.layers.append(AffineCoupling(input_dim, hidden_dim, hidden_block_layers))

    def forward(self, z, return_intermediates=False):
        intermediates = [z.detach().cpu()]
        log_det_jacobian = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            if isinstance(layer, AffineCoupling):
                z, log_s = layer(z)
                log_det_jacobian += layer.log_det_jacobian(log_s)
            else:
                z = layer(z)
            if return_intermediates:
                intermediates.append(z.detach().cpu())
        return (z, log_det_jacobian) if not return_intermediates else (z, log_det_jacobian, intermediates)

    def inverse(self, y, return_intermediates=False):
        intermediates = [y.detach().cpu()]
        log_det_jacobian = torch.zeros(y.shape[0], device=y.device)
        for layer in reversed(self.layers):
            if isinstance(layer, AffineCoupling):
                y, log_s = layer.inverse(y)
                log_det_jacobian -= layer.log_det_jacobian(log_s)
            else:
                y = layer.inverse(y)
            intermediates.append(y.detach().cpu())
        return (y, log_det_jacobian) if not return_intermediates else (y, log_det_jacobian, intermediates)

def train_flow(model, train_data, val_data, epochs=20, batch_size=128, lr=1e-3, device='cpu', saved_epochs=None):
    if saved_epochs is None:
        saved_epochs = [5, 7, 10, 15, 17, 20]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    # Track metrics
    train_losses, val_losses = [], []
    train_log_pz, train_log_det = [], []
    val_log_pz, val_log_det = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model.train()
        epoch_train_loss, epoch_train_pz, epoch_train_det = 0.0, 0.0, 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            z, log_det = model.inverse(x)
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(2).to(device), torch.eye(2).to(device)
            )
            log_pz = base_dist.log_prob(z)
            loss = -(log_pz + log_det).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_pz += log_pz.mean().item()
            epoch_train_det += log_det.mean().item()

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_pz, epoch_val_det = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                z, log_det = model.inverse(x)
                log_pz = base_dist.log_prob(z)
                loss = -(log_pz + log_det).mean()
                epoch_val_loss += loss.item()
                epoch_val_pz += log_pz.mean().item()
                epoch_val_det += log_det.mean().item()

        # Normalize metrics
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_log_pz.append(epoch_train_pz / len(train_loader))
        train_log_det.append(epoch_train_det / len(train_loader))
        val_log_pz.append(epoch_val_pz / len(val_loader))
        val_log_det.append(epoch_val_det / len(val_loader))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()
        print("Epoch ", epoch ," Test Loss: ",train_loss, ". Val Loss: ",val_loss)
        if (epoch + 1) in saved_epochs:
            torch.save(model.state_dict(), f'../models/normalizing_flow_epoch_{epoch + 1}.pth')
            print(f"Saved model at epoch {epoch + 1}")

    return train_losses, val_losses, train_log_pz, train_log_det, val_log_pz, val_log_det