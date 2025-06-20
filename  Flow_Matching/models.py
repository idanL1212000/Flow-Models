import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm

class FlowMatchingNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4, time_embed_dim=32):

        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_net = nn.Sequential(
            nn.Linear(1, self.time_embed_dim) #can change to be more complex
        )

        layers = []
        in_dim = input_dim + self.time_embed_dim

        for i in range(n_layers-1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.main_net = nn.Sequential(*layers)

    def forward(self, x, t):

        t_embed =  self.time_net(t.unsqueeze(-1))
        x_t = torch.cat([x, t_embed], dim=-1)
        return self.main_net(x_t)

class UnconditionalFlowMatchingModel:
    def __init__(self, network, device='cpu'):
        self.network = network.to(device)
        self.device = device
        self.losses = []

    def train_step(self, x_batch, optimizer):

        batch_size = x_batch.shape[0]
        epsilon = torch.randn_like(x_batch).to(self.device)
        t = torch.rand(batch_size).to(self.device)
        y_t = (1 - t.unsqueeze(-1)) * epsilon + t.unsqueeze(-1) * x_batch
        true_v = x_batch - epsilon
        pred_v = self.network(y_t, t)
        loss = torch.mean((pred_v - true_v) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, data_loader, epochs=20, lr=1e-3):
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        pbar = tqdm.tqdm(range(epochs))

        self.losses = []
        for _ in pbar:
            loss = 0
            for batch_idx, (x_batch,) in enumerate(data_loader):
                x_batch = x_batch.to(self.device)
                loss += self.train_step(x_batch, optimizer)
            avg_loss = loss / len(data_loader)
            self.losses.append(avg_loss)
            scheduler.step()
            pbar.set_postfix(loss = avg_loss)
        self.save_network()

    def sample(self, n_samples=1000, dt=0.001, return_trajectory=False):
        self.network.eval()
        with torch.no_grad():
            y = torch.randn(n_samples, 2).to(self.device)
            if return_trajectory:
                trajectory = [y.cpu().numpy()]

            time_steps = torch.arange(0, 1 + dt, dt).to(self.device)
            for t_val in time_steps[:-1]:
                t_batch = torch.full((n_samples,), t_val.item()).to(self.device)
                v_pred = self.network(y, t_batch)
                y = y + v_pred * dt

                if return_trajectory:
                    trajectory.append(y.cpu().numpy())

            if return_trajectory:
                return np.array(trajectory), time_steps.cpu().numpy()
            else:
                return y.cpu().numpy()

    def sample_at_times(self, n_samples=1000, times=None, dt=0.001):
        if times is None:
            times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.network.eval()

        with torch.no_grad():
            y = torch.randn(n_samples, 2).to(self.device)

            results = {}
            current_time = 0.0
            time_idx = 0

            if 0.0 in times:
                results[0.0] = y.cpu().numpy()
                time_idx += 1

            while current_time < 1.0 and time_idx < len(times):
                target_time = times[time_idx]

                while current_time < target_time and current_time < 1.0:
                    t_batch = torch.full((n_samples,), current_time).to(self.device)
                    v_pred = self.network(y, t_batch)
                    y = y + v_pred * dt
                    current_time += dt

                if abs(current_time - target_time) < dt:
                    results[target_time] = y.cpu().numpy()
                    time_idx += 1

        return results

    def reverse_sample(self, points, dt=0.001):
        self.network.eval()

        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            trajectory = [points_tensor.cpu().numpy()]

            time_steps = torch.arange(1, -dt, -dt).to(self.device)

            y = points_tensor.clone()

            for t_val in time_steps:
                if t_val >= 0:
                    t_batch = torch.full((points_tensor.shape[0],), t_val.item()).to(self.device)
                    v_pred = self.network(y, t_batch)
                    y = y - v_pred * dt
                    trajectory.append(y.cpu().numpy())

        return np.array(trajectory), time_steps.cpu().numpy()

    def save_network(self, save_path = "../models/UnconditionalFlowMatchingModel.pth"):
        torch.save(self.network.state_dict(), save_path)

class ConditionalFlowMatchingNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4, time_embed_dim=32, num_classes=5, class_embed_dim=16):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_net = nn.Sequential(
            nn.Linear(1, self.time_embed_dim)
        )
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        in_dim = input_dim + time_embed_dim + class_embed_dim
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.main_net = nn.Sequential(*layers)

    def forward(self, x, t, class_labels):
        t_embed = self.time_net(t.unsqueeze(-1))
        c_embed = self.class_embed(class_labels)
        x_t_c = torch.cat([x, t_embed, c_embed], dim=-1)
        return self.main_net(x_t_c)

class ConditionalFlowMatchingModel:
    def __init__(self, network, device='cpu'):
        self.network = network.to(device)
        self.device = device
        self.losses = []

    def train_step(self, x_batch, class_batch, optimizer):
        batch_size = x_batch.shape[0]
        epsilon = torch.randn_like(x_batch).to(self.device)
        t = torch.rand(batch_size).to(self.device)
        y_t = (1 - t.unsqueeze(-1)) * epsilon + t.unsqueeze(-1) * x_batch
        true_v = x_batch - epsilon
        pred_v = self.network(y_t, t, class_batch)
        loss = torch.mean((pred_v - true_v) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, data_loader, epochs=20, lr=1e-3):
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        pbar = tqdm.tqdm(range(epochs))
        self.losses = []
        for _ in pbar:
            loss = 0
            for batch_idx, (x_batch, class_batch) in enumerate(data_loader):
                x_batch, class_batch = x_batch.to(self.device), class_batch.to(self.device)
                loss += self.train_step(x_batch, class_batch, optimizer)
            avg_loss = loss / len(data_loader)
            self.losses.append(avg_loss)
            scheduler.step()
            pbar.set_postfix(loss=avg_loss)
        self.save_network()

    def sample_specific(self, samples, dt=0.001, classes=None, return_trajectory=False):
        self.network.eval()
        n_samples = len(samples)
        with torch.no_grad():
            classes = torch.full((n_samples,), 0, dtype=torch.long) if classes is None else classes.to(self.device)
            y = samples.to(self.device)
            trajectory = [y.cpu().numpy()] if return_trajectory else None
            time_steps = torch.arange(0, 1 + dt, dt).to(self.device)
            for t_val in time_steps[:-1]:
                t_batch = torch.full((n_samples,), t_val.item()).to(self.device)
                v_pred = self.network(y, t_batch, classes)
                y = y + v_pred * dt
                if return_trajectory: trajectory.append(y.cpu().numpy())
            return (np.array(trajectory), time_steps.cpu().numpy()) if return_trajectory else y.cpu().numpy()

    def sample(self, n_samples=1000, dt=0.001, classes=None, return_trajectory=False):
        return self.sample_specific(torch.randn(n_samples, 2), dt, classes, return_trajectory)

    def reverse_sample(self, points, classes, dt=0.001, return_trajectory=False):
        self.network.eval()

        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            class_tensor = torch.LongTensor(classes).to(self.device)

            if return_trajectory:
                trajectory = [points_tensor.cpu().numpy()]

            y = points_tensor.clone()
            time_steps = torch.arange(1, -dt, -dt).to(self.device)

            for t_val in time_steps:
                if t_val >= 0:
                    t_batch = torch.full((points_tensor.shape[0],), t_val.item()).to(self.device)
                    v_pred = self.network(y, t_batch, class_tensor)
                    y = y - v_pred * dt

                    if return_trajectory:
                        trajectory.append(y.cpu().numpy())

            if return_trajectory:
                return np.array(trajectory), time_steps.cpu().numpy()
            else:
                return y.cpu().numpy()

    def save_network(self, save_path="../models/ConditionalFlowMatchingModel.pth"):
        torch.save(self.network.state_dict(), save_path)