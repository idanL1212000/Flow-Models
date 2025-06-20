import os
import torch

import models
from torch.utils.data import TensorDataset, DataLoader

from create_data import create_unconditional_olympic_rings
from create_plot import plot_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("../models",exist_ok=True)

#same seed for reproducibly
seed = 42
torch.manual_seed(seed)

print("Generating Olympic rings data...")
point_num = 250000
data_np = create_unconditional_olympic_rings(point_num,verbose=False)
data_tensor = torch.tensor(data_np, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

print("Creating Flow Matching model...")
network = models.FlowMatchingNetwork(input_dim=2, hidden_dim=64, n_layers=5)
model = models.UnconditionalFlowMatchingModel(network, device=device)

print("Training model...")
model.train(data_loader, epochs=20, lr=1e-3)

print("Q1: Plotting training loss...")
plot_loss(model.losses, "Flow Matching Training Loss")
