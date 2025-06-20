import os
import torch
import models
from torch.utils.data import TensorDataset, DataLoader
from create_data import create_olympic_rings
from create_plot import plot_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("../models", exist_ok=True)

#same seed for reproducibly
seed = 42
torch.manual_seed(seed)

print("Generating Olympic rings data...")
data_np, classes_np, classes_np_int = create_olympic_rings(250000)
data_tensor = torch.tensor(data_np, dtype=torch.float32)
class_tensor = torch.tensor(classes_np, dtype=torch.long)
dataset = TensorDataset(data_tensor, class_tensor)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

print("Creating Conditional Flow Matching model...")
network = models.ConditionalFlowMatchingNetwork(num_classes=5)
model = models.ConditionalFlowMatchingModel(network, device=device)
print("Training model...")
model.train(data_loader, epochs=20, lr=1e-3)

plot_loss(model.losses, "Conditional Flow Matching Training Loss")