import torch
import os
from torch.utils.data import random_split
from create_data import create_unconditional_olympic_rings
from Normalizing_Flows.flow_model import NormalizingFlowModel, train_flow
from create_plot import plot_flow_model_loss

os.makedirs("../models",exist_ok=True)
os.makedirs("/plots",exist_ok=True)

#same seed for reproducibly
seed = 42
torch.manual_seed(seed)

# Generate data 250000 test points and 62500 val point (0.8,0.2) split
data_np = create_unconditional_olympic_rings(n_points=312500, verbose=False)
data_tensor = torch.tensor(data_np, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data_tensor)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NormalizingFlowModel(input_dim=2, num_layers=15).to(device)

# Train
train_loss, val_loss, train_pz, train_det, val_pz, val_det = train_flow(
    model, train_dataset, val_dataset, epochs=20, device=device
)

# Plot results
plot_flow_model_loss(train_loss, val_loss, train_pz, train_det, val_pz, val_det)

