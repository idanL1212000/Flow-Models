import os
import torch
import models
import create_plot

device = 'cpu'
os.makedirs("plots", exist_ok=True)

#same seed for reproducibly
seed = 40
torch.manual_seed(seed)

print("Loading model...")
network = models.ConditionalFlowMatchingNetwork(num_classes=5)
network.load_state_dict(torch.load('../models/ConditionalFlowMatchingModel.pth', map_location=device))
model = models.ConditionalFlowMatchingModel(network, device=device)
num_classes = 5

print("Q2: A Point from each Class...")

classes = torch.arange(num_classes)
traj, _ = model.sample(n_samples=5, classes=classes, return_trajectory=True)
create_plot.plot_class_trajectories(traj, 'plots/class_trajectories.png')

print("Q3: Sampling")
samples_per_class = 1000
all_samples = []
for c in range(num_classes):
    classes = torch.full((samples_per_class,), c, dtype=torch.long)
    samples = model.sample(n_samples=samples_per_class, classes=classes)
    all_samples.append(samples)
create_plot.plot_conditional_samples(all_samples, num_classes, 'plots/conditional_samples.png')














