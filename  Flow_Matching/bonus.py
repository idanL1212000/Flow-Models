import os
import torch
import models
import create_plot

device = 'cpu'
os.makedirs("plots", exist_ok=True)

#same seed for reproducibly
seed = 40
torch.manual_seed(seed)

network = models.ConditionalFlowMatchingNetwork(num_classes=5)
network.load_state_dict(torch.load('../models/ConditionalFlowMatchingModel.pth', map_location=device))
model = models.ConditionalFlowMatchingModel(network, device=device)

point = model.reverse_sample([[4,5]],classes=torch.tensor([2]))

traj, _ = model.sample_specific(torch.tensor(point), classes=torch.tensor([2]), return_trajectory=True)

create_plot.plot_class_trajectories(traj, 'plots/bonus_trajectories.png')
