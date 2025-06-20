import os

import torch

import models
import create_plot

device = 'cpu'
os.makedirs("/plots",exist_ok=True)

#same seed for reproducibly
seed = 42
torch.manual_seed(seed)

print("Creating Flow Matching model...")
network = models.FlowMatchingNetwork(input_dim=2, hidden_dim=64, n_layers=5)
network.load_state_dict(torch.load('../models/UnconditionalFlowMatchingModel.pth'))
model = models.UnconditionalFlowMatchingModel(network, device=device)

print("Q2: Sampling at different time steps...")
times = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
time_samples = model.sample_at_times(n_samples=1000,
                                     times=times)
for i in range(0,len(times),2):
    create_plot.plot_points_2(time_samples[times[i]], time_samples[times[i + 1]], f't = {times[i]}',
                             f't = {times[i + 1]}', f'plots/Flow_Progression_Over_Time{i // 2}.png')

print("Q3: Point Trajectory...")
num_points = 10
intermediates,_ = model.sample(num_points,return_trajectory=True)
create_plot.plot_trajectories_one_fig(intermediates, num_points, 'plots/points_trajectories.png')

print("Q4: Time Quantization...")
dt_values = [0.002, 0.02, 0.05, 0.1, 0.2]
for dt in dt_values:
    samples = model.sample(n_samples=1000, dt=dt)
    create_plot.plot_points(samples,f"Sampled Distribution with dt = {dt}",
                           f'plots/time_quantization_dt_{dt}.png')

print("Q5: Reversing the Flow...")
inside_points = [[0.5, 0.0], [-0.5, 0.0], [0.0, -0.5]]
outside_points = [[1.8, 1.8], [-1.3, -1.3]]
points = inside_points + outside_points
intermediates, _ = model.reverse_sample(points)
create_plot.plot_trajectories_one_fig(intermediates,5,'plots/reverse_points_trajectories.png')

