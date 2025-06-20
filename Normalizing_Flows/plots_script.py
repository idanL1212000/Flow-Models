import numpy as np
import torch
import os
from Normalizing_Flows.flow_model import NormalizingFlowModel
from create_data import create_unconditional_olympic_rings
import create_plot

os.makedirs("plots", exist_ok=True)

print("Loading Normalizing Flow model...")
input_dim = 2
hidden_dim = 8
num_layers = 15
hidden_block_layers = 3

model = NormalizingFlowModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    hidden_block_layers=hidden_block_layers
)
model.load_state_dict(torch.load('../models/normalizing_flow_epoch_20.pth'))
model.eval()

def generate_samples(model, n_samples=1000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    z = torch.randn(n_samples, model.input_dim)
    with torch.no_grad():
        samples, _ = model(z)
    return samples.cpu().numpy()

print("Q2: Generating sample comparisons...")
data = create_unconditional_olympic_rings(1000, verbose=False)
seeds = [42, 123, 7777]
n_samples = 1000

seed = seeds[0]
samples = generate_samples(model, n_samples, seed)
create_plot.plot_points_2(data, samples, 'Original Data', f'Samples (Seed={seed})', 'plots/samples_comparison_1.png', 'blue')

seed1, seed2 = seeds[1], seeds[2]
samples1 = generate_samples(model, n_samples, seed1)
samples2 = generate_samples(model, n_samples, seed2)
create_plot.plot_points_2(samples1, samples2, f'Seed={seed1}', f'Seed={seed2}', 'plots/samples_comparison_2.png')

#same seed for reproducibly
torch.manual_seed(42)
np.random.seed(42)

print("Q3: Plotting layer progression...")
selected_layers = [0, 3, 6, 9, 12, 14]
model.eval()
z = torch.randn(1000, 2)
with torch.no_grad():
    _, _, intermediates = model(z, return_intermediates=True)
titles = [f"After Layer {i}" for i in selected_layers]
for i in range(len(selected_layers)):
    create_plot.plot_points(intermediates[selected_layers[i]],titles[i], f'plots/samples_layer_{selected_layers[i]}.png', color='blue')

print("Q4: Plotting point trajectories...")
num_points = 2
for i in range(5):
    z = torch.randn(num_points, 2)
    with torch.no_grad():
        _, _, intermediates = model(z, return_intermediates=True)
    create_plot.plot_trajectories_dif_fig(intermediates, i, num_points)


print("Q5: Calculating log probabilities...")
inside_points = [[0.5, 0.0], [-0.5, 0.0], [0.0, -0.5]]
outside_points = [[1.8, 1.8], [-1.3, -1.3]]
labels_inside = ['Inside 1', 'Inside 2', 'Inside 3']
labels_outside = ['Outside 1', 'Outside 2']

z_inside, log_det_inside, intermediates_inside = model.inverse(torch.tensor(inside_points), True)
z_outside, log_det_outside, intermediates_outside = model.inverse(torch.tensor(outside_points), True)

create_plot.plot_trajectories_one_fig(intermediates_inside, len(inside_points), 'plots/inside_trajectories.png')
create_plot.plot_trajectories_one_fig(intermediates_outside, len(outside_points), 'plots/outside_trajectories.png')

base_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
log_p_inside = (base_dist.log_prob(z_inside) + log_det_inside).detach().numpy()
log_p_outside = (base_dist.log_prob(z_outside) + log_det_outside).detach().numpy()

print("\n--- Log Probabilities ---")
for i, (label, pt) in enumerate(zip(labels_inside, inside_points)):
    print(f"{label} {pt}: {log_p_inside[i]:.4f}")
for i, (label, pt) in enumerate(zip(labels_outside, outside_points)):
    print(f"{label} {pt}: {log_p_outside[i]:.4f}")

print(f"\nAverage Inside: {np.mean(log_p_inside):.4f}, Average Outside: {np.mean(log_p_outside):.4f}")