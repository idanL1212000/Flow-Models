import matplotlib.pyplot as plt
import numpy as np

from create_data import create_unconditional_olympic_rings, create_olympic_rings


def plot_flow_model_loss(train_loss, val_loss, train_log_pz, train_log_det, val_log_pz, val_log_det):
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_log_pz, label='Train log(pz)', linestyle='--')
    plt.plot(epochs, train_log_det, label='Train log(det)', linestyle='--')
    plt.plot(epochs, val_log_pz, label='Val log(pz)', linewidth=2)
    plt.plot(epochs, val_log_det, label='Val log(det)', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Component Value")
    plt.title("Loss Components Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/loss_curve.png')
    plt.show()

def plot_loss(losses, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    safe_title = title.replace(" ", "_")
    plt.savefig(f'plots/{safe_title}.png')
    plt.show()

def plot_points(points, title, save_at, color = 'red'):
    plt.figure(figsize=(15, 5))
    plt.scatter(points[:, 0], points[:, 1], s=3, alpha=0.6, color=color)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_at, dpi=300)
    plt.close()

def plot_points_2(points1, points2, title1, title2, save_at, color1 = 'red', color2 = 'red'):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(points1[:, 0], points1[:, 1], s=3, alpha=0.6, color=color1)
    plt.title(title1)
    plt.gca().set_aspect('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(points2[:, 0], points2[:, 1], s=3, alpha=0.6, color=color2)
    plt.title(title2)
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_at, dpi=300)
    plt.close()

def plot_trajectories_dif_fig(intermediates, fig_num, num_points):
    background_data = create_unconditional_olympic_rings(n_points=10000, verbose=False)

    trajectories = [np.array([intermediates[t][i] for t in range(len(intermediates))])
                    for i in range(num_points)]

    fig, axes = plt.subplots(1, num_points, figsize=(15, 6))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=len(intermediates) - 1)


    for i, traj in enumerate(trajectories):
        points = np.array(traj)
        ax = axes[i]

        ax.scatter(background_data[:, 0], background_data[:, 1], s=1, alpha=0.3, color='gray',
                   label='Background Data Distribution')

        ax.scatter(points[:, 0], points[:, 1],
                   c=np.arange(len(intermediates)),
                   cmap=cmap, norm=norm,
                   s=15, alpha=0.8)

        ax.plot(points[:, 0], points[:, 1], alpha=0.6, linewidth=1)

        ax.annotate('Start', xy=points[0], xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='green', ha='left')
        ax.annotate('End', xy=points[-1], xytext=(-10, -10), textcoords='offset points',
                    fontsize=8, color='red', ha='right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Layer (Time step)', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'plots/points_trajectories_{fig_num}.png', dpi=300)
    plt.close()

def plot_trajectories_one_fig(intermediates, num_points, save_as):
    background_data = create_unconditional_olympic_rings(n_points=10000, verbose=False)

    trajectories = [np.array([intermediates[t][i] for t in range(len(intermediates))])
                    for i in range(num_points)]

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=len(intermediates) - 1)

    ax.scatter(background_data[:, 0], background_data[:, 1], s=1, alpha=0.3, color='gray',
               label='Background Data Distribution')

    for traj in trajectories:
        points = np.array(traj)
        ax.scatter(points[:, 0], points[:, 1],
                   c=np.arange(len(intermediates)),
                   cmap=cmap, norm=norm,
                   s=15, alpha=0.8)
        ax.plot(points[:, 0], points[:, 1], alpha=0.6, linewidth=1)
        ax.annotate('Start', xy=points[0], xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='green', ha='left')
        ax.annotate('End', xy=points[-1], xytext=(-10, -10), textcoords='offset points',
                    fontsize=8, color='red', ha='right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Layer (Time step)', fontsize=10)

    plt.tight_layout()

    plt.savefig(save_as, dpi=300)
    plt.close()

RING_COLORS = ['black', 'blue', 'green', 'red', 'yellow']

def plot_conditional_samples(all_samples, num_classes, save_path):
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        samples_class = all_samples[c]
        plt.scatter(samples_class[:, 0], samples_class[:, 1], s=5, color=RING_COLORS[c], alpha=0.7)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_class_trajectories(combined_trajectories, save_path):
    plt.figure(figsize=(8, 6))

    num_timesteps, num_classes, _ = combined_trajectories.shape

    bg_points, bg_labels, _ = create_olympic_rings(n_points=10000, verbose=False)

    for c in range(num_classes):
        class_mask = (bg_labels == c)
        class_points = bg_points[class_mask]
        plt.scatter(class_points[:, 0], class_points[:, 1], color=RING_COLORS[c], s=1, alpha=0.1, label=f'Class {c}')

    for c in range(num_classes):
        trajectory_for_class_c = combined_trajectories[:, c, :]

        plt.plot(trajectory_for_class_c[:, 0], trajectory_for_class_c[:, 1],
                 color=RING_COLORS[c], alpha=0.7, linestyle='-')

        plt.scatter(trajectory_for_class_c[0, 0], trajectory_for_class_c[0, 1],
                    s=10, color=RING_COLORS[c], marker='o', alpha=0.8)
        plt.scatter(trajectory_for_class_c[-1, 0], trajectory_for_class_c[-1, 1],
                    s=20, color=RING_COLORS[c], marker='x', alpha=0.9)


    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(['start - o','end - x'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
