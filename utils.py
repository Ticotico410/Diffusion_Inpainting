import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def denorm(x):
    """ Convert image from [-1, 1] to [0, 1] """
    return (x * 0.5 + 0.5).clamp(0, 1)


def viz_loss(epoch_losses, num_epochs, save_dir, name="loss"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(name.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{name}_epoch_{num_epochs}.png")
    plt.savefig(save_path)
    print(f"{name} curve saved: {save_path}")