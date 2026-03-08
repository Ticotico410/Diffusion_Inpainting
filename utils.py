import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def to_normal_from_tanh(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def viz_loss(
    epoch_g_losses,
    epoch_d_losses,
    train_epoch,
    ckpt_dir,
):
    """
    Visualize DCGAN losses per epoch.

    Args:
        epoch_g_losses: list[float], generator loss per epoch
        epoch_d_losses: list[float], discriminator loss per epoch
        train_epoch: int, total number of epochs
        ckpt_dir: str, directory to save plots
    Returns:
        (g_path, d_path)
    """
    epochs = range(1, train_epoch + 1)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Generator Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_g_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("G loss")
    plt.title("Generator Loss", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    g_path = os.path.join(ckpt_dir, "g_loss.png")
    plt.savefig(g_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"G loss saved: {g_path}")

    # Discriminator Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_d_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("D loss")
    plt.title("Discriminator Loss", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    d_path = os.path.join(ckpt_dir, "d_loss.png")
    plt.savefig(d_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"D loss saved: {d_path}")

    return g_path, d_path