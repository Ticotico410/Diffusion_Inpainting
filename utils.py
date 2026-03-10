import os
import torch
import lpips
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pytorch_msssim import ssim


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


def crop_masked_patch(img, mask):
    """
    img:  [B, C, H, W]
    mask: [B, 1, H, W], 1 means masked square area

    return:
        patches: list of [C, h, w]
    """
    patches = []

    B = img.shape[0]
    for i in range(B):
        coords = torch.nonzero(mask[i, 0], as_tuple=False)  # [N, 2]

        top = coords[:, 0].min().item()
        bottom = coords[:, 0].max().item() + 1
        left = coords[:, 1].min().item()
        right = coords[:, 1].max().item() + 1

        patch = img[i, :, top:bottom, left:right]
        patches.append(patch)

    return patches


def masked_psnr(pred, gt, mask):
    pred_patches = crop_masked_patch(pred, mask)
    gt_patches = crop_masked_patch(gt, mask)

    psnr_list = []

    for pred_patch, gt_patch in zip(pred_patches, gt_patches):
        mse = torch.mean((pred_patch - gt_patch) ** 2)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        psnr_list.append(psnr)

    return torch.stack(psnr_list).mean()


def masked_ssim(pred, gt, mask):
    pred_patches = crop_masked_patch(pred, mask)
    gt_patches = crop_masked_patch(gt, mask)

    ssim_list = []

    for pred_patch, gt_patch in zip(pred_patches, gt_patches):
        pred_patch = pred_patch.unsqueeze(0)   # [1, C, h, w]
        gt_patch = gt_patch.unsqueeze(0)

        ssim_val = ssim(pred_patch, gt_patch, data_range=1.0, size_average=True)
        ssim_list.append(ssim_val)

    return torch.stack(ssim_list).mean()


_lpips_metric = None
def get_lpips_metric(device):
    global _lpips_metric
    if _lpips_metric is None:
        _lpips_metric = lpips.LPIPS(net="vgg").to(device)
        _lpips_metric.eval()
    return _lpips_metric

def masked_lpips(pred, gt, mask):
    pred_patches = crop_masked_patch(pred, mask)
    gt_patches = crop_masked_patch(gt, mask)

    device = pred.device
    lpips_metric = get_lpips_metric(device)

    lpips_list = []

    for pred_patch, gt_patch in zip(pred_patches, gt_patches):
        pred_patch = pred_patch.unsqueeze(0) * 2 - 1   # [0,1] -> [-1,1]
        gt_patch = gt_patch.unsqueeze(0) * 2 - 1

        val = lpips_metric(pred_patch, gt_patch).mean()
        lpips_list.append(val)

    return torch.stack(lpips_list).mean()