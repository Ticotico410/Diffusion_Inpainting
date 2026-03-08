from __future__ import print_function

import os
import copy
import math
import shutil
import argparse

import torch
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm
from ignite.metrics import FID, InceptionScore

from dcgan import *
from dataLoader import get_dataloader
from utils import viz_loss, to_normal_from_tanh


IM_SIZE = 128


def train_dcgan(config, train_loader):
    lr = float(config["lr"])
    z_size = int(config["z_size"])
    train_epoch = int(config["num_epochs"])
    save_per_epoch = int(config["save_per_epoch"])
    sample_per_epoch = int(config.get("sample_per_epoch", 1))

    ckpt_dir = config["ckpt_dir"]
    out_dir = config.get("out_dir", "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Models
    G = Generator(img_size=IM_SIZE, channels=3, latent_dim=z_size).to(device)
    D = Discriminator(img_size=IM_SIZE, channels=3).to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    # Optimizers
    b1 = float(config.get("b1", 0.5))
    b2 = float(config.get("b2", 0.999))
    d_lr_mult = float(config.get("d_lr_mult", 2.0))  # safer default than 4.0
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(D.parameters(), lr=lr * d_lr_mult, betas=(b1, b2))

    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

    print("Loading training dataset...")
    print("Train set size:", len(train_loader.dataset))

    epoch_g_losses, epoch_d_losses = [], []

    fixed_z = torch.randn(25, z_size, device=device)

    # For safe return even if train_epoch==0
    last_avg_g = float("nan")
    last_avg_d = float("nan")

    for epoch in range(train_epoch):
        print(f"\nEpoch {epoch+1}/{train_epoch}")
        G.train()
        D.train()

        g_loss_sum = 0.0
        d_loss_sum = 0.0

        batch_pbar = tqdm(
            train_loader,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}",
        )

        for imgs, _ in batch_pbar:
            imgs = imgs.to(device, non_blocking=True)  # real imgs in [-1,1]
            bsz = imgs.size(0)
            real_targets = torch.empty(bsz, 1, device=device).uniform_(0.85, 1.0)
            fake_targets = torch.empty(bsz, 1, device=device).uniform_(0.0, 0.15)
            gen_targets = torch.ones(bsz, 1, device=device)  # or 0.9*ones

            # ---- Train Generator ----
            optimizer_G.zero_grad(set_to_none=True)
            z = torch.randn(bsz, z_size, device=device)
            gen_imgs = G(z)
            g_loss = adversarial_loss(D(gen_imgs), gen_targets)
            g_loss.backward()
            optimizer_G.step()

            # ---- Train Discriminator (fresh fake) ----
            optimizer_D.zero_grad(set_to_none=True)
            real_loss = adversarial_loss(D(imgs), real_targets)

            z = torch.randn(bsz, z_size, device=device)
            fake_imgs = G(z).detach()
            fake_loss = adversarial_loss(D(fake_imgs), fake_targets)

            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            g_loss_sum += float(g_loss.item())
            d_loss_sum += float(d_loss.item())

        batch_count = len(train_loader)
        if batch_count == 0:
            raise RuntimeError(
                "train_loader is empty (len(train_loader)==0). "
                "Check data_dir / split_ratio / dataset."
            )

        last_avg_g = g_loss_sum / batch_count
        last_avg_d = d_loss_sum / batch_count
        epoch_g_losses.append(last_avg_g)
        epoch_d_losses.append(last_avg_d)
        print(f"Train loss: G={last_avg_g:.6f}, D={last_avg_d:.6f}")

        # ---- Sample images (fixed z) ----
        if epoch == 0 or (epoch + 1) % sample_per_epoch == 0:
            G.eval()
            with torch.no_grad():
                sample_imgs = G(fixed_z)
                save_image(
                    sample_imgs,
                    os.path.join(out_dir, f"sample_epoch_{epoch+1}.png"),
                    nrow=5,
                    normalize=True,  # [-1,1] -> [0,1]
                )
            G.train()

        # ---- Interval checkpoint ----
        if (epoch + 1) % save_per_epoch == 0 or (epoch + 1) == train_epoch:
            checkpoint = {
                "epoch": epoch + 1,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "optG_state_dict": optimizer_G.state_dict(),
                "optD_state_dict": optimizer_D.state_dict(),
                "avg_g_loss": last_avg_g,
                "avg_d_loss": last_avg_d,
                "z_size": z_size,
                "img_size": IM_SIZE,
                "channels": 3,
                "d_lr_mult": d_lr_mult,
            }
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, ckpt_path)

    # ---- Save final as best.pth ----
    best_checkpoint = {
        "epoch": train_epoch,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "optG_state_dict": optimizer_G.state_dict(),
        "optD_state_dict": optimizer_D.state_dict(),
        "z_size": z_size,
        "img_size": IM_SIZE,
        "channels": 3,
        "d_lr_mult": d_lr_mult,
        "final_avg_g_loss": last_avg_g,
        "final_avg_d_loss": last_avg_d,
    }
    torch.save(best_checkpoint, os.path.join(ckpt_dir, "best.pth"))
    print("Saved final as best.pth")

    # ---- Visualize loss ----
    viz_loss(epoch_g_losses, epoch_d_losses, train_epoch, ckpt_dir)

    return train_epoch, last_avg_g, G.state_dict()


def eval_dcgan(config, test_loader):
    batch_size = int(config["batch_size"])
    ckpt_dir = config["ckpt_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_name = config.get("ckpt_name", "best.pth")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    z_size = int(checkpoint.get("z_size", config.get("z_size", 100)))
    img_size = int(checkpoint.get("img_size", IM_SIZE))
    channels = int(checkpoint.get("channels", 3))

    G = Generator(img_size=img_size, channels=channels, latent_dim=z_size).to(device)
    G.load_state_dict(checkpoint["G_state_dict"])
    G.eval()
    print(f"Generator loaded: z_size={z_size}, img_size={img_size}, channels={channels}")

    # Metrics
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device)

    if len(test_loader.dataset) == 0:
        raise RuntimeError("test_loader.dataset is empty. Check split_ratio / data_dir.")

    # How many samples to evaluate
    max_eval_samples = int(config.get("max_eval_samples", len(test_loader.dataset)))
    max_eval_samples = min(max_eval_samples, len(test_loader.dataset))
    num_eval_batches = math.ceil(max_eval_samples / batch_size)
    print(f"Evaluating with {max_eval_samples} samples (~{num_eval_batches} batches, batch_size={batch_size}) ...")

    os.makedirs("results_gen", exist_ok=True)

    sample_real_list, sample_fake_list = [], []
    seen = 0

    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for batch_idx, (real_imgs, _) in enumerate(eval_pbar):
            real_imgs = real_imgs.to(device, non_blocking=True)  # expected [-1,1]
            bsz = real_imgs.size(0)

            # generate fake
            z = torch.randn(bsz, z_size, device=device)
            fake_imgs = G(z)  # [-1,1]

            # save a small visualization from the first batch
            if batch_idx == 0:
                sample_real_list.append(to_normal_from_tanh(real_imgs[:8]).cpu())
                sample_fake_list.append(to_normal_from_tanh(fake_imgs[:8]).cpu())

            # metric input: [0,1] and 299x299
            real_01 = to_normal_from_tanh(real_imgs)
            fake_01 = to_normal_from_tanh(fake_imgs)

            real_in = F.interpolate(real_01, size=(299, 299), mode="bilinear", align_corners=False)
            fake_in = F.interpolate(fake_01, size=(299, 299), mode="bilinear", align_corners=False)

            # update metrics
            fid_metric.update((fake_in, real_in))
            is_metric.update(fake_in)

            seen += bsz
            if seen >= max_eval_samples:
                break

    # compute
    fid_score = fid_metric.compute()
    is_out = is_metric.compute()

    print(f"\n{'='*50}")
    print(f"FID Score: {float(fid_score):.4f} (Lower is Better)")
    if isinstance(is_out, (tuple, list)) and len(is_out) == 2:
        is_mean, is_std = is_out
        print(f"IS Score: {float(is_mean):.4f} ± {float(is_std):.4f} (Higher is Better)")
        is_score = is_mean
    else:
        is_score = is_out
        print(f"IS Score: {float(is_score):.4f} (Higher is Better)")
    print(f"{'='*50}\n")

    # save visualization
    if sample_real_list and sample_fake_list:
        real_samples = torch.cat(sample_real_list, dim=0)
        fake_samples = torch.cat(sample_fake_list, dim=0)
        save_image(real_samples, "results_gen/real_sample.png", nrow=8)
        save_image(fake_samples, "results_gen/fake_sample.png", nrow=8)
        print("Saved samples: results_gen/real_sample.png, results_gen/fake_sample.png")

    return fid_score, is_score


def main(args):
    lr = args['lr']
    is_eval = args['eval']
    z_size = args['z_size']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    ckpt_dir = args['ckpt_dir']
    data_dir = args['data_dir']
    save_per_epoch = args['save_per_epoch']

    print("Loading data...")

    train_loader, test_loader = get_dataloader(data_dir, batch_size=batch_size, im_size=IM_SIZE, channels=3, split_ratio=args.get('split_ratio', 0.9))
    print(f"Train set size: {len(train_loader.dataset)}, Test set size: {len(test_loader.dataset)}\n")

    config = {
        'ckpt_dir': ckpt_dir,
        'out_dir': args.get('out_dir', 'images'),
        'batch_size': batch_size,
        'z_size': z_size,
        'lr': lr,
        'num_epochs': num_epochs,
        'data_dir': data_dir,
        'b1': args.get('b1', 0.5),
        'b2': args.get('b2', 0.999),
        'ckpt_name': args.get('ckpt_name', 'best.pth'),
        'save_per_epoch': save_per_epoch,
        'sample_per_epoch': args.get('sample_per_epoch', 1),
    }

    if is_eval:
        fid_score, is_score = eval_dcgan(config, test_loader)
        print(f"Evaluation completed: FID={float(fid_score):.4f}, IS={float(is_score):.4f}")
        return

    best_epoch, min_loss, _ = train_dcgan(config, train_loader)
    print(f'Training finished: Best ckpt with G loss {min_loss:.6f} @ epoch {best_epoch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b1', type=float, default=0.5, help='beta1')
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=0.0002, help='lr')
    parser.add_argument('--z_size', type=int, default=100, help='latent z dim')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=200, help='num_epochs')
    parser.add_argument('--split_ratio', type=float, default=0.95, help='train/test split ratio')
    parser.add_argument('--save_per_epoch', type=int, default=100, help='save checkpoint per N epochs')
    parser.add_argument('--sample_per_epoch', type=int, default=20, help='save sample image per N epochs')
    parser.add_argument('--eval', action='store_true', help='eval')
    parser.add_argument('--ckpt_name', type=str, default='best.pth', help='ckpt_name')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='ckpt_dir')
    parser.add_argument('--out_dir', type=str, default='./images', help='sample output dir')
    parser.add_argument('--data_dir', type=str, default='/home/ycb410/ycb_ws/ECE285_HW3/datasets/', help='data_dir')

    main(vars(parser.parse_args()))