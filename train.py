from __future__ import print_function

import os
import math
import argparse

import torch
from torch import optim
from torchvision.utils import save_image

from tqdm import tqdm
from ignite.metrics import FID, InceptionScore

from utils import viz_loss, denorm
from maskGenerator import get_dataloader
from policy import DiffusionInpaintPolicy


def build_policy(config, device):
    policy = DiffusionInpaintPolicy(
        pred_type=config["pred_type"],
        image_channels=3,
        mask_channels=1,
        bilinear=False,
        timesteps=config["timesteps"],
    ).to(device)
    return policy


def train(config, train_loader):
    lr = float(config["lr"])
    num_epochs = int(config["num_epochs"])
    save_per_epoch = int(config["save_per_epoch"])
    ckpt_dir = config["ckpt_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    policy = build_policy(config, device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print("Train set size:", len(train_loader.dataset))

    total_losses = []
    l1_losses = []
    perceptual_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        policy.train()

        running_total_loss = 0.0
        running_l1_loss = 0.0
        running_perc_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", unit="batch")

        for batch in pbar:
            x0 = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            masked_image = batch["masked_image"].to(device, non_blocking=True)

            # Setup optimizer
            optimizer.zero_grad()

            # Compute loss
            loss_dict = policy.compute_loss(masked_image=masked_image, mask=mask, x0=x0)
            l1 = loss_dict["l1"]
            perc = loss_dict["perc"]
            total_loss = l1 + perc

            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()
            running_l1_loss += l1.item()
            running_perc_loss += perc.item()

            pbar.set_postfix(total=f"{total_loss.item():.6f}", l1=f"{l1.item():.6f}", perc=f"{perc.item():.6f}")

        avg_total_loss = running_total_loss / len(train_loader)
        avg_l1_loss = running_l1_loss / len(train_loader)
        avg_perc_loss = running_perc_loss / len(train_loader)

        total_losses.append(avg_total_loss)
        l1_losses.append(avg_l1_loss)
        perceptual_losses.append(avg_perc_loss)

        print(f"Epoch {epoch + 1} | total: {avg_total_loss:.6f}, l1: {avg_l1_loss:.6f}, perc: {avg_perc_loss:.6f}")

        if (epoch + 1) % save_per_epoch == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "policy_state_dict": policy.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    torch.save(
        {
            "epoch": num_epochs,
            "policy_state_dict": policy.state_dict(),
            "config": config,
        },
        best_ckpt_path,
    )
    print(f"Saved final checkpoint: {best_ckpt_path}")

    viz_loss(total_losses, num_epochs, ckpt_dir, name="total_loss")
    viz_loss(l1_losses, num_epochs, ckpt_dir, name="l1_loss")
    viz_loss(perceptual_losses, num_epochs, ckpt_dir, name="perceptual_loss")

    return num_epochs, {
        "total": total_losses,
        "l1": l1_losses,
        "perc": perceptual_losses,
    }


@torch.no_grad()
def eval(config, test_loader):
    ckpt_dir = config["ckpt_dir"]
    ckpt_name = config.get("ckpt_name", "best.pth")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    policy = build_policy(config, device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Test set size: {len(test_loader.dataset)}")

    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device)

    os.makedirs("results_gen", exist_ok=True)

    sample_real_list = []
    sample_fake_list = []

    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        x0 = batch["image"].to(device, non_blocking=True)               # real image, [-1,1]
        mask = batch["mask"].to(device, non_blocking=True)
        masked_image = batch["masked_image"].to(device, non_blocking=True)

        # full reverse diffusion sampling
        x0_pred = policy.predict_x0(masked_image=masked_image, mask=mask)   # [-1,1]

        real_imgs = denorm(x0)
        fake_imgs = denorm(x0_pred)

        # Ignite FID: update((y_pred, y))
        fid_metric.update((fake_imgs, real_imgs))
        is_metric.update(fake_imgs)

        if batch_idx == 0:
            sample_real_list.append(real_imgs[:8].cpu())
            sample_fake_list.append(fake_imgs[:8].cpu())

    fid_score = fid_metric.compute()
    is_score = is_metric.compute()

    if sample_real_list and sample_fake_list:
        real_samples = torch.cat(sample_real_list, dim=0)
        fake_samples = torch.cat(sample_fake_list, dim=0)
        save_image(real_samples, "results_gen/real_sample.png", nrow=4)
        save_image(fake_samples, "results_gen/fake_sample.png", nrow=4)
        print("Saved samples: results_gen/real_sample.png, results_gen/fake_sample.png")

    return fid_score, is_score


def main(args):
    config = {
        "lr": args["lr"],
        "batch_size": args["batch_size"],
        "num_epochs": args["num_epochs"],
        "split_ratio": args["split_ratio"],
        "save_per_epoch": args["save_per_epoch"],
        "ckpt_dir": args["ckpt_dir"],
        "ckpt_name": args["ckpt_name"],
        "data_dir": args["data_dir"],
        "img_size": args["img_size"],
        "mask_size": args["mask_size"],
        "pred_type": args["pred_type"],
        "timesteps": args["timesteps"],
        "num_workers": args["num_workers"],
    }

    print("Loading data...")
    train_loader, test_loader = get_dataloader(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        mask_size=config["mask_size"],
        split_ratio=config["split_ratio"],
        num_workers=config["num_workers"],
    )

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}\n")

    if args["eval"]:
        fid_score, is_score = eval(config, test_loader)
        print(f"Evaluation completed: FID={float(fid_score):.4f}, IS={float(is_score):.4f}")
        return

    best_epoch, epoch_losses = train(config, train_loader)
    print(
        f"Training finished: "
        f"last total loss {epoch_losses['total'][-1]:.6f}, "
        f"last l1 loss {epoch_losses['l1'][-1]:.6f}, "
        f"last perc loss {epoch_losses['perc'][-1]:.6f} "
        f"@ epoch {best_epoch}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--save_per_epoch", type=int, default=10)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--mask_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--pred_type", type=str, default="x0", choices=["x0", "eps"])

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--ckpt_name", type=str, default="best.pth")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    main(vars(parser.parse_args()))