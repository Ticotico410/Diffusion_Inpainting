from __future__ import print_function

import os
import math
import argparse

import torch
from torch import optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from tqdm import tqdm
import yaml
from ignite.metrics import FID, InceptionScore

from utils import viz_loss, denorm, masked_psnr, masked_ssim, masked_lpips
from maskGenerator import get_dataloader, InpaintingTensorDataset
from policy import DiffusionInpaintPolicy, DirectPredictPolicy


def build_policy(config, device):
    if config["policy_type"] == "diffusion":
        policy = DiffusionInpaintPolicy(
            pred_type=config["pred_type"],
            image_channels=3,
            mask_channels=1,
            bilinear=False,
            timesteps=config["timesteps"],
            no_skip=config["no_skip"],
        ).to(device)
    elif config["policy_type"] == "direct":
        policy = DirectPredictPolicy(image_channels=3, mask_channels=1, bilinear=False, no_skip=config["no_skip"]).to(device)
    else:
        raise NotImplementedError
    
    return policy


def train(config, train_loader):
    lr = float(config["lr"])
    num_epochs = int(config["num_epochs"])
    save_per_epoch = int(config["save_per_epoch"])

    ckpt_dir = os.path.join(config["ckpt_dir"], config["task_name"])
    lambda_perceptual = float(config["perceptual_weight"])

    os.makedirs(ckpt_dir, exist_ok=True)

    # save config
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

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
            total_loss = l1 + lambda_perceptual * perc

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
    task_name = config["task_name"]
    ckpt_name = config.get("ckpt_name", "best.pth")
    ckpt_path = os.path.join(ckpt_dir, task_name, ckpt_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    policy = build_policy(config, device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Test set size: {len(test_loader.dataset)}")

    os.makedirs("results_gen", exist_ok=True)

    sample_real_list = []
    sample_fake_list = []

    psnr_list = []
    ssim_list = []
    lpips_list = []

    traj = input_traj = None

    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        x0 = batch["image"].to(device, non_blocking=True)               # real image, [-1,1]
        mask = batch["mask"].to(device, non_blocking=True)
        masked_image = batch["masked_image"].to(device, non_blocking=True)

        # full reverse diffusion sampling
        x0_pred = policy.predict_x0(masked_image=masked_image, mask=mask)   # [-1,1]

        real_imgs = denorm(x0)
        fake_imgs = denorm(x0_pred)

        psnr = masked_psnr(fake_imgs, real_imgs, mask)
        ssim = masked_ssim(fake_imgs, real_imgs, mask)
        lpips = masked_lpips(fake_imgs, real_imgs, mask)

        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips.item())

        if batch_idx == 0:
            if config["policy_type"] == "diffusion":
                x0_pred, traj, input_traj = policy.predict_x0(masked_image=masked_image, mask=mask, return_trajectory=True) 
            else:
                x0_pred = policy.predict_x0(masked_image=masked_image, mask=mask) 
            sample_real_list.append(real_imgs[:8].cpu())
            sample_fake_list.append(fake_imgs[:8].cpu())

    psnr_score = sum(psnr_list) / len(psnr_list)
    ssim_score = sum(ssim_list) / len(ssim_list)
    lpips_score = sum(lpips_list) / len(lpips_list)

    if sample_real_list and sample_fake_list:
        real_samples = torch.cat(sample_real_list, dim=0)
        fake_samples = torch.cat(sample_fake_list, dim=0)
        traj_sample = []
        input_traj_sample = []

        if traj and input_traj:
            for i in traj:
                traj_sample.append(denorm(i)[0].cpu())
            for i in input_traj:
                input_traj_sample.append(denorm(i)[0].cpu())
            sample_traj = torch.stack(traj_sample, dim=0)
            input_traj = torch.stack(input_traj_sample, dim=0)
            save_image(sample_traj, "results_gen/sample_traj.png", nrow=10)
            save_image(input_traj, "results_gen/input_traj.png", nrow=10)

        save_image(real_samples, "results_gen/real_sample.png", nrow=4)
        save_image(fake_samples, "results_gen/fake_sample.png", nrow=4)
        print("Saved samples: results_gen/real_sample.png, results_gen/fake_sample.png")

    return psnr_score, ssim_score, lpips_score


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
        "task_name": args["task_name"],
        "perceptual_weight": args["perceptual_weight"],
        "cached_dir": args["cached_dir"],
        "policy_type": args["policy_type"],
        "no_skip": args["no_skip"],
    }

    print("Loading data...")
    train_dataset_path = os.path.join(args["cached_dir"], "train-32.pt")
    test_dataset_path = os.path.join(args["cached_dir"], "test-32.pt")
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        print("Loading data from pre processed .pt file")
        train_dataset = InpaintingTensorDataset(train_dataset_path)
        test_dataset = InpaintingTensorDataset(test_dataset_path)
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    else:
        print("Loading data from original file")
        train_loader, test_loader = get_dataloader(
            train_dir=config["data_dir"],
            test_dir="./datasets/testset",
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            mask_size=config["mask_size"],
            num_workers=config["num_workers"],
        )

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}\n")

    if args["eval"]:
        psnr_score, ssim_score, lpips_score = eval(config, test_loader)
        print(f"Evaluation completed: PSNR={float(psnr_score):.4f}, SSIM={float(ssim_score):.4f}, LPIPS={float(lpips_score)}")
        return

    best_epoch, epoch_losses = train(config, train_loader)
    print(
        f"Training finished: "
        f"last total loss {epoch_losses['total'][-1]:.6f}, "
        f"last l1 loss {epoch_losses['l1'][-1]:.6f}, "
        f"last perc loss {epoch_losses['perc'][-1]:.6f} "
        f"@ epoch {best_epoch}"
    )

    psnr_score, ssim_score, lpips_score = eval(config, test_loader)
    print(f"Evaluation completed: PSNR={float(psnr_score):.4f}, SSIM={float(ssim_score):.4f}, LPIPS={float(lpips_score)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--save_per_epoch", type=int, default=20)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--mask_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--pred_type", type=str, default="x0", choices=["x0", "eps"])
    parser.add_argument("--perceptual_weight", type=float, default=0.05)
    parser.add_argument("--policy_type", default="diffusion", choices=["diffusion", "direct"])
    parser.add_argument("--no_skip", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--ckpt_name", type=str, default="best.pth")
    parser.add_argument("--data_dir", type=str, default="./datasets/coast")
    parser.add_argument("--cached_dir", type=str, default="./datasets/cached")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--task_name", type=str, default="diff-inpainting")

    main(vars(parser.parse_args()))