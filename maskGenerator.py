import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


IMG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
IMG_SIZE = 128
MASK_SIZE = 64


class InpaintingDataset(Dataset):
    """
    Simple inpainting dataset.

    Returns:
        image: [3, H, W]
        mask: [1, H, W], 1 means masked region
        masked_image: [3, H, W]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: int = IMG_SIZE,
        mask_size: int = MASK_SIZE,
        normalize: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.mask_size = mask_size

        self.image_paths = sorted(
            [p for p in self.root_dir.rglob("*") if p.suffix in IMG_EXTENSIONS]
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No jpg/jpeg images found under: {self.root_dir}")

        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def generate_mask(self) -> torch.Tensor:
        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        top = torch.randint(0, self.img_size - self.mask_size + 1, (1,)).item()
        left = torch.randint(0, self.img_size - self.mask_size + 1, (1,)).item()

        mask[:, top:top + self.mask_size, left:left + self.mask_size] = 1.0
        return mask

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        mask = self.generate_mask()
        masked_image = image * (1.0 - mask)

        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image
        }


def get_dataloader(
    root_dir,
    batch_size=64,
    img_size=256,
    mask_size=64,
    split_ratio=0.7,
    seed=42,
    num_workers=8,
    pin_memory=True,
    normalize=True,
):
    dataset = InpaintingDataset(
        root_dir=root_dir,
        img_size=img_size,
        mask_size=mask_size,
        normalize=normalize,
    )

    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(
        root_dir="/home/ycb410/ycb_ws/Diffusion_Inpainting/datasets",
        batch_size=4,
        img_size=256,
        mask_size=64,
        split_ratio=0.9,
    )

    batch = next(iter(train_loader))

    images = batch["image"]
    masks = batch["mask"]
    masked_images = batch["masked_image"]

    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for i in range(4):
        img = denorm(images[i]).permute(1, 2, 0).cpu().numpy()
        msk = masks[i, 0].cpu().numpy()
        mimg = denorm(masked_images[i]).permute(1, 2, 0).cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(msk, cmap="gray")
        axes[i, 1].set_title("mask")
        axes[i, 2].imshow(mimg)
        axes[i, 2].set_title("masked_image")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()