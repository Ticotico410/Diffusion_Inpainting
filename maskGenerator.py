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
        train_set: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.mask_size = mask_size

        self.image_paths = sorted(
            [p for p in self.root_dir.rglob("*") if p.suffix in IMG_EXTENSIONS]
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No jpg/jpeg images found under: {self.root_dir}")

        if train_set:
            transform_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02
                ),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        else:
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
    train_dir,
    test_dir,
    batch_size=64,
    img_size=256,
    mask_size=64,
    num_workers=8,
    pin_memory=True,
    normalize=True,
):
    train_set = InpaintingDataset(
        root_dir=train_dir,
        img_size=img_size,
        mask_size=mask_size,
        normalize=normalize,
    )

    test_set = InpaintingDataset(
        root_dir=test_dir,
        img_size=img_size,
        mask_size=mask_size,
        normalize=normalize,
        train_set=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return train_loader, val_loader



class InpaintingTensorDataset(Dataset):

    def __init__(self, pt_path, mask_size=32):
        data = torch.load(pt_path, map_location="cpu")
        self.images = data["images"]
        self.masks = data["masks"]
        self.masked_images = data["masked_images"]
        self.mask_size = mask_size

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # generate different mask
        img = self.images[idx]
        img_size = img.shape[2]
        mask_size = self.mask_size
        mask = torch.zeros((1, img_size, img_size))
        top = torch.randint(0, img_size - mask_size + 1, (1,)).item()
        left = torch.randint(0, img_size - mask_size + 1, (1,)).item()
        mask[:, top:top+mask_size, left:left+mask_size] = 1.0
        masked_image = img * (1 - mask)

        return {
            "image": img,
            "mask": mask,
            "masked_image": masked_image,
        }


def build_and_save_pt_dataset(
    root_dir,
    output_dir,
    img_size=128,
    mask_size=64,
    split_ratio=0.8,
    seed=42,
    normalize=True,
):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in root_dir.rglob("*") if p.suffix in IMG_EXTENSIONS]
    )

    if len(image_paths) == 0:
        raise RuntimeError("No images found")

    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize([0.5]*3, [0.5]*3)
        )
    transform = transforms.Compose(transform_list)

    generator = torch.Generator().manual_seed(seed)

    images = []
    masks = []
    masked_images = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = transform(image)
        # generate mask
        mask = torch.zeros((1, img_size, img_size))
        top = torch.randint(0, img_size - mask_size + 1, (1,), generator=generator).item()
        left = torch.randint(0, img_size - mask_size + 1, (1,), generator=generator).item()
        mask[:, top:top+mask_size, left:left+mask_size] = 1.0
        masked_image = image * (1 - mask)

        images.append(image)
        masks.append(mask)
        masked_images.append(masked_image)

    images = torch.stack(images)
    masks = torch.stack(masks)
    masked_images = torch.stack(masked_images)

    N = images.shape[0]
    train_size = int(N * split_ratio)

    perm = torch.randperm(N, generator=generator)

    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    train_data = {
        "images": images[train_idx],
        "masks": masks[train_idx],
        "masked_images": masked_images[train_idx],
    }

    test_data = {
        "images": images[test_idx],
        "masks": masks[test_idx],
        "masked_images": masked_images[test_idx],
    }

    torch.save(train_data, output_dir / "train.pt")
    torch.save(test_data, output_dir / "test.pt")

    print("Saved dataset")
    print("train:", train_data["images"].shape)
    print("test :", test_data["images"].shape)



def rewrite_dataset_and_save(original_path, output_path, mask_size=32, seed=42):
    original_dataset = torch.load(original_path)
    img_size = original_dataset["images"].shape[2]
    generator = torch.Generator().manual_seed(seed)

    images = []
    masks = []
    masked_images = []
    for img in original_dataset["images"]:
        # generate mask
        mask = torch.zeros((1, img_size, img_size))
        top = torch.randint(0, img_size - mask_size + 1, (1,), generator=generator).item()
        left = torch.randint(0, img_size - mask_size + 1, (1,), generator=generator).item()
        mask[:, top:top+mask_size, left:left+mask_size] = 1.0
        masked_image = img * (1 - mask)

        images.append(img)
        masks.append(mask)
        masked_images.append(masked_image)

    images = torch.stack(images)
    masks = torch.stack(masks)
    masked_images = torch.stack(masked_images)

    new_data = {
        "images": images,
        "masks": masks,
        "masked_images": masked_images
    }

    torch.save(new_data, output_path)
    print("Dataset Converted.", "Image shape:", images.shape)


if __name__ == "__main__":
    
    # build_and_save_pt_dataset(
    #     root_dir="./datasets/images",
    #     output_dir="./datasets/cached",
    #     img_size=128,
    #     mask_size=64,
    # )
    rewrite_dataset_and_save(
        original_path="./datasets/cached/train.pt",
        output_path="./datasets/cached/train-32.pt",
        mask_size=32
    )

    rewrite_dataset_and_save(
        original_path="./datasets/cached/test.pt",
        output_path="./datasets/cached/test-32.pt",
        mask_size=32
    )