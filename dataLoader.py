import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.utils import save_image


class WikiArtDataset(Dataset):
    """
    Read images recursively from a folder.
    Returns: (img_tensor, 0)
    img_tensor is normalized to [-1, 1] to match GAN Tanh output.
    """
    def __init__(self, data_dir, im_size=64, channels=3, recursive=True, transform=None):
        self.data_dir = data_dir
        self.im_size = im_size
        self.channels = channels

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        self.image_paths = []
        for ext in exts:
            if recursive:
                self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
                self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext.upper()), recursive=True))
            else:
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext.upper())))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found under: {data_dir}")

        print(f"Found {len(self.image_paths)} images in {data_dir}")

        if transform is not None:
            self.transform = transform
        else:
            if channels == 3:
                mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            else:
                mean, std = (0.5,), (0.5,)

            resize_size = int(im_size * 1.125)

            self.transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        try:
            img = Image.open(p)
            img = img.convert("RGB") if self.channels == 3 else img.convert("L")
            img = self.transform(img)
            return img, 0
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
            c = 3 if self.channels == 3 else 1
            return torch.zeros(c, self.im_size, self.im_size), 0


def get_dataloader(
    data_dir,
    batch_size=64,
    im_size=64,
    channels=3,
    split_ratio=1.0,
    seed=42,
    num_workers=8,
    pin_memory=True
):
    dataset = WikiArtDataset(data_dir=data_dir, im_size=im_size, channels=channels, recursive=True)

    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader

# if __name__ == "__main__":
#     data_dir = "/home/ycb410/ycb_ws/ECE285_HW3/datasets/"
#     loader = get_dataloader(data_dir, batch_size=16, im_size=64, channels=3, num_workers=0, pin_memory=False)
#     imgs, _ = next(iter(loader))
#     os.makedirs("results_dataloader", exist_ok=True)
#     save_image((imgs * 0.5 + 0.5), "results_dataloader/train_batch_grid.png", nrow=4)
#     print("Batch shape:", imgs.shape, "range:", imgs.min().item(), imgs.max().item())


if __name__ == "__main__":
    import os
    from torchvision.utils import save_image

    data_dir = "/home/ycb410/ycb_ws/ECE285_HW3/datasets/"

    train_loader, val_loader = get_dataloader(
        data_dir,
        batch_size=16,
        im_size=64,
        channels=3,
        num_workers=0,
        pin_memory=False
    )

    imgs, _ = next(iter(train_loader))

    os.makedirs("results_dataloader", exist_ok=True)
    save_image((imgs * 0.5 + 0.5).clamp(0, 1), "results_dataloader/train_batch_grid.png", nrow=4)

    print("Saved: results_dataloader/train_batch_grid.png")
    print("Batch shape:", imgs.shape, "range:", imgs.min().item(), imgs.max().item())