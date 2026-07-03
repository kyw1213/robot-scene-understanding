from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class MITIndoor67Dataset(Dataset):
    def __init__(self, dataset_root, split_file, image_size=224, train=True):
        self.dataset_root = Path(dataset_root)
        self.image_root = self.dataset_root / "Images"
        self.split_path = self.dataset_root / split_file
        self.transform = build_transforms(image_size, train)

        self.classes = sorted([p.name for p in self.image_root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.samples = []
        with open(self.split_path, "r") as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue

                class_name = rel_path.split("/")[0]
                image_path = self.image_root / rel_path
                label = self.class_to_idx[class_name]
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
