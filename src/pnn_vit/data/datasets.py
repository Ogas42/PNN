from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from pnn_vit.config import DatasetConfig


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.base[index]
        path = None
        if hasattr(self.base, "samples"):
            path = self.base.samples[index][0]
        return {"image": image, "label": label, "index": index, "path": path}


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        wnids = [line.strip() for line in (root / "wnids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        val_dir = root / "val"
        image_dir = val_dir / "images"
        annotations_path = val_dir / "val_annotations.txt"
        if annotations_path.exists():
            for line in annotations_path.read_text(encoding="utf-8").splitlines():
                image_name, wnid, *_ = line.split("\t")
                self.samples.append((image_dir / image_name, self.class_to_idx[wnid]))
        else:
            folder_dataset = datasets.ImageFolder(val_dir, transform=transform)
            self.samples = [(Path(path), label) for path, label in folder_dataset.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_dataset(config: DatasetConfig, train: bool) -> Dataset:
    root = Path(config.root)
    transform = build_transforms(config.image_size, train)
    if config.name.lower() == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=train, download=config.download, transform=transform)
        return IndexedDataset(dataset)
    if config.name.lower() == "tiny_imagenet":
        if (root / "tiny-imagenet-200").exists():
            root = root / "tiny-imagenet-200"
        if train:
            dataset = datasets.ImageFolder(root / "train", transform=transform)
            return IndexedDataset(dataset)
        dataset = TinyImageNetValDataset(root, transform)
        return IndexedDataset(dataset)
    raise ValueError(f"Unsupported dataset: {config.name}")


def build_loaders(config: DatasetConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = build_dataset(config, train=True)
    val_dataset = build_dataset(config, train=False)
    probe_count = min(config.probe_samples, len(val_dataset))
    probe_dataset = Subset(val_dataset, range(probe_count))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, probe_loader
