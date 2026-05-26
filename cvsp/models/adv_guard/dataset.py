from __future__ import annotations

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from pathlib import Path


class AdvGuardDataset(Dataset):
    def __init__(
        self, manifest_path: Path, data_root: Path, split: str, augment: bool = False
    ):
        df = pd.read_parquet(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.data_root = Path(data_root)
        self.augment = augment

        if augment:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
            )
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.data_root / row["path"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row["label"], dtype=torch.long)

        return x, y
