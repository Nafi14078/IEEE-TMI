import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test", "external")


def lab_column(label_fraction, split_seed) -> str:
    return f"lab{label_fraction}_s{split_seed}"


def load_splits(cfg):
    """Return (df, {split: row indices}, labelled row indices)."""
    df = pd.read_csv(cfg["data"]["csv"])
    col = lab_column(cfg["data"]["label_fraction"], cfg["data"]["split_seed"])
    if col not in df.columns:
        raise KeyError(f"{col} not in {cfg['data']['csv']}; re-run scripts/prepare_splits.py")
    idx = {s: np.where(df["split"].values == s)[0] for s in SPLITS}
    labelled = np.where((df["split"].values == "train") & (df[col].values == 1))[0]
    return df, idx, labelled


class RandomRot90:
    """Rotate by a random multiple of 90 degrees (lesions have no canonical orientation)."""

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k, expand=True) if k else img


def student_transform(img_size: int, train: bool):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            RandomRot90(),
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize(int(img_size * 256 / 224)), T.CenterCrop(img_size), T.ToTensor(), norm])


class ImageTable(Dataset):
    """Images addressed by *global row index* of the splits CSV (same order as the teacher cache)."""

    def __init__(self, df, image_root, indices, transform):
        self.paths = [os.path.join(image_root, p) for p in df["path"].values]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=num_workers > 0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch
