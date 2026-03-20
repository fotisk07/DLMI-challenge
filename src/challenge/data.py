from __future__ import annotations

from typing import Callable, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 112


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        mode: Literal["train", "val", "test"],
        transform: Callable | None = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.hdf = None

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def _get_hdf(self):
        if self.hdf is None:
            self.hdf = h5py.File(self.dataset_path, "r")
        return self.hdf

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        hdf = self._get_hdf()

        img = torch.tensor(np.array(hdf[img_id]["img"])).float()
        label = torch.tensor(int(np.array(hdf[img_id]["label"]))) if self.mode != "test" else torch.tensor(-1)
        center = torch.tensor(int(np.array(hdf[img_id]["metadata"][0])))

        if self.transform:
            img = self.transform(img)

        return {
            "img": img,
            "label": label,
            "center": center,
        }


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_valid_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


TRAIN_TRANSFORMS_BASELINE = get_train_transforms()
VALID_TRANSFORMS = get_valid_transforms()
