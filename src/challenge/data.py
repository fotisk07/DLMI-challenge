from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        img_id = self.image_ids[idx]
        hdf = self._get_hdf()

        img = torch.tensor(np.array(hdf[img_id]["img"])).float()

        if self.mode != "test":
            label = torch.tensor(int(np.array(hdf[img_id]["label"])))
            center = torch.tensor(int(np.array(hdf[img_id]["metadata"][0])))
        else:
            label = torch.tensor(-1)
            center = torch.tensor(-1)

        if self.transform:
            img = self.transform(img)

        return {
            "img": img,
            "label": label,
            "center": center,
        }


class SavedEmbeddingsDataset(Dataset):
    def __init__(self, embds_path: str):
        data = torch.load(embds_path)
        self.features = data["features"]
        self.labels = data["labels"]
        self.centers = data["centers"]
        self.image_ids = data["image_ids"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        return {
            "embeddings": self.features[idx],
            "label": self.labels[idx],
            "center": self.centers[idx],
            "image_id": self.image_ids[idx],
        }


@dataclass
class Normalizer:
    center_norm: bool
    l2_norm: bool
    _stats: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None  # center_id -> (mean, std)
    _global_stats: tuple[torch.Tensor, torch.Tensor] | None = None  # (mean, std) for unknown centers

    def fit(self, features: torch.Tensor, centers: torch.Tensor | None = None, eps: float = 1e-6) -> None:
        if not self.center_norm:
            return

        # always fit global stats as fallback
        mean = features.mean(dim=0)
        std = features.std(dim=0).clamp(min=eps)
        self._global_stats = (mean, std)

        if centers is None:
            return

        unique_centers, inverse = torch.unique(centers, return_inverse=True)
        K, D = unique_centers.size(0), features.size(1)

        means = torch.zeros(K, D).index_add_(0, inverse, features)
        counts = torch.zeros(K).index_add_(0, inverse, torch.ones(len(centers)))
        means /= counts.unsqueeze(1)

        stds = torch.zeros(K, D).index_add_(0, inverse, (features - means[inverse]) ** 2)
        stds = torch.sqrt(stds / counts.unsqueeze(1) + eps)

        self._stats = {int(c): (means[i], stds[i]) for i, c in enumerate(unique_centers)}

    def __call__(self, embeddings: torch.Tensor, centers: torch.Tensor | None = None) -> torch.Tensor:
        if self.center_norm:
            if centers is not None and self._stats is not None:
                out = embeddings.clone()
                for i, c in enumerate(centers.tolist()):
                    mu, std = self._stats[int(c)]
                    out[i] = (embeddings[i] - mu) / std
                embeddings = out
            else:
                mu, std = self._global_stats  # ty:ignore[not-iterable]
                embeddings = (embeddings - mu) / std

        if self.l2_norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class Normalisation:
    def __init__(self, use_center_norm=True, use_l2_norm=False, eps=1e-6):
        self.use_center_norm = use_center_norm
        self.use_l2_norm = use_l2_norm
        self.eps = eps

        self.means = None  # (K, D)
        self.stds = None  # (K, D)
        self.center_ids = None  # (K,)

    def fit(self, features: torch.Tensor, centers: torch.Tensor):
        unique_centers, inverse = torch.unique(centers, return_inverse=True)
        K, D = unique_centers.size(0), features.size(1)

        device, dtype = features.device, features.dtype

        means = torch.zeros(K, D, device=device, dtype=dtype)
        counts = torch.zeros(K, device=device, dtype=dtype)

        means.index_add_(0, inverse, features)
        counts.index_add_(0, inverse, torch.ones_like(centers, dtype=dtype))

        means = means / counts.unsqueeze(1)

        centered = features - means[inverse]
        var = torch.zeros(K, D, device=device, dtype=dtype)
        var.index_add_(0, inverse, centered**2)
        stds = torch.sqrt(var / counts.unsqueeze(1) + self.eps)

        self.means = means
        self.stds = stds
        self.center_ids = unique_centers

        return self

    def _l2(self, x):
        if not self.use_l2_norm:
            return x
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

    def transform(self, embeddings: torch.Tensor, centers: torch.Tensor):
        out = embeddings

        if self.use_center_norm:
            # map each center → index in [0, K)
            idx = (centers.unsqueeze(1) == self.center_ids).nonzero()[:, 1]
            out = (embeddings - self.means[idx]) / self.stds[idx]  # ty:ignore[not-subscriptable]

        return self._l2(out)

    def transform_unknown(self, embeddings: torch.Tensor):
        out = embeddings

        if self.use_center_norm:
            test_mean = embeddings.mean(dim=0)

            # find closest center mean
            dists = torch.norm(self.means - test_mean, dim=1)
            best_idx = torch.argmin(dists)

            mu = self.means[best_idx]  # ty:ignore[not-subscriptable]
            std = self.stds[best_idx]  # ty:ignore[not-subscriptable]

            out = (embeddings - mu) / std

        return self._l2(out)


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=MEAN, std=STD),
        ],
    )


def get_valid_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=MEAN, std=STD),
        ],
    )


TRAIN_TRANSFORMS_BASELINE = get_train_transforms()
VALID_TRANSFORMS = get_valid_transforms()
