import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from challenge.data import VALID_TRANSFORMS, PatchDataset

MODEL_NAME = "dinov2_vits14"
INPUT = Path("data/val.h5")
OUTPUT_DIR = Path("data/embeddings")
BATCH_SIZE = 64
NUM_WORKERS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mode = INPUT.stem


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(DEVICE)
model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).to(device)
model.eval()

print(f"Using {device}")

dataset = PatchDataset(str(INPUT), mode=mode, transform=VALID_TRANSFORMS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

features = []
labels = []
centers = []

for batch in tqdm(dataloader, desc=INPUT.stem):
    images = batch["img"].to(device, non_blocking=True)
    with torch.no_grad():
        features.append(model(images).cpu())

    labels.append(batch["label"].cpu())
    centers.append(batch["center"].cpu())

save_path = OUTPUT_DIR / f"{INPUT.stem}_{MODEL_NAME}_embeddings.pt"
torch.save(
    {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "centers": torch.cat(centers, dim=0),
        "image_ids": dataset.image_ids,
        "model_name": MODEL_NAME,
        "mode": mode,
    },
    save_path,
)
print(f"Saved embeddings to {save_path}")
