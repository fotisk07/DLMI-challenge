import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T

import argparse
import csv
from pathlib import Path

from tqdm.auto import tqdm

from challenge.data import PatchDataset
from challenge import stain
from challenge import builder
from challenge import preprocessing
from challenge import compute

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def build_val_transform() -> T.Compose:
    """Deterministic centre-crop pipeline for validation / inference."""
    val_transform = T.Compose(
        [
            T.Normalize(mean=MEAN, std=STD)
        ]
    )
    
    return val_transform

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on binary classification.")

    parser.add_argument("checkpoint", type=Path,
                        help="Directory where model checkpoints are saved.")
    parser.add_argument("--test-path", type=Path, default="data/test.h5",
                        help="Path to training .h5 file.")
    parser.add_argument("--val-path", type=Path, default="data/val.h5",
                        help="Path to training .h5 file.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size for both train and val loaders.")
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/predictions.csv"))

    # ── Infrastructure ────────────────────────────────────────────────────────
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader worker processes.")

    return parser.parse_args()


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    prediction_path: Path,
) -> None:
    model.eval()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Pred"])

        with torch.no_grad():
            for images, image_id in tqdm(dataloader):
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu().tolist()

                for image_id, prediction in zip(image_id, predictions):
                    writer.writerow([image_id.cpu().item(), prediction])

def build_model(arch, checkpoints) -> nn.Module:
    model = builder.build_model(arch, pretrained=False)
    
    print(f'loading {arch}')
    
    model.load_state_dict(checkpoints)

    return model

def build_normalizer(method, path):
    if method == 'macenko':
        normalizer = stain.MacenkoNormalizer.load(path)
        print('Using macenko normalizer\n')
    else:
        normalizer = T.Lambda(lambda x:x)
    
    return normalizer

def build(checkpoint):
    
    model_data = torch.load(str(checkpoint), weights_only=False)
    
    model = build_model(model_data['arch'], model_data['model'])
    normalizer = build_normalizer(*model_data['normalizer'])
    
    return model, normalizer



class TTA:
    
    def __init__(self, augmentor, n):
        self.augmentor = augmentor
        self.n = n
    
    def __call__(self, x):
        augs = []
        
        for _ in range(self.n):
            aug = self.augmentor(x)
            augs.append(aug)
        
        return torch.stack(augs)
    
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool = True,
) -> tuple[float, float]:
    """Run one epoch. Returns (mean_loss, accuracy). Works with CutMix or not."""
    
    model.train(is_train)
    total_loss = 0.0

    # Torchmetrics-style counters
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if is_train else "val ", leave=False):
            images, labels = batch
            images = images.to(device)

            labels = labels.to(device)

            B, N, C, H, W = images.shape
            
            logits = model(images.reshape(B*N, C, H, W)).reshape(B, N, 2)
            logits = logits.mean(dim=1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total



def main() -> None:
    args = parse_args()

    # Reproducibility


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Stain normaliser ──────────────────────────────────────────────────────
    model, normalizer = build(args.checkpoint)
    model.eval()
    model = model.to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    
    def collate_fn(batch, transform):
        images = torch.stack([transform(item['img']) for item in batch])
        image_ids = torch.tensor([int(item['img_id']) for item in batch])
        
        return images, image_ids
    
    transform_ = T.Compose([TTA(stain.HEAugmentor(0.7, 0.7), 3), build_val_transform()])
    
    transform = build_val_transform()
    
    val_dataset = PatchDataset(str(args.val_path), mode="val", transform=normalizer)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda x : preprocessing.base_collate(x, transform)
    )
    
    
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss, val_acc = compute.run_epoch(
            model, val_dataloader, criterion, optimizer=None, device=device,
            is_train=False,
        )
    
    print(f'val accuracy : {val_acc}')

    test_dataset = PatchDataset(str(args.test_path), mode="test", transform=normalizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda x : collate_fn(x, transform)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    
    generate_predictions(
        model=model,
        dataloader=test_dataloader,
        device=device,
        prediction_path=args.prediction_path,
    )
    print(f"Saved predictions to {args.prediction_path}")


if __name__ == "__main__":
    main()