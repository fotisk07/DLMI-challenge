from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from challenge.data import Normalizer, SavedEmbeddingsDataset


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear probe on saved embeddings.")
    parser.add_argument(
        "--train-embeddings",
        type=Path,
        default=Path("data/embeddings/train_dinov2_vits14_embeddings.pt"),
    )
    parser.add_argument(
        "--val-embeddings",
        type=Path,
        default=Path("data/embeddings/val_dinov2_vits14_embeddings.pt"),
    )
    parser.add_argument(
        "--predict-embeddings",
        type=Path,
        default=Path("data/embeddings/test_dinov2_vits14_embeddings.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--center-norm", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--l2-norm", action="store_true")
    parser.add_argument("--save-path", type=Path, default=Path("outputs/linear_probe.pt"))
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/linear_probe_predictions.csv"))
    parser.add_argument(
        "--loocv",
        action="store_true",
        help="Run leave-one-center-out cross validation instead of normal training.",
    )
    return parser.parse_args()


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    normalizer: Normalizer,
    train: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = batch["embeddings"].float()
    labels = batch["label"].long()
    centers = batch["center"]
    embeddings = normalizer(embeddings, centers) if train else normalizer(embeddings)
    return embeddings.to(device), labels.to(device)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    is_val: bool = False,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        embeddings, labels = prepare_batch(batch, device, normalizer, train=not is_val)

        with torch.set_grad_enabled(is_train):
            logits = model(embeddings)
            loss = F.cross_entropy(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    return EpochMetrics(loss=total_loss / total_examples, accuracy=total_correct / total_examples)


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: Normalizer,
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    held_out_center: int | None = None,
) -> float:
    model = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_accuracy = -1.0

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_metrics = run_epoch(
            model,
            train_loader,
            normalizer=normalizer,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, normalizer=normalizer, device=device, is_val=True)

        # prefix = f"[held out center {held_out_center}] " if held_out_center is not None else ""
        # print(
        #     f"{prefix}Epoch {epoch:02d}/{args.epochs} | "
        #     f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
        #     f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}",
        # )

        best_val_accuracy = max(best_val_accuracy, val_metrics.accuracy)

    return best_val_accuracy


def run_loocv(dataset: SavedEmbeddingsDataset, args: argparse.Namespace, device: torch.device) -> None:
    centers = dataset.centers  # (N,)
    unique_centers = centers.unique().tolist()
    input_dim = int(dataset[0]["embeddings"].numel())

    fold_accuracies = []

    for held_out in unique_centers:
        print(f"\n{'=' * 60}")
        print(f"LOOCV fold: holding out center {int(held_out)}")
        print(f"{'=' * 60}")

        train_idx = (centers != held_out).nonzero(as_tuple=True)[0].tolist()
        val_idx = (centers == held_out).nonzero(as_tuple=True)[0].tolist()

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # fit normalizer only on this fold's training data
        train_features = dataset.features[train_idx]
        train_centers = centers[train_idx]
        normalizer = Normalizer(args.center_norm, args.l2_norm)
        normalizer.fit(train_features, train_centers)

        # for the held-out center, re-fit normalizer on val features (self-normalization)
        val_features = dataset.features[val_idx]
        normalizer.fit(val_features)  # fits global_stats for unknown center

        best_acc = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            normalizer=normalizer,
            input_dim=input_dim,
            device=device,
            args=args,
            held_out_center=int(held_out),
        )

        print(f"Center {int(held_out)} best val accuracy: {best_acc:.4f}")
        fold_accuracies.append(best_acc)

    print(f"\n{'=' * 60}")
    print("LOOCV results:")
    for center, acc in zip(unique_centers, fold_accuracies):
        print(f"  center {int(center)}: {acc:.4f}")
    print(f"  mean: {sum(fold_accuracies) / len(fold_accuracies):.4f}")
    print(f"{'=' * 60}")


def infer_dimensions(dataloader: DataLoader) -> int:
    sample = dataloader.dataset[0]
    return int(sample["embeddings"].numel())


def save_checkpoint(
    save_path: Path,
    model: LinearProbe,
    args: argparse.Namespace,
    input_dim: int,
    metrics: EpochMetrics,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "normalization": {"center_norm": args.center_norm, "l2_norm": args.l2_norm},
            "train_embeddings": str(args.train_embeddings),
            "val_embeddings": str(args.val_embeddings),
            "metrics": {"val_loss": metrics.loss, "val_accuracy": metrics.accuracy},
        },
        save_path,
    )


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalizer: Normalizer,
    prediction_path: Path,
) -> None:
    model.eval()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Pred"])

        with torch.no_grad():
            for batch in dataloader:
                embeddings, _ = prepare_batch(batch, device=device, normalizer=normalizer, train=False)
                logits = model(embeddings)
                predictions = logits.argmax(dim=1).cpu().tolist()

                for image_id, prediction in zip(batch["image_id"], predictions):
                    writer.writerow([image_id, prediction])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SavedEmbeddingsDataset(args.train_embeddings)
    val_dataset = SavedEmbeddingsDataset(args.val_embeddings)

    if args.loocv:
        run_loocv(train_dataset, args, device)
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    if args.full_train:
        fit_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        train_loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = None  # no validation in full-train mode

        # concat features and centers for normalizer fitting
        all_features = torch.cat([train_dataset.features, val_dataset.features])
        all_centers = torch.cat([train_dataset.centers, val_dataset.centers])
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        all_features = train_dataset.features
        all_centers = train_dataset.centers

    normalizer = Normalizer(args.center_norm, args.l2_norm)
    normalizer.fit(all_features, all_centers)

    input_dim = infer_dimensions(train_loader)
    model = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training on {args.train_embeddings} | Input dim: {input_dim} | Device: {device}")

    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            normalizer=normalizer,
            device=device,
            optimizer=optimizer,
        )
        if val_loader is not None:
            
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, normalizer=normalizer, device=device, is_val=True)
            
            best_val_accuracy = max(val_metrics.accuracy, best_val_accuracy)
            
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
                f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}",
            )
        else:
            print(
                f"Epoch {epoch:02d} | train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f}",
            )
    
    

    save_checkpoint(
        save_path=args.save_path,
        model=model,
        args=args,
        input_dim=input_dim,
        metrics=val_metrics,
    )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    if args.predict:
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_dataset = SavedEmbeddingsDataset(args.predict_embeddings)
        normalizer.fit(test_dataset.features)  # self-normalize test set

        prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        generate_predictions(
            model=model,
            dataloader=prediction_loader,
            device=device,
            normalizer=normalizer,
            prediction_path=args.prediction_path,
        )
        print(f"Saved predictions to {args.prediction_path}")


if __name__ == "__main__":
    main()
