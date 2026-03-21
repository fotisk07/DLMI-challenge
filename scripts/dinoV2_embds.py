from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from challenge.data import STAIN_TRANSFORMS, VALID_TRANSFORMS, PatchDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .h5 file.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta-views", type=int, default=1, help="Augmented views to average. 1 = no TTA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.input.stem  # "train", "val", "test"

    print(f"Device: {device} | Input: {args.input} | TTA views: {args.tta_views}")

    model = torch.hub.load("facebookresearch/dinov2", args.model).to(device)
    model.eval()

    # always load without stain augmentation — TTA applies it per view
    dataset = PatchDataset(str(args.input), mode=mode, transform=VALID_TRANSFORMS)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_features, all_labels, all_centers = [], [], []

    for batch in tqdm(dataloader):
        images = batch["img"].to(device, non_blocking=True)

        with torch.no_grad():
            if args.tta_views == 1:
                features = model(images)
            else:
                views = torch.stack(
                    [
                        model(torch.stack([STAIN_TRANSFORMS(img.cpu()) for img in images]).to(device))
                        for _ in range(args.tta_views)
                    ]
                )  # (K, B, D)
                features = views.mean(dim=0)  # (B, D)

        all_features.append(features.cpu())
        all_labels.append(batch["label"])
        all_centers.append(batch["center"])

    suffix = f"_tta{args.tta_views}" if args.tta_views > 1 else ""
    save_path = args.output_dir / f"{mode}_{args.model}{suffix}_embeddings.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": torch.cat(all_features),
            "labels": torch.cat(all_labels),
            "centers": torch.cat(all_centers),
            "image_ids": dataset.image_ids,
            "model": args.model,
            "mode": mode,
        },
        save_path,
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
