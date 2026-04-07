import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from challenge import data, stain

# =========================
# Argument parser
# =========================

"""def get_parser():
    parser = argparse.ArgumentParser(
        description="Apply a normalization method to a dataset"
    )

    parser.add_argument(
        "method",
        choices=["macenko", "HEDJitter"],
        help="Normalization method to use"
    )
    
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint"
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset"
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of exemples."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default='outputs/normalizer_exemples',
        help="Number of exemples."
    )

    return parser"""


def get_parser():
    parser = argparse.ArgumentParser(description="Stain normalization pipeline")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--n", type=int, default=10, help="Number of exemples.")

    common_parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/normalizer_exemples",
        help="Number of exemples.",
    )

    common_parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset")

    subparsers = parser.add_subparsers(dest="method", required=True)
    macenko_parser = subparsers.add_parser("macenko", parents=[common_parser], help="Macenko normalization")

    macenko_parser.add_argument("checkpoint", type=Path, help="Path to checkpoint")

    HEDJitter_parser = subparsers.add_parser(
        "HEDJitter",
        parents=[common_parser],
        help="HEDJitter normalization",
    )

    HEDJitter_parser.add_argument("theta", type=float, help="beta parameter")

    HEAugmentor_parser = subparsers.add_parser("HE", parents=[common_parser], help="HE augmentation")

    HEAugmentor_parser.add_argument("--sigma1", default=0.4, type=float, help="beta parameter")

    HEAugmentor_parser.add_argument("--sigma2", default=0.4, type=float, help="beta parameter")

    return parser


# =========================
# Main
# =========================


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.method == "macenko":
        normalizer = stain.MacenkoNormalizer.load(args.checkpoint)
    if args.method == "HEDJitter":
        normalizer = stain.HEDJitter(args.theta)
    if args.method == "HE":
        normalizer = stain.HEAugmentor(args.sigma1, args.sigma2)

    dataset = data.PatchDataset(args.dataset, mode=args.dataset.stem)

    indices = torch.randperm(len(dataset))[: args.n]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.n)

    batch = next(iter(loader))
    images = batch["img"]
    centers = batch["center"]

    plt.figure()

    n_col = 4
    for i, (image, center) in enumerate(zip(images, centers)):
        norm = normalizer(image)
        plt.subplot(args.n // n_col + 1, n_col, i + 1)
        norm = norm.permute(1, 2, 0).numpy()
        plt.imshow(norm)
        plt.title(f"{center}")
        plt.axis("off")

    args.output_dir.mkdir(exist_ok=True)
    plt.savefig(args.output_dir / f"{args.method}.png")
    print("done")


if __name__ == "__main__":
    main()
