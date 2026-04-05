import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import challenge.data as data
import challenge.stain as stain


# =========================
# Normalization functions
# =========================

def macenko(args):
    out = Path("outputs/tmp")
    out.mkdir(exist_ok=True)

    dataset = data.PatchDataset(args.target, mode=args.target.stem)

    normalizer = stain.MacenkoNormalizer()
    normalizer.fit(dataset, args.n)

    normalizer.save(args.output_dir / f'{args.method}_{args.target.stem}.pth')


def reinhard(args):
    raise NotImplementedError("Reinhard normalization not implemented yet")


# =========================
# Argument parser
# =========================

def get_parser():
    parser = argparse.ArgumentParser(
        description="Stain normalization pipeline"
    )
    
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="target dataset"
    )
    common_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/normalizer"),
        help="Directory where outputs will be saved"
    )

    subparsers = parser.add_subparsers(dest="method", required=True)
    macenko_parser = subparsers.add_parser(
        "macenko",
        parents=[common_parser],
        help="Macenko normalization"
    )
    
    macenko_parser.add_argument(
        "n",
        type=int,
        help="Number of samples used to fit the normalizer"
    )
    macenko_parser.set_defaults(func=macenko)

    return parser


# =========================
# Main
# =========================

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    args.func(args)
    print('done')


if __name__ == "__main__":
    main()