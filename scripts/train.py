from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

from peft import LoraConfig, get_peft_model

from challenge.data import PatchDataset
from challenge import stain
from challenge import builder
from challenge import preprocessing
from challenge import utils
from challenge import compute

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on binary classification.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--train", type=Path, default="data/train.h5",
                        help="Path to training .h5 file.")
    parser.add_argument("--val", type=Path, default="data/val.h5",
                        help="Path to validation .h5 file.")
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--HEDJitter",
                        type=float,
                        nargs=2,
                        default=[0, 0],
                        help="Theta parameter.")
    parser.add_argument("--HEAug",
                        type=float,
                        nargs=3,
                        default=[0, 0, 0],
                        help="Theta parameter.")
    parser.add_argument("--normalizer", nargs=2,
                        metavar=("method", "checkpoint-path"),
                        default=[None, None],
                        help="Method and path to the normalizer.")
    parser.add_argument("--output-dir", type=Path, default="checkpoints",
                        help="Directory where model checkpoints are saved.")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "kimianet"],
                        help="ResNet architecture to use.")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Initialise backbone with ImageNet weights.")
    parser.add_argument("--freeze-backbone", type=int, default=0,
                        help="Freeze all layers except the final classifier head.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=5,
                        help="Total number of training epochs.")
    parser.add_argument("--lora", type=int, nargs=2, default=[0, 0],
                        help="LoRa parameters")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size for both train and val loaders.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight-decay / regularisation coefficient.")
    parser.add_argument("--pos-weight", type=float, default=1,
                        help="Scalar weight for the positive class in BCEWithLogitsLoss "
                             "(useful for imbalanced datasets). Defaults to 1.0 (no reweighting).")
    parser.add_argument("--use-stain-aug", action="store_true", default=False,
                        help="Apply stain-colour augmentation during training "
                             "(uses STAIN_TRANSFORMS).")

    # ── Data augmentation ─────────────────────────────────────────────────────
    parser.add_argument("--color-jitter", type=float, default=0.4,
                        help="Strength for ColorJitter (brightness/contrast/saturation). "
                             "Set to 0 to disable.")
    parser.add_argument("--hue-jitter", type=float, default=0.1,
                        help="Hue jitter range for ColorJitter. Set to 0 to disable.")
    parser.add_argument("--no-hflip", action="store_true", default=False,
                        help="Disable random horizontal flip.")
    parser.add_argument("--no-vflip", action="store_true", default=False,
                        help="Disable random vertical flip.")
    parser.add_argument("--no-rotation", action="store_true", default=False,
                        help="Disable random 90° rotation.")
    parser.add_argument("--gaussian-blur", action="store_true", default=False,
                        help="Enable random Gaussian blur augmentation.")
    parser.add_argument("--mix-alpha", nargs=2, type=float, default=[0, 0],
                        help="Alpha parameter for CutMix Beta distribution. "
                             "Set to 0 to disable CutMix .")
    parser.add_argument("--mix-prob", type=float, default=0.1,
                        help="Probability of applying CutMix to a given batch.")

    # ── LR schedule ───────────────────────────────────────────────────────────
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"],
                        help="Learning-rate scheduler type.")
    parser.add_argument("--lr-step-size", type=int, default=7,
                        help="Step size (epochs) for StepLR scheduler.")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="Multiplicative factor for StepLR scheduler.")

    # ── Infrastructure ────────────────────────────────────────────────────────
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint (.pt) to resume training from.")

    return parser.parse_args()


# ── Augmentation helpers ──────────────────────────────────────────────────────

def build_train_transform(args) -> T.Compose:
    """
    Build the per-image transform pipeline for training.
    Order: stain normalisation (PIL) → tensor → spatial augs → colour augs → normalise.
    """
    steps = []
    
    if args.HEDJitter[0] > 0:
        steps.append(T.RandomApply([stain.HEDJitter(args.HEDJitter[0])], p=args.HEDJitter[1]))
    
    if args.HEAug[0] > 0:
        steps.append(T.RandomApply(
                [stain.HEAugmentor(args.HEAug[0], args.HEAug[1])]
                ,p=args.HEAug[2])
                     )

    # 1. Stain normalisation (operates on PIL images, returns PIL)
    steps.append(T.ToPILImage())

    # 2. Spatial augmentations
    if not args.no_hflip:
        steps.append(T.RandomHorizontalFlip())
    if not args.no_vflip:
        steps.append(T.RandomVerticalFlip())
    if not args.no_rotation:
        # Histology patches have no canonical orientation → all 4 rotations are valid
        steps.append(T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5))

    # 3. Convert to tensor before colour / pixel-level augmentations
    steps.append(T.ToTensor())

    # 4. Colour augmentations (tensor-space)
    if args.color_jitter > 0 or args.hue_jitter > 0:
        steps.append(T.ColorJitter(
            brightness=args.color_jitter,
            contrast=args.color_jitter,
            saturation=args.color_jitter,
            hue=args.hue_jitter,
        ))
    if args.gaussian_blur:
        kernel = 3   # odd kernel ≤ crop_size/16
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=kernel)], p=0.2))

    # 5. ImageNet normalisation
    steps.append(T.Normalize(mean=MEAN, std=STD))

    return T.Compose(steps)


def build_val_transform() -> T.Compose:
    """Deterministic centre-crop pipeline for validation / inference."""
    val_transform = T.Compose(
        [
            T.Normalize(mean=MEAN, std=STD)
        ]
    )
    
    return val_transform


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Arch      : {args.arch}  |  pretrained={args.pretrained}")

    # ── Stain normaliser ──────────────────────────────────────────────────────
    normalizer = stain.build_normaliser(*args.normalizer)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = build_train_transform(args)
    val_transform   = build_val_transform()

    # ── Datasets & loaders ────────────────────────────────────────────────────
    
    collate_fn = preprocessing.base_collate
    
    if args.mix_alpha[0] > 0 or args.mix_alpha[1] > 0:
        mix = preprocessing.RandomSubsetV2Mix(alphas=args.mix_alpha, p=args.mix_prob)
    else:
        mix = None
    
    
    dataset1 = PatchDataset(str(args.train), mode="train", transform=normalizer)
    dataset2 = PatchDataset(str(args.val), mode="val", transform=normalizer)
    if args.shuffle:
        merged_dataset = ConcatDataset([dataset1, dataset2])
        train_size = int(0.8 * len(merged_dataset))
        val_size = len(merged_dataset) - train_size
        train_dataset, val_dataset = random_split(merged_dataset,
                                                  [train_size, val_size]
                                                  )
    else:
        train_dataset = dataset1
        val_dataset = dataset2
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda x : collate_fn(x, train_transform, mix=mix)
    )

    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda x : collate_fn(x, val_transform)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = builder.build_model(args.arch, args.pretrained).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, args.pos_weight], dtype=torch.float32).to(device)
    )

    def get_optim(model):
        # ── Optimiser ─────────────────────────────────────────────────────────────
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        # ── LR scheduler ─────────────────────────────────────────────────────────
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
            )
        else:
            scheduler = None
            
        return optimizer, scheduler

    optimizer, scheduler = get_optim(model)

    # ── (Optional) Resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        
        if epoch == args.freeze_backbone:
            print('Unfreeze backbone')
            for param in model.parameters():
                param.requires_grad = True
            
            if args.lora[0] > 0 and args.lora[1] > 0:
                target_modules = ["conv1", "conv2"]
                config = LoraConfig(
                    r=args.lora[0],
                    lora_alpha=args.lora[1],
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    bias="none",
                    modules_to_save=["fc_4", "fc"]
                )

                model = get_peft_model(model, config)
                
                model.print_trainable_parameters()
            
            optimizer, scheduler = get_optim(model)
        
        model.train()
        train_loss, train_acc = compute.run_epoch(
            model, train_dataloader, criterion, optimizer, device,
            is_train=True,
        )
        
        model.eval()
        val_loss, val_acc = compute.run_epoch(
            model, val_dataloader, criterion, optimizer=None, device=device,
            is_train=False,
        )

        if scheduler is not None:
            scheduler.step()
            
        name_save = 'model'

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val   loss {val_loss:.4f}  acc {val_acc:.4f} | "
            f"saving {name_save}"
        )

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            utils.save_model(model, epoch, optimizer, val_acc, args, f'best{name_save}')

        # Save latest checkpoint (for resuming)
        utils.save_model(model, epoch, optimizer, val_acc, args, f'last{name_save}')

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()