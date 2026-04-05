import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models
import torchvision.transforms as T

import argparse
import csv
from pathlib import Path

from tqdm.auto import tqdm

from peft import PeftModel

from challenge.data import PatchDataset
from challenge import stain
from challenge import builder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on binary classification.")

    parser.add_argument("checkpoint", type=Path,
                        help="Directory where model checkpoints are saved.")
    parser.add_argument("--test_path", type=Path, default="data/test.h5",
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





def main() -> None:
    args = parse_args()

    # Reproducibility


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Stain normaliser ──────────────────────────────────────────────────────
    model, normalizer = build(args.checkpoint)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    
    def collate_fn(batch,):
        images = torch.stack([item['img'] for item in batch])
        image_ids = torch.tensor([int(item['img_id']) for item in batch])
        
        return images, image_ids

    test_dataset = PatchDataset(str(args.test_path), mode="test", transform=normalizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda x : collate_fn(x)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = model.to(device)
    
    generate_predictions(
        model=model,
        dataloader=test_dataloader,
        device=device,
        prediction_path=args.prediction_path,
    )
    print(f"Saved predictions to {args.prediction_path}")


if __name__ == "__main__":
    main()