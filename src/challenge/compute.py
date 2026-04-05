import torch
import torch.nn as nn

from tqdm.auto import tqdm

def run_epoch(
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

            cutmix_active = labels.ndim == 2
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            if cutmix_active:
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels

            correct += (preds == targets).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total