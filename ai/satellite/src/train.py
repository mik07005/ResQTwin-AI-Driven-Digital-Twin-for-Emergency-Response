"""Baseline U-Net training for Sen1Floods11."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.satellite.models.unet import UNet
from ai.satellite.src.dataset import Sen1Floods11Dataset
from ai.satellite.src.losses import FloodSegmentationLoss


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
NUM_WORKERS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "satellite" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "unet_baseline_best.pt"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Calculate Dice and IoU while ignoring target label -1."""

    probabilities = torch.sigmoid(logits[:, 0])
    predictions = probabilities >= 0.5

    valid = targets != -1
    target_binary = targets == 1

    predictions = predictions[valid]
    target_binary = target_binary[valid]

    intersection = (predictions & target_binary).sum().float()
    prediction_sum = predictions.sum().float()
    target_sum = target_binary.sum().float()

    dice_denominator = prediction_sum + target_sum
    dice = (
        (2.0 * intersection / dice_denominator)
        if dice_denominator > 0
        else torch.tensor(1.0, device=logits.device)
    )

    union = prediction_sum + target_sum - intersection
    iou = (
        (intersection / union)
        if union > 0
        else torch.tensor(1.0, device=logits.device)
    )

    return float(dice.detach()), float(iou.detach())


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:

    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        dice, iou = segmentation_metrics(logits.detach(), masks)

        total_loss += float(loss.detach())
        total_dice += dice
        total_iou += iou
        batches += 1

    return (
        total_loss / batches,
        total_dice / batches,
        total_iou / batches,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        dice, iou = segmentation_metrics(logits, masks)

        total_loss += float(loss.detach())
        total_dice += dice
        total_iou += iou
        batches += 1

    return (
        total_loss / batches,
        total_dice / batches,
        total_iou / batches,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== ResQTwin: Baseline U-Net training ===")
    print(f"Device:        {device}")

    if device.type == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(0)}")

    print(f"Seed:          {SEED}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs:        {EPOCHS}")
    print()

    train_dataset = Sen1Floods11Dataset("train")
    validation_dataset = Sen1Floods11Dataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = UNet().to(device)
    loss_fn = FloodSegmentationLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        validation_loss, validation_dice, validation_iou = validate(
            model,
            validation_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Dice: {validation_dice:.4f} | "
            f"Val IoU: {validation_iou:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "seed": SEED,
                },
                BEST_CHECKPOINT,
            )

            print(f"  Saved best checkpoint: {BEST_CHECKPOINT}")

    print()
    print("Training smoke test complete.")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()