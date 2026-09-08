"""
ResQTwin - Positive Class Weight Experiment

Experiment:
    Same baseline U-Net
    Same Sen1Floods11 dataset
    Same preprocessing
    Same seed
    Same batch size
    Same learning rate
    Same number of epochs
    Same Dice weight

Only changed parameter:
    positive_weight = 5.0

Purpose:
    Determine whether the strong positive-class weighting
    used by the baseline (9.52) contributes to excessive
    flood-area overprediction / false positives.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.satellite.models.unet import UNet
from ai.satellite.src.dataset import Sen1Floods11Dataset
from ai.satellite.src.losses import FloodSegmentationLoss


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 4

LEARNING_RATE = 1e-3

EPOCHS = 50

NUM_WORKERS = 0

# ------------------------------------------------------------
# EXPERIMENTAL PARAMETER
# ------------------------------------------------------------

POSITIVE_WEIGHT = 5.0

# Keep identical to baseline.
DICE_WEIGHT = 0.5


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "positive_weight_5"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METRICS
# ============================================================

def segmentation_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int]:

    probabilities = torch.sigmoid(
        logits[:, 0]
    )

    predictions = (
        probabilities >= 0.5
    )

    valid = targets != -1

    target_binary = targets == 1

    predictions = predictions[valid]
    target_binary = target_binary[valid]

    intersection = int(
        (
            predictions
            & target_binary
        ).sum().item()
    )

    prediction_sum = int(
        predictions.sum().item()
    )

    target_sum = int(
        target_binary.sum().item()
    )

    return (
        intersection,
        prediction_sum,
        target_sum,
    )


def calculate_dice_iou(
    intersection: int,
    prediction_sum: int,
    target_sum: int,
) -> tuple[float, float]:

    dice_denominator = (
        prediction_sum
        + target_sum
    )

    if dice_denominator > 0:

        dice = (
            2.0
            * intersection
            / dice_denominator
        )

    else:

        dice = 1.0

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    if union > 0:

        iou = (
            intersection
            / union
        )

    else:

        iou = 1.0

    return dice, iou


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:

    model.train()

    total_loss = 0.0

    total_intersection = 0
    total_prediction_sum = 0
    total_target_sum = 0

    batches = 0

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = loss_fn(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        (
            intersection,
            prediction_sum,
            target_sum,
        ) = segmentation_counts(
            logits.detach(),
            masks,
        )

        total_loss += float(
            loss.detach()
        )

        total_intersection += (
            intersection
        )

        total_prediction_sum += (
            prediction_sum
        )

        total_target_sum += (
            target_sum
        )

        batches += 1

    dice, iou = calculate_dice_iou(
        total_intersection,
        total_prediction_sum,
        total_target_sum,
    )

    return (
        total_loss / batches,
        dice,
        iou,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:

    model.eval()

    total_loss = 0.0

    total_intersection = 0
    total_prediction_sum = 0
    total_target_sum = 0

    batches = 0

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = loss_fn(
            logits,
            masks,
        )

        (
            intersection,
            prediction_sum,
            target_sum,
        ) = segmentation_counts(
            logits,
            masks,
        )

        total_loss += float(
            loss.detach()
        )

        total_intersection += (
            intersection
        )

        total_prediction_sum += (
            prediction_sum
        )

        total_target_sum += (
            target_sum
        )

        batches += 1

    dice, iou = calculate_dice_iou(
        total_intersection,
        total_prediction_sum,
        total_target_sum,
    )

    return (
        total_loss / batches,
        dice,
        iou,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "ResQTwin - Positive Weight Experiment"
    )
    print("=" * 70)

    print(
        f"Device:          {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:             "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory:      "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"Seed:             {SEED}"
    )

    print(
        f"Batch size:       {BATCH_SIZE}"
    )

    print(
        f"Learning rate:    {LEARNING_RATE}"
    )

    print(
        f"Epochs:           {EPOCHS}"
    )

    print(
        f"Positive weight:  {POSITIVE_WEIGHT}"
    )

    print(
        f"Dice weight:      {DICE_WEIGHT}"
    )

    print()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("Loading datasets...")

    train_dataset = Sen1Floods11Dataset(
        "train"
    )

    validation_dataset = (
        Sen1Floods11Dataset(
            "validation"
        )
    )

    print(
        f"Training samples:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet().to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters:  "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = FloodSegmentationLoss(
        positive_weight=POSITIVE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    print(
        "Loss:"
    )

    print(
        f"  Weighted BCE positive weight = "
        f"{POSITIVE_WEIGHT}"
    )

    print(
        f"  Dice weight = "
        f"{DICE_WEIGHT}"
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    training_start = (
        time.perf_counter()
    )

    print("=" * 70)
    print("Starting training")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        (
            train_loss,
            train_dice,
            train_iou,
        ) = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_dice,
            validation_iou,
        ) = evaluate(
            model,
            validation_loader,
            loss_fn,
            device,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Dice: {validation_dice:.4f} | "
            f"Val IoU: {validation_iou:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "validation_loss":
                        validation_loss,

                    "validation_dice":
                        validation_dice,

                    "validation_iou":
                        validation_iou,

                    "seed":
                        SEED,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "epochs":
                        EPOCHS,

                    "positive_weight":
                        POSITIVE_WEIGHT,

                    "dice_weight":
                        DICE_WEIGHT,

                    "experiment":
                        "positive_weight_5",
                },
                BEST_CHECKPOINT,
            )

            print(
                "  Saved best checkpoint: "
                f"{BEST_CHECKPOINT}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    training_time = (
        time.perf_counter()
        - training_start
    )

    print()

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch:        {best_epoch}"
    )

    print(
        f"Best Val Loss:     "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Total time:        "
        f"{training_time:.1f}s"
    )

    print(
        f"Checkpoint:        "
        f"{BEST_CHECKPOINT}"
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "  Baseline positive weight = 9.52"
    )

    print(
        "  This experiment          = 5.00"
    )

    print()

    print(
        "Training pipeline complete."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()