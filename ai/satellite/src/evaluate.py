"""Paper-oriented evaluation for the Sen1Floods11 U-Net baseline."""

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

BATCH_SIZE = 4
NUM_WORKERS = 0
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "unet_baseline_best.pt"
)


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

def calculate_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Calculate TP, FP, FN and valid-pixel count."""

    valid = targets != -1

    predictions = predictions[valid]
    targets = targets[valid]

    tp = int(
        ((predictions == 1) & (targets == 1))
        .sum()
        .item()
    )

    fp = int(
        ((predictions == 1) & (targets == 0))
        .sum()
        .item()
    )

    fn = int(
        ((predictions == 0) & (targets == 1))
        .sum()
        .item()
    )

    valid_pixels = int(valid.sum().item())

    return tp, fp, fn, valid_pixels


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> tuple[float, float, float, float]:
    """Calculate Dice, IoU, precision and recall."""

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator > 0:
        dice = (
            2.0 * tp
            / dice_denominator
        )
    else:
        dice = 1.0

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator > 0:
        iou = (
            tp
            / iou_denominator
        )
    else:
        iou = 1.0

    precision_denominator = tp + fp

    if precision_denominator > 0:
        precision = (
            tp
            / precision_denominator
        )
    else:
        precision = 1.0

    recall_denominator = tp + fn

    if recall_denominator > 0:
        recall = (
            tp
            / recall_denominator
        )
    else:
        recall = 1.0

    return (
        dice,
        iou,
        precision,
        recall,
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate a split using global and per-chip metrics."""

    model.eval()

    total_loss = 0.0
    batches = 0

    global_tp = 0
    global_fp = 0
    global_fn = 0

    total_valid_pixels = 0
    total_ignore_pixels = 0

    chip_dice = []
    chip_iou = []
    chip_precision = []
    chip_recall = []

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

        total_loss += float(
            loss.detach()
        )

        batches += 1

        probabilities = torch.sigmoid(
            logits[:, 0]
        )

        predictions = (
            probabilities >= THRESHOLD
        ).long()

        # -------------------------------------------------------------
        # Per-chip metrics
        # -------------------------------------------------------------

        for prediction, target in zip(
            predictions,
            masks,
        ):

            (
                tp,
                fp,
                fn,
                valid_pixels,
            ) = calculate_counts(
                prediction,
                target,
            )

            (
                dice,
                iou,
                precision,
                recall,
            ) = metrics_from_counts(
                tp,
                fp,
                fn,
            )

            chip_dice.append(dice)
            chip_iou.append(iou)
            chip_precision.append(precision)
            chip_recall.append(recall)

            global_tp += tp
            global_fp += fp
            global_fn += fn

            total_valid_pixels += valid_pixels

            total_ignore_pixels += int(
                (target == -1).sum().item()
            )

    # ---------------------------------------------------------------
    # Global metrics
    # ---------------------------------------------------------------

    (
        global_dice,
        global_iou,
        global_precision,
        global_recall,
    ) = metrics_from_counts(
        global_tp,
        global_fp,
        global_fn,
    )

    # ---------------------------------------------------------------
    # Per-chip statistics
    # ---------------------------------------------------------------

    chip_dice = np.asarray(
        chip_dice,
        dtype=np.float64,
    )

    chip_iou = np.asarray(
        chip_iou,
        dtype=np.float64,
    )

    chip_precision = np.asarray(
        chip_precision,
        dtype=np.float64,
    )

    chip_recall = np.asarray(
        chip_recall,
        dtype=np.float64,
    )

    return {
        "loss": total_loss / batches,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "mean_dice": float(chip_dice.mean()),
        "std_dice": float(chip_dice.std()),
        "mean_iou": float(chip_iou.mean()),
        "std_iou": float(chip_iou.std()),
        "mean_precision": float(
            chip_precision.mean()
        ),
        "std_precision": float(
            chip_precision.std()
        ),
        "mean_recall": float(
            chip_recall.mean()
        ),
        "std_recall": float(
            chip_recall.std()
        ),
        "num_chips": len(chip_dice),
        "valid_pixels": total_valid_pixels,
        "ignore_pixels": total_ignore_pixels,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== ResQTwin: Paper-oriented U-Net evaluation ==="
    )

    print(
        f"Device:        {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU:           "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint:    {CHECKPOINT}"
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = UNet().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    loss_fn = FloodSegmentationLoss()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss:  "
        f"{checkpoint['validation_loss']:.6f}"
    )

    print()

    for split in [
        "test",
        "bolivia_holdout",
    ]:

        dataset = Sen1Floods11Dataset(
            split
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        results = evaluate_split(
            model,
            loader,
            loss_fn,
            device,
        )

        total_pixels = (
            results["valid_pixels"]
            + results["ignore_pixels"]
        )

        usable_ratio = (
            100.0
            * results["valid_pixels"]
            / total_pixels
        )

        print(
            f"## {split}"
        )

        print(
            f"Samples:             "
            f"{len(dataset)}"
        )

        print(
            f"Evaluated chips:     "
            f"{results['num_chips']}"
        )

        print(
            f"Valid pixels:        "
            f"{results['valid_pixels']:,}"
        )

        print(
            f"Ignore pixels:       "
            f"{results['ignore_pixels']:,}"
        )

        print(
            f"Usable pixel ratio:  "
            f"{usable_ratio:.2f}%"
        )

        print()

        print(
            f"Loss:                "
            f"{results['loss']:.6f}"
        )

        print()

        print("Global metrics")

        print(
            f"  Dice:              "
            f"{results['global_dice']:.6f}"
        )

        print(
            f"  IoU:               "
            f"{results['global_iou']:.6f}"
        )

        print(
            f"  Precision:         "
            f"{results['global_precision']:.6f}"
        )

        print(
            f"  Recall:            "
            f"{results['global_recall']:.6f}"
        )

        print()

        print("Per-chip metrics")

        print(
            f"  Dice:              "
            f"{results['mean_dice']:.6f} "
            f"+/- {results['std_dice']:.6f}"
        )

        print(
            f"  IoU:               "
            f"{results['mean_iou']:.6f} "
            f"+/- {results['std_iou']:.6f}"
        )

        print(
            f"  Precision:         "
            f"{results['mean_precision']:.6f} "
            f"+/- {results['std_precision']:.6f}"
        )

        print(
            f"  Recall:            "
            f"{results['mean_recall']:.6f} "
            f"+/- {results['std_recall']:.6f}"
        )

        print()

        print("Confusion counts")

        print(
            f"  TP:                "
            f"{results['tp']:,}"
        )

        print(
            f"  FP:                "
            f"{results['fp']:,}"
        )

        print(
            f"  FN:                "
            f"{results['fn']:,}"
        )

        print()

    print(
        "Evaluation complete."
    )

    print(
        "No training or data modification was performed."
    )


if __name__ == "__main__":
    main()