"""
ResQTwin - Threshold Diagnostic Visualization
==============================================

Compares the frozen baseline U-Net at thresholds:
    0.50, 0.80, 0.90

on difficult / representative Sen1Floods11 test scenes.

No retraining is performed.

Outputs:
    ai/satellite/outputs/threshold_diagnostics/

For each selected scene:
    - Probability map
    - Ground truth
    - Prediction @ 0.50
    - Prediction @ 0.80
    - Prediction @ 0.90

Also prints IoU, Dice, Precision, Recall and flood-area
statistics for each threshold.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from ai.satellite.src.dataset import Sen1Floods11Dataset
from ai.satellite.models.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
)

checkpoint_matches = list(
    CHECKPOINT_DIR.rglob("*best.pt")
)

if not checkpoint_matches:
    raise FileNotFoundError(
        f"No '*best.pt' checkpoint found under:\n"
        f"{CHECKPOINT_DIR}"
    )

CHECKPOINT_PATH = checkpoint_matches[0]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "outputs"
    / "threshold_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.50,
    0.80,
    0.90,
]


# ============================================================
# SELECTED TEST SCENES
# ============================================================
#
# These are scenes identified during our complete test-set
# failure analysis.
#
# 00  Ghana_313799       - strong underprediction
# 29  Mekong_254910      - severe area overprediction
# 40  Pakistan_167553    - severe area overprediction
# 38  Pakistan_70625     - highest raw FP
# 35  Pakistan_849790    - high raw FN
# 44  Paraguay_232281    - strong recall but area inflation
# 58  Somalia_166342     - tiny GT flood, huge FP
# 39  Pakistan_528249    - no-flood scene with false alarm
#
# ============================================================

SELECTED_INDICES = [
    0,
    29,
    40,
    38,
    35,
    44,
    58,
    39,
]


# Known names from our earlier test-set analysis.
KNOWN_NAMES = {
    0: "Ghana_313799",
    29: "Mekong_254910",
    40: "Pakistan_167553",
    38: "Pakistan_70625",
    35: "Pakistan_849790",
    44: "Paraguay_232281",
    58: "Somalia_166342",
    39: "Pakistan_528249",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_input_and_target(sample):
    """
    Extract image and target from the dataset sample.
    """

    if isinstance(sample, (tuple, list)):

        image = sample[0]
        target = sample[1]

    elif isinstance(sample, dict):

        image = None
        target = None

        for key in [
            "image",
            "images",
            "x",
            "input",
        ]:
            if key in sample:
                image = sample[key]
                break

        for key in [
            "mask",
            "target",
            "label",
            "labels",
            "y",
        ]:
            if key in sample:
                target = sample[key]
                break

        if image is None or target is None:
            raise ValueError(
                "Could not identify image/target "
                "inside dataset sample."
            )

    else:
        raise TypeError(
            f"Unsupported dataset sample type: "
            f"{type(sample)}"
        )

    return image, target


def prepare_image(image):
    """
    Convert image into [1, C, H, W].
    """

    if not torch.is_tensor(image):
        image = torch.tensor(image)

    image = image.float()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    elif image.ndim != 4:
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    return image


def prepare_target(target):
    """
    Convert target to H x W numpy array.
    """

    if torch.is_tensor(target):
        target = (
            target
            .detach()
            .cpu()
            .numpy()
        )

    target = np.asarray(target)

    target = np.squeeze(target)

    return target


def calculate_metrics(
    probability_map,
    target,
    threshold,
):
    """
    Calculate segmentation metrics for one scene.
    """

    valid = target != -1

    if not np.any(valid):
        return {
            "iou": np.nan,
            "dice": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "gt_percent": np.nan,
            "pred_percent": np.nan,
            "area_ratio": np.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    gt = target[valid] > 0

    pred = (
        probability_map[valid]
        >= threshold
    )

    tp = int(
        np.sum(pred & gt)
    )

    fp = int(
        np.sum(pred & ~gt)
    )

    fn = int(
        np.sum(~pred & gt)
    )

    tn = int(
        np.sum(~pred & ~gt)
    )

    valid_pixels = (
        tp + fp + fn + tn
    )

    gt_flood_pixels = tp + fn

    pred_flood_pixels = tp + fp

    union = tp + fp + fn

    if union > 0:
        iou = tp / union
    else:
        iou = 1.0

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator > 0:
        dice = (
            2 * tp
            / dice_denominator
        )
    else:
        dice = 1.0

    precision_denominator = (
        tp + fp
    )

    if precision_denominator > 0:
        precision = (
            tp
            / precision_denominator
        )
    else:
        precision = 0.0

    recall_denominator = (
        tp + fn
    )

    if recall_denominator > 0:
        recall = (
            tp
            / recall_denominator
        )
    else:
        recall = 0.0

    gt_percent = (
        100.0
        * gt_flood_pixels
        / valid_pixels
    )

    pred_percent = (
        100.0
        * pred_flood_pixels
        / valid_pixels
    )

    if gt_flood_pixels > 0:
        area_ratio = (
            pred_flood_pixels
            / gt_flood_pixels
        )
    else:
        area_ratio = np.nan

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_percent": gt_percent,
        "pred_percent": pred_percent,
        "area_ratio": area_ratio,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def make_mask(
    probability_map,
    threshold,
):
    """
    Binary prediction mask.
    """

    return (
        probability_map
        >= threshold
    ).astype(np.uint8)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("ResQTwin - Threshold Diagnostic Visualization")
print("=" * 70)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Checkpoint   : {CHECKPOINT_PATH}"
)

print(
    f"Device       : {DEVICE}"
)

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()

dataset = Sen1Floods11Dataset(
    split="test"
)

print(
    f"Test samples : {len(dataset)}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading baseline U-Net...")

model = UNet(
    in_channels=2,
    out_channels=1,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        state_dict = (
            checkpoint["state_dict"]
        )

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[len("module."):]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict
)

model.to(DEVICE)
model.eval()

print(
    "Baseline U-Net loaded successfully."
)

print()


# ============================================================
# PROCESS SELECTED SCENES
# ============================================================

all_results = []


for scene_index in SELECTED_INDICES:

    print("=" * 70)

    scene_name = KNOWN_NAMES.get(
        scene_index,
        f"sample_{scene_index:02d}"
    )

    print(
        f"Scene {scene_index:02d}: "
        f"{scene_name}"
    )

    # --------------------------------------------------------
    # Load scene
    # --------------------------------------------------------

    sample = dataset[scene_index]

    image, target = (
        extract_input_and_target(sample)
    )

    image_tensor = (
        prepare_image(image)
        .to(DEVICE)
    )

    target_np = prepare_target(
        target
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability_map = (
            torch.sigmoid(logits)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Print probability information
    # --------------------------------------------------------

    print(
        f"Probability range: "
        f"{probability_map.min():.6f} "
        f"to "
        f"{probability_map.max():.6f}"
    )

    print()

    # --------------------------------------------------------
    # Calculate all thresholds
    # --------------------------------------------------------

    threshold_metrics = {}

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            probability_map,
            target_np,
            threshold,
        )

        threshold_metrics[
            threshold
        ] = metrics

        print(
            f"Threshold {threshold:.2f} | "
            f"IoU {metrics['iou'] * 100:.2f}% | "
            f"Dice {metrics['dice'] * 100:.2f}% | "
            f"Precision "
            f"{metrics['precision'] * 100:.2f}% | "
            f"Recall "
            f"{metrics['recall'] * 100:.2f}% | "
            f"GT "
            f"{metrics['gt_percent']:.2f}% | "
            f"Pred "
            f"{metrics['pred_percent']:.2f}% | "
            f"Area "
            f"{metrics['area_ratio']:.2f}x"
        )

        all_results.append(
            {
                "scene_index": scene_index,
                "scene_name": scene_name,
                "threshold": threshold,
                **metrics,
            }
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    valid = target_np != -1

    # Create display versions.
    probability_display = (
        probability_map.copy()
    )

    probability_display[~valid] = np.nan

    gt_display = target_np.astype(
        np.float32
    )

    gt_display[~valid] = np.nan

    prediction_masks = {}

    for threshold in THRESHOLDS:

        prediction = make_mask(
            probability_map,
            threshold,
        ).astype(np.float32)

        prediction[~valid] = np.nan

        prediction_masks[
            threshold
        ] = prediction

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    fig.suptitle(
        f"{scene_name} - "
        f"Threshold Comparison",
        fontsize=15,
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    axes[0].imshow(
        probability_display,
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Model Probability"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(
        gt_display,
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"Ground Truth\n"
        f"Flood = "
        f"{np.sum(target_np[valid] > 0) / np.sum(valid) * 100:.2f}%"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Threshold 0.50
    # --------------------------------------------------------

    metrics = threshold_metrics[0.50]

    axes[2].imshow(
        prediction_masks[0.50],
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"Threshold 0.50\n"
        f"IoU {metrics['iou'] * 100:.1f}%"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Threshold 0.80
    # --------------------------------------------------------

    metrics = threshold_metrics[0.80]

    axes[3].imshow(
        prediction_masks[0.80],
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        f"Threshold 0.80\n"
        f"IoU {metrics['iou'] * 100:.1f}%"
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # Threshold 0.90
    # --------------------------------------------------------

    metrics = threshold_metrics[0.90]

    axes[4].imshow(
        prediction_masks[0.90],
        vmin=0,
        vmax=1,
    )

    axes[4].set_title(
        f"Threshold 0.90\n"
        f"IoU {metrics['iou'] * 100:.1f}%"
    )

    axes[4].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"scene_{scene_index:02d}_"
        f"{scene_name}_thresholds.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Saved visualization:"
    )
    print(
        output_path
    )


# ============================================================
# SAVE CSV
# ============================================================

import csv

csv_path = (
    OUTPUT_DIR
    / "threshold_diagnostic_metrics.csv"
)

fieldnames = [
    "scene_index",
    "scene_name",
    "threshold",
    "iou",
    "dice",
    "precision",
    "recall",
    "gt_percent",
    "pred_percent",
    "area_ratio",
    "tp",
    "fp",
    "fn",
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("THRESHOLD DIAGNOSTICS COMPLETE")
print("=" * 70)

print()
print(
    f"Visualizations saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    f"Metrics saved to:"
)

print(
    csv_path
)

print()
print(
    "Compared thresholds: "
    "0.50, 0.80, 0.90"
)

print(
    f"Scenes analyzed: "
    f"{len(SELECTED_INDICES)}"
)

print("=" * 70)