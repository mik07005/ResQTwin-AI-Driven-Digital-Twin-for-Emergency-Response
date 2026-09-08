"""
ResQTwin - Threshold Sensitivity Analysis
==========================================

Evaluates the Positive-Weight=5 U-Net on the Sen1Floods11
test set across multiple probability thresholds.

No retraining is performed.

Thresholds:
0.10, 0.20, 0.30, ..., 0.90

Outputs:
    ai/satellite/outputs/threshold_analysis_pw5/
        threshold_metrics.csv
        global_metrics.png
        precision_recall.png
        predicted_flood_area.png
        no_flood_false_alarm.png
"""

from pathlib import Path
import sys
import csv

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

# IMPORTANT:
# Explicitly use the Positive-Weight=5 checkpoint.
# Do NOT automatically search for *best.pt because both the
# original baseline and PW=5 checkpoints now exist.

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "positive_weight_5"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "outputs"
    / "threshold_analysis_pw5"
)

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ResQTwin - Threshold Sensitivity Analysis")
print("Positive Weight = 5.0")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Checkpoint   : {CHECKPOINT_PATH}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# CHECK PATHS
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n"
        f"{CHECKPOINT_PATH}\n\n"
        f"Please verify that the Positive-Weight=5 "
        f"checkpoint exists."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading Sen1Floods11 test dataset...")

dataset = Sen1Floods11Dataset(
    split="test"
)

print(f"Test samples : {len(dataset)}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Positive-Weight=5 U-Net...")

model = UNet(
    in_channels=2,
    out_channels=1,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

# Support common checkpoint formats.
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# ------------------------------------------------------------
# Remove possible DataParallel prefix.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Print checkpoint information if available.
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch : "
            f"{checkpoint['epoch']}"
        )

    if "validation_loss" in checkpoint:
        print(
            f"Validation loss  : "
            f"{checkpoint['validation_loss']:.6f}"
        )

    if "validation_dice" in checkpoint:
        print(
            f"Validation Dice  : "
            f"{checkpoint['validation_dice']:.6f}"
        )

    if "validation_iou" in checkpoint:
        print(
            f"Validation IoU   : "
            f"{checkpoint['validation_iou']:.6f}"
        )

print()

print(
    "Positive-Weight=5 U-Net loaded successfully."
)

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_input_and_target(sample):
    """
    Handles the expected dataset return format.

    The Sen1Floods11Dataset normally returns:
        image, target

    This helper also supports dictionary-style samples.
    """

    if isinstance(sample, (tuple, list)):

        image = sample[0]
        target = sample[1]

    elif isinstance(sample, dict):

        possible_image_keys = [
            "image",
            "images",
            "x",
            "input",
        ]

        possible_target_keys = [
            "mask",
            "target",
            "label",
            "labels",
            "y",
        ]

        image = None
        target = None

        for key in possible_image_keys:

            if key in sample:
                image = sample[key]
                break

        for key in possible_target_keys:

            if key in sample:
                target = sample[key]
                break

        if image is None or target is None:
            raise ValueError(
                "Could not identify image/target "
                "keys in dataset sample."
            )

    else:

        raise TypeError(
            f"Unsupported dataset sample type: "
            f"{type(sample)}"
        )

    return image, target


def prepare_image(image):
    """
    Converts image to a 4D tensor:

        [1, C, H, W]
    """

    if not torch.is_tensor(image):
        image = torch.tensor(image)

    image = image.float()

    if image.ndim == 3:

        image = image.unsqueeze(0)

    elif image.ndim != 4:

        raise ValueError(
            f"Unexpected image dimensions: "
            f"{image.shape}"
        )

    return image


def prepare_target(target):
    """
    Converts target to numpy array of shape H x W.
    """

    if torch.is_tensor(target):

        target = (
            target
            .detach()
            .cpu()
            .numpy()
        )

    target = np.asarray(target)

    # Remove channel/batch dimensions if present.
    target = np.squeeze(target)

    return target


# ============================================================
# RUN INFERENCE ONCE
# ============================================================

print("=" * 70)
print("Running inference once on all test samples")
print("=" * 70)

probability_maps = []
target_maps = []
sample_names = []

with torch.no_grad():

    for index in range(len(dataset)):

        sample = dataset[index]

        image, target = (
            extract_input_and_target(sample)
        )

        image_tensor = (
            prepare_image(image)
            .to(DEVICE)
        )

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = (
            probabilities
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        target_np = prepare_target(
            target
        )

        probability_maps.append(
            probabilities.astype(
                np.float32
            )
        )

        target_maps.append(
            target_np
        )


        # ----------------------------------------------------
        # Try to obtain useful sample name.
        # ----------------------------------------------------

        name = (
            f"sample_{index:02d}"
        )

        try:

            if hasattr(
                dataset,
                "samples"
            ):

                item = dataset.samples[
                    index
                ]

                if isinstance(
                    item,
                    dict
                ):

                    for key in [
                        "name",
                        "id",
                        "filename",
                        "image",
                    ]:

                        if key in item:

                            name = str(
                                item[key]
                            )

                            break

                elif isinstance(
                    item,
                    (str, Path)
                ):

                    name = Path(
                        item
                    ).stem

        except Exception:
            pass


        sample_names.append(name)


        if (
            (index + 1) % 10 == 0
            or index == 0
        ):

            print(
                f"Processed "
                f"{index + 1:3d}/"
                f"{len(dataset)} samples"
            )


print()
print("Inference complete.")
print("Probability maps cached in memory.")
print()


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics_for_threshold(
    threshold
):
    """
    Calculates metrics for one probability threshold.

    Returns:
        Dictionary containing global and
        per-scene metrics.
    """

    # ========================================================
    # Global counters
    # ========================================================

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0


    # ========================================================
    # Scene-level values
    # ========================================================

    positive_scene_ious = []
    positive_scene_dices = []
    positive_scene_precisions = []
    positive_scene_recalls = []
    positive_scene_fprs = []
    positive_scene_fnrs = []

    positive_scene_pred_percentages = []
    positive_scene_gt_percentages = []
    positive_scene_area_ratios = []

    no_flood_pred_percentages = []
    no_flood_false_alarm_flags = []

    no_flood_valid_pixels = 0
    no_flood_false_positive_pixels = 0

    flood_scene_count = 0
    no_flood_scene_count = 0


    # ========================================================
    # Process every scene
    # ========================================================

    for probabilities, target in zip(
        probability_maps,
        target_maps,
    ):

        # ----------------------------------------------------
        # Ignore pixels labelled -1.
        # ----------------------------------------------------

        valid = target != -1

        if not np.any(valid):
            continue

        gt = target[valid] > 0

        pred = (
            probabilities[valid]
            >= threshold
        )


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        tp = np.sum(
            pred & gt
        )

        fp = np.sum(
            pred & ~gt
        )

        fn = np.sum(
            ~pred & gt
        )

        tn = np.sum(
            ~pred & ~gt
        )


        tp = int(tp)
        fp = int(fp)
        fn = int(fn)
        tn = int(tn)


        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn


        # ----------------------------------------------------
        # Pixel counts
        # ----------------------------------------------------

        gt_flood_pixels = (
            tp + fn
        )

        gt_non_flood_pixels = (
            tn + fp
        )

        pred_flood_pixels = (
            tp + fp
        )

        valid_pixels = (
            tp
            + fp
            + fn
            + tn
        )


        # ----------------------------------------------------
        # Flood percentages
        # ----------------------------------------------------

        gt_flood_percentage = (

            100.0
            * gt_flood_pixels
            / valid_pixels

            if valid_pixels > 0
            else 0.0
        )


        pred_flood_percentage = (

            100.0
            * pred_flood_pixels
            / valid_pixels

            if valid_pixels > 0
            else 0.0
        )


        # ====================================================
        # NO-FLOOD SCENE
        # ====================================================

        if gt_flood_pixels == 0:

            no_flood_scene_count += 1

            no_flood_pred_percentages.append(
                pred_flood_percentage
            )


            if pred_flood_pixels > 0:

                no_flood_false_alarm_flags.append(
                    1
                )

            else:

                no_flood_false_alarm_flags.append(
                    0
                )


            no_flood_valid_pixels += (
                valid_pixels
            )

            no_flood_false_positive_pixels += (
                fp
            )

            continue


        # ====================================================
        # FLOOD-CONTAINING SCENE
        # ====================================================

        flood_scene_count += 1


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        union = (
            tp
            + fp
            + fn
        )

        if union > 0:

            iou = (
                tp / union
            )

        else:

            iou = 1.0


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice_denominator = (
            2 * tp
            + fp
            + fn
        )

        if dice_denominator > 0:

            dice = (
                2 * tp
                / dice_denominator
            )

        else:

            dice = 1.0


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # False Positive Rate
        # ----------------------------------------------------

        if gt_non_flood_pixels > 0:

            fpr = (
                fp
                / gt_non_flood_pixels
            )

        else:

            fpr = 0.0


        # ----------------------------------------------------
        # False Negative Rate
        # ----------------------------------------------------

        if gt_flood_pixels > 0:

            fnr = (
                fn
                / gt_flood_pixels
            )

        else:

            fnr = 0.0


        # ----------------------------------------------------
        # Flood-area inflation ratio
        # ----------------------------------------------------

        area_ratio = (

            pred_flood_pixels
            / gt_flood_pixels

            if gt_flood_pixels > 0
            else np.nan
        )


        # ----------------------------------------------------
        # Store scene metrics
        # ----------------------------------------------------

        positive_scene_ious.append(
            iou
        )

        positive_scene_dices.append(
            dice
        )

        positive_scene_precisions.append(
            precision
        )

        positive_scene_recalls.append(
            recall
        )

        positive_scene_fprs.append(
            fpr
        )

        positive_scene_fnrs.append(
            fnr
        )

        positive_scene_pred_percentages.append(
            pred_flood_percentage
        )

        positive_scene_gt_percentages.append(
            gt_flood_percentage
        )

        positive_scene_area_ratios.append(
            area_ratio
        )


    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    global_union = (
        global_tp
        + global_fp
        + global_fn
    )

    if global_union > 0:

        global_iou = (
            global_tp
            / global_union
        )

    else:

        global_iou = 1.0


    global_dice_denominator = (
        2 * global_tp
        + global_fp
        + global_fn
    )

    if global_dice_denominator > 0:

        global_dice = (
            2 * global_tp
            / global_dice_denominator
        )

    else:

        global_dice = 1.0


    global_precision_denominator = (
        global_tp
        + global_fp
    )

    if global_precision_denominator > 0:

        global_precision = (
            global_tp
            / global_precision_denominator
        )

    else:

        global_precision = 0.0


    global_recall_denominator = (
        global_tp
        + global_fn
    )

    if global_recall_denominator > 0:

        global_recall = (
            global_tp
            / global_recall_denominator
        )

    else:

        global_recall = 0.0


    # ========================================================
    # GLOBAL FLOOD AREA
    # ========================================================

    total_gt_flood = (
        global_tp
        + global_fn
    )

    total_pred_flood = (
        global_tp
        + global_fp
    )


    if total_gt_flood > 0:

        global_area_ratio = (
            total_pred_flood
            / total_gt_flood
        )

    else:

        global_area_ratio = np.nan


    # ========================================================
    # NO-FLOOD FALSE ALARM
    # ========================================================

    if no_flood_valid_pixels > 0:

        no_flood_global_fpr = (
            no_flood_false_positive_pixels
            / no_flood_valid_pixels
        )

    else:

        no_flood_global_fpr = 0.0


    if no_flood_scene_count > 0:

        no_flood_mean_pred_percentage = (
            np.mean(
                no_flood_pred_percentages
            )
        )

        no_flood_scene_false_alarm_rate = (
            np.mean(
                no_flood_false_alarm_flags
            )
        )

    else:

        no_flood_mean_pred_percentage = 0.0
        no_flood_scene_false_alarm_rate = 0.0


    # ========================================================
    # SAFE AGGREGATION FUNCTIONS
    # ========================================================

    def safe_mean(values):

        if len(values) == 0:
            return np.nan

        return float(
            np.mean(values)
        )


    def safe_median(values):

        if len(values) == 0:
            return np.nan

        return float(
            np.median(values)
        )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "threshold": threshold,


        # ----------------------------------------------------
        # Global pixel-level metrics
        # ----------------------------------------------------

        "global_iou": global_iou,
        "global_dice": global_dice,
        "global_precision": global_precision,
        "global_recall": global_recall,


        # ----------------------------------------------------
        # Global counts
        # ----------------------------------------------------

        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "tn": global_tn,


        # ----------------------------------------------------
        # Global flood area
        # ----------------------------------------------------

        "global_gt_flood_percent": (

            100.0
            * total_gt_flood
            / (
                global_tp
                + global_fp
                + global_fn
                + global_tn
            )

        ),

        "global_pred_flood_percent": (

            100.0
            * total_pred_flood
            / (
                global_tp
                + global_fp
                + global_fn
                + global_tn
            )

        ),

        "global_area_ratio":
            global_area_ratio,


        # ----------------------------------------------------
        # Positive-scene IoU
        # ----------------------------------------------------

        "positive_scene_mean_iou":
            safe_mean(
                positive_scene_ious
            ),

        "positive_scene_median_iou":
            safe_median(
                positive_scene_ious
            ),


        # ----------------------------------------------------
        # Positive-scene Dice
        # ----------------------------------------------------

        "positive_scene_mean_dice":
            safe_mean(
                positive_scene_dices
            ),

        "positive_scene_median_dice":
            safe_median(
                positive_scene_dices
            ),


        # ----------------------------------------------------
        # Positive-scene precision / recall
        # ----------------------------------------------------

        "positive_scene_mean_precision":
            safe_mean(
                positive_scene_precisions
            ),

        "positive_scene_mean_recall":
            safe_mean(
                positive_scene_recalls
            ),


        # ----------------------------------------------------
        # Positive-scene FPR / FNR
        # ----------------------------------------------------

        "positive_scene_mean_fpr":
            safe_mean(
                positive_scene_fprs
            ),

        "positive_scene_mean_fnr":
            safe_mean(
                positive_scene_fnrs
            ),


        # ----------------------------------------------------
        # Positive-scene flood area
        # ----------------------------------------------------

        "positive_scene_mean_gt_flood_percent":
            safe_mean(
                positive_scene_gt_percentages
            ),

        "positive_scene_mean_pred_flood_percent":
            safe_mean(
                positive_scene_pred_percentages
            ),

        "positive_scene_mean_area_ratio":
            safe_mean(
                positive_scene_area_ratios
            ),

        "positive_scene_median_area_ratio":
            safe_median(
                positive_scene_area_ratios
            ),


        # ----------------------------------------------------
        # Dataset composition
        # ----------------------------------------------------

        "flood_scene_count":
            flood_scene_count,

        "no_flood_scene_count":
            no_flood_scene_count,


        # ----------------------------------------------------
        # No-flood false alarms
        # ----------------------------------------------------

        "no_flood_global_fpr":
            no_flood_global_fpr,

        "no_flood_mean_pred_flood_percent":
            no_flood_mean_pred_percentage,

        "no_flood_scene_false_alarm_rate":
            no_flood_scene_false_alarm_rate,
    }


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

print("=" * 70)
print("Evaluating thresholds")
print("=" * 70)

all_results = []


for threshold in THRESHOLDS:

    result = (
        calculate_metrics_for_threshold(
            threshold
        )
    )

    all_results.append(result)


    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        f"  Global IoU       : "
        f"{result['global_iou']:.4f} "
        f"({result['global_iou'] * 100:.2f}%)"
    )

    print(
        f"  Global Dice      : "
        f"{result['global_dice']:.4f} "
        f"({result['global_dice'] * 100:.2f}%)"
    )

    print(
        f"  Precision        : "
        f"{result['global_precision']:.4f} "
        f"({result['global_precision'] * 100:.2f}%)"
    )

    print(
        f"  Recall           : "
        f"{result['global_recall']:.4f} "
        f"({result['global_recall'] * 100:.2f}%)"
    )

    print(
        f"  Predicted flood  : "
        f"{result['global_pred_flood_percent']:.2f}%"
    )

    print(
        f"  Area ratio       : "
        f"{result['global_area_ratio']:.2f}x"
    )

    print(
        f"  No-flood FPR     : "
        f"{result['no_flood_global_fpr'] * 100:.2f}%"
    )

    print(
        f"  No-flood scenes "
        f"with FP           : "
        f"{result['no_flood_scene_false_alarm_rate'] * 100:.2f}%"
    )


# ============================================================
# FIND BEST THRESHOLDS
# ============================================================

best_global_iou = max(
    all_results,
    key=lambda x: x["global_iou"]
)

best_global_dice = max(
    all_results,
    key=lambda x: x["global_dice"]
)

best_positive_scene_iou = max(
    all_results,
    key=lambda x:
        x["positive_scene_mean_iou"]
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "threshold_metrics.csv"
)

fieldnames = list(
    all_results[0].keys()
)


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
# BEST THRESHOLD SUMMARY
# ============================================================

print()

print("=" * 70)
print("Best thresholds")
print("=" * 70)

print(
    f"Best global IoU threshold       : "
    f"{best_global_iou['threshold']:.2f}"
)

print(
    f"Best global IoU                 : "
    f"{best_global_iou['global_iou'] * 100:.2f}%"
)

print(
    f"Best global Dice threshold      : "
    f"{best_global_dice['threshold']:.2f}"
)

print(
    f"Best global Dice                : "
    f"{best_global_dice['global_dice'] * 100:.2f}%"
)

print(
    f"Best mean scene IoU threshold   : "
    f"{best_positive_scene_iou['threshold']:.2f}"
)

print(
    f"Best mean scene IoU             : "
    f"{best_positive_scene_iou['positive_scene_mean_iou'] * 100:.2f}%"
)

print()


# ============================================================
# PLOT 1: GLOBAL IOU / DICE
# ============================================================

threshold_values = [
    r["threshold"]
    for r in all_results
]

global_iou_values = [
    r["global_iou"] * 100
    for r in all_results
]

global_dice_values = [
    r["global_dice"] * 100
    for r in all_results
]


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    threshold_values,
    global_iou_values,
    marker="o",
    label="Global IoU",
)

plt.plot(
    threshold_values,
    global_dice_values,
    marker="o",
    label="Global Dice",
)

plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "PW=5 Threshold Sensitivity - IoU and Dice"
)

plt.xticks(
    threshold_values
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR
    / "global_metrics.png"
)

plt.savefig(
    plot_path,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT 2: PRECISION / RECALL
# ============================================================

precision_values = [
    r["global_precision"] * 100
    for r in all_results
]

recall_values = [
    r["global_recall"] * 100
    for r in all_results
]


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    threshold_values,
    precision_values,
    marker="o",
    label="Precision",
)

plt.plot(
    threshold_values,
    recall_values,
    marker="o",
    label="Recall",
)

plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "PW=5 Threshold Sensitivity - Precision vs Recall"
)

plt.xticks(
    threshold_values
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR
    / "precision_recall.png"
)

plt.savefig(
    plot_path,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT 3: FLOOD AREA
# ============================================================

predicted_flood_values = [
    r["global_pred_flood_percent"]
    for r in all_results
]

area_ratio_values = [
    r["global_area_ratio"]
    for r in all_results
]


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    threshold_values,
    predicted_flood_values,
    marker="o",
    label="Predicted Flood Area (%)",
)

plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "Predicted Flood Area (%)"
)

plt.title(
    "PW=5 Threshold Sensitivity - Predicted Flood Area"
)

plt.xticks(
    threshold_values
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR
    / "predicted_flood_area.png"
)

plt.savefig(
    plot_path,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT 4: NO-FLOOD FALSE ALARM
# ============================================================

no_flood_fpr_values = [
    r["no_flood_global_fpr"] * 100
    for r in all_results
]

no_flood_pred_values = [
    r["no_flood_mean_pred_flood_percent"]
    for r in all_results
]


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    threshold_values,
    no_flood_fpr_values,
    marker="o",
    label="No-Flood Pixel FPR (%)",
)

plt.plot(
    threshold_values,
    no_flood_pred_values,
    marker="o",
    label=(
        "Mean Predicted Flood in "
        "No-Flood Scenes (%)"
    ),
)

plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "False Alarm (%)"
)

plt.title(
    "PW=5 Threshold Sensitivity - No-Flood False Alarms"
)

plt.xticks(
    threshold_values
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR
    / "no_flood_false_alarm.png"
)

plt.savefig(
    plot_path,
    dpi=200,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("PW=5 THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print()

print("Output directory:")
print(OUTPUT_DIR)

print()

print("Files generated:")

print(
    f"  - {csv_path.name}"
)

print(
    "  - global_metrics.png"
)

print(
    "  - precision_recall.png"
)

print(
    "  - predicted_flood_area.png"
)

print(
    "  - no_flood_false_alarm.png"
)

print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "This experiment evaluates the Positive-Weight=5 "
    "model only."
)

print(
    "No retraining was performed."
)

print(
    "Do NOT change the production threshold yet."
)

print(
    "We will compare this threshold curve against "
    "the original baseline threshold curve before "
    "selecting the final satellite configuration."
)

print("=" * 70)