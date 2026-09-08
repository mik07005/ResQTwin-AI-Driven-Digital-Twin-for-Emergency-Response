"""
ResQTwin - Positive Class Weight Evaluation
============================================

Evaluates the frozen best checkpoint from the Positive Weight = 5.0
experiment on the Sen1Floods11 test set.

Experiment:
    Baseline U-Net
    positive_weight = 5.0
    dice_weight = 0.5

No retraining is performed.

Evaluation threshold:
    0.50

Outputs:
    ai/satellite/outputs/positive_weight_5/
        evaluation_metrics.csv
"""

from pathlib import Path
import sys
import csv

import numpy as np
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)

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
    / "positive_weight_5"
)

THRESHOLD = 0.50

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "ResQTwin - Positive Weight = 5.0 Evaluation"
)
print("=" * 70)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Checkpoint   : {CHECKPOINT_PATH}"
)

print(
    f"Threshold    : {THRESHOLD:.2f}"
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


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n"
        f"{CHECKPOINT_PATH}\n\n"
        f"Please verify that the PW=5 training "
        f"completed successfully."
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATASET
# ============================================================

print(
    "Loading Sen1Floods11 test dataset..."
)

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

print(
    "Loading Positive Weight = 5.0 U-Net..."
)

model = UNet(
    in_channels=2,
    out_channels=1,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)


# ------------------------------------------------------------
# Support common checkpoint formats
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Remove possible DataParallel prefix
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


print(
    "Positive Weight = 5.0 U-Net loaded successfully."
)

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_input_and_target(sample):
    """
    Handles the expected dataset return format.

    Sen1Floods11Dataset normally returns:

        image, target

    Also supports dictionary-style samples.
    """

    if isinstance(
        sample,
        (tuple, list),
    ):

        image = sample[0]
        target = sample[1]

    elif isinstance(
        sample,
        dict,
    ):

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

        if (
            image is None
            or target is None
        ):

            raise ValueError(
                "Could not identify "
                "image/target keys "
                "in dataset sample."
            )

    else:

        raise TypeError(
            "Unsupported dataset sample type: "
            f"{type(sample)}"
        )

    return image, target


def prepare_image(image):
    """
    Converts image to:

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
    Converts target to numpy array:

        H x W
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


# ============================================================
# GLOBAL COUNTERS
# ============================================================

global_tp = 0
global_fp = 0
global_fn = 0
global_tn = 0


# ============================================================
# SCENE-LEVEL METRICS
# ============================================================

scene_ious = []
scene_dices = []
scene_precisions = []
scene_recalls = []
scene_fprs = []
scene_fnrs = []

scene_gt_percentages = []
scene_pred_percentages = []
scene_area_ratios = []

no_flood_scene_predictions = []
no_flood_scene_false_alarm_flags = []

no_flood_valid_pixels = 0
no_flood_false_positive_pixels = 0

flood_scene_count = 0
no_flood_scene_count = 0


# ============================================================
# RUN TEST INFERENCE
# ============================================================

print("=" * 70)
print("Running test-set evaluation")
print("=" * 70)

with torch.no_grad():

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        image, target = (
            extract_input_and_target(
                sample
            )
        )

        image_tensor = (
            prepare_image(image)
            .to(DEVICE)
        )

        logits = model(
            image_tensor
        )

        probabilities = (
            torch.sigmoid(logits)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        target_np = (
            prepare_target(target)
        )

        # ----------------------------------------------------
        # Ignore pixels
        # ----------------------------------------------------

        valid = (
            target_np != -1
        )

        if not np.any(valid):

            continue

        gt = (
            target_np[valid] > 0
        )

        pred = (
            probabilities[valid]
            >= THRESHOLD
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

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

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

        # ----------------------------------------------------
        # Scene statistics
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

        gt_flood_percentage = (
            100.0
            * gt_flood_pixels
            / valid_pixels
        )

        pred_flood_percentage = (
            100.0
            * pred_flood_pixels
            / valid_pixels
        )

        # ----------------------------------------------------
        # No-flood scene
        # ----------------------------------------------------

        if gt_flood_pixels == 0:

            no_flood_scene_count += 1

            no_flood_scene_predictions.append(
                pred_flood_percentage
            )

            if pred_flood_pixels > 0:

                no_flood_scene_false_alarm_flags.append(
                    1
                )

            else:

                no_flood_scene_false_alarm_flags.append(
                    0
                )

            no_flood_valid_pixels += (
                valid_pixels
            )

            no_flood_false_positive_pixels += (
                fp
            )

            continue

        # ----------------------------------------------------
        # Flood-containing scene
        # ----------------------------------------------------

        flood_scene_count += 1

        # IoU
        union = (
            tp + fp + fn
        )

        if union > 0:

            iou = (
                tp / union
            )

        else:

            iou = 1.0

        # Dice
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

        # Precision
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

        # Recall
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

        # FPR
        if gt_non_flood_pixels > 0:

            fpr = (
                fp
                / gt_non_flood_pixels
            )

        else:

            fpr = 0.0

        # FNR
        if gt_flood_pixels > 0:

            fnr = (
                fn
                / gt_flood_pixels
            )

        else:

            fnr = 0.0

        # Area ratio
        area_ratio = (
            pred_flood_pixels
            / gt_flood_pixels
        )

        scene_ious.append(iou)
        scene_dices.append(dice)
        scene_precisions.append(
            precision
        )
        scene_recalls.append(
            recall
        )
        scene_fprs.append(fpr)
        scene_fnrs.append(fnr)

        scene_gt_percentages.append(
            gt_flood_percentage
        )

        scene_pred_percentages.append(
            pred_flood_percentage
        )

        scene_area_ratios.append(
            area_ratio
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

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

print(
    "Test inference complete."
)

print()


# ============================================================
# GLOBAL METRICS
# ============================================================

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


# ============================================================
# GLOBAL FLOOD AREA
# ============================================================

total_gt_flood = (
    global_tp
    + global_fn
)

total_pred_flood = (
    global_tp
    + global_fp
)

total_valid_pixels = (
    global_tp
    + global_fp
    + global_fn
    + global_tn
)

global_gt_flood_percentage = (
    100.0
    * total_gt_flood
    / total_valid_pixels
)

global_pred_flood_percentage = (
    100.0
    * total_pred_flood
    / total_valid_pixels
)

if total_gt_flood > 0:

    global_area_ratio = (
        total_pred_flood
        / total_gt_flood
    )

else:

    global_area_ratio = np.nan


# ============================================================
# NO-FLOOD FALSE ALARMS
# ============================================================

if no_flood_valid_pixels > 0:

    no_flood_global_fpr = (
        no_flood_false_positive_pixels
        / no_flood_valid_pixels
    )

else:

    no_flood_global_fpr = 0.0


if no_flood_scene_count > 0:

    no_flood_mean_pred_percentage = (
        float(
            np.mean(
                no_flood_scene_predictions
            )
        )
    )

    no_flood_scene_false_alarm_rate = (
        float(
            np.mean(
                no_flood_scene_false_alarm_flags
            )
        )
    )

else:

    no_flood_mean_pred_percentage = 0.0

    no_flood_scene_false_alarm_rate = 0.0


# ============================================================
# SAFE SUMMARY FUNCTIONS
# ============================================================

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


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("POSITIVE WEIGHT = 5.0 TEST RESULTS")
print("=" * 70)

print()

print(
    f"Threshold              : "
    f"{THRESHOLD:.2f}"
)

print(
    f"Flood scenes           : "
    f"{flood_scene_count}"
)

print(
    f"No-flood scenes        : "
    f"{no_flood_scene_count}"
)

print()

print(
    "Global pixel-level metrics"
)

print(
    f"  IoU                  : "
    f"{global_iou:.4f} "
    f"({global_iou * 100:.2f}%)"
)

print(
    f"  Dice                 : "
    f"{global_dice:.4f} "
    f"({global_dice * 100:.2f}%)"
)

print(
    f"  Precision            : "
    f"{global_precision:.4f} "
    f"({global_precision * 100:.2f}%)"
)

print(
    f"  Recall               : "
    f"{global_recall:.4f} "
    f"({global_recall * 100:.2f}%)"
)

print()

print(
    "Global confusion counts"
)

print(
    f"  TP                   : "
    f"{global_tp:,}"
)

print(
    f"  FP                   : "
    f"{global_fp:,}"
)

print(
    f"  FN                   : "
    f"{global_fn:,}"
)

print(
    f"  TN                   : "
    f"{global_tn:,}"
)

print()

print(
    "Flood area"
)

print(
    f"  Ground-truth flood   : "
    f"{global_gt_flood_percentage:.2f}%"
)

print(
    f"  Predicted flood      : "
    f"{global_pred_flood_percentage:.2f}%"
)

print(
    f"  Area ratio           : "
    f"{global_area_ratio:.2f}x"
)

print()

print(
    "Mean positive-scene metrics"
)

print(
    f"  Mean IoU             : "
    f"{safe_mean(scene_ious) * 100:.2f}%"
)

print(
    f"  Median IoU           : "
    f"{safe_median(scene_ious) * 100:.2f}%"
)

print(
    f"  Mean Dice            : "
    f"{safe_mean(scene_dices) * 100:.2f}%"
)

print(
    f"  Mean Precision       : "
    f"{safe_mean(scene_precisions) * 100:.2f}%"
)

print(
    f"  Mean Recall          : "
    f"{safe_mean(scene_recalls) * 100:.2f}%"
)

print(
    f"  Mean FPR             : "
    f"{safe_mean(scene_fprs) * 100:.2f}%"
)

print(
    f"  Mean FNR             : "
    f"{safe_mean(scene_fnrs) * 100:.2f}%"
)

print(
    f"  Mean Area Ratio      : "
    f"{safe_mean(scene_area_ratios):.2f}x"
)

print(
    f"  Median Area Ratio    : "
    f"{safe_median(scene_area_ratios):.2f}x"
)

print()

print(
    "No-flood false alarms"
)

print(
    f"  Pixel-level FPR      : "
    f"{no_flood_global_fpr * 100:.2f}%"
)

print(
    f"  Mean predicted flood : "
    f"{no_flood_mean_pred_percentage:.2f}%"
)

print(
    f"  Scenes with FP       : "
    f"{no_flood_scene_false_alarm_rate * 100:.2f}%"
)

print()


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "evaluation_metrics.csv"
)

result = {
    "positive_weight": 5.0,
    "dice_weight": 0.5,
    "threshold": THRESHOLD,

    "global_iou": global_iou,
    "global_dice": global_dice,
    "global_precision": global_precision,
    "global_recall": global_recall,

    "tp": global_tp,
    "fp": global_fp,
    "fn": global_fn,
    "tn": global_tn,

    "global_gt_flood_percent":
        global_gt_flood_percentage,

    "global_pred_flood_percent":
        global_pred_flood_percentage,

    "global_area_ratio":
        global_area_ratio,

    "positive_scene_mean_iou":
        safe_mean(scene_ious),

    "positive_scene_median_iou":
        safe_median(scene_ious),

    "positive_scene_mean_dice":
        safe_mean(scene_dices),

    "positive_scene_mean_precision":
        safe_mean(scene_precisions),

    "positive_scene_mean_recall":
        safe_mean(scene_recalls),

    "positive_scene_mean_fpr":
        safe_mean(scene_fprs),

    "positive_scene_mean_fnr":
        safe_mean(scene_fnrs),

    "positive_scene_mean_area_ratio":
        safe_mean(scene_area_ratios),

    "positive_scene_median_area_ratio":
        safe_median(scene_area_ratios),

    "flood_scene_count":
        flood_scene_count,

    "no_flood_scene_count":
        no_flood_scene_count,

    "no_flood_global_fpr":
        no_flood_global_fpr,

    "no_flood_mean_pred_flood_percent":
        no_flood_mean_pred_percentage,

    "no_flood_scene_false_alarm_rate":
        no_flood_scene_false_alarm_rate,
}


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            result.keys()
        ),
    )

    writer.writeheader()

    writer.writerow(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print(
    "POSITIVE WEIGHT EVALUATION COMPLETE"
)
print("=" * 70)

print()

print(
    f"Results saved to:"
)

print(
    f"{csv_path}"
)

print()

print(
    "Experiment:"
)

print(
    "  Baseline positive weight = 9.52"
)

print(
    "  Experimental weight      = 5.00"
)

print(
    f"  Evaluation threshold     = "
    f"{THRESHOLD:.2f}"
)

print()

print(
    "Next step:"
)

print(
    "Compare these results directly "
    "against the frozen baseline."
)

print("=" * 70)