"""
ResQTwin - Canonical Scene-Level Model Comparison
==================================================

Compares:

    1. Baseline U-Net
    2. Positive-Weight=5 U-Net

Across thresholds:

    0.50, 0.70, 0.80, 0.90

The same Sen1Floods11 test scenes are used for both models.

This script is evaluation-only.
NO RETRAINING IS PERFORMED.

Outputs:
    ai/satellite/outputs/canonical_scene_comparison/

        scene_level_metrics.csv
        summary_metrics.csv

        model_comparison_iou.png
        model_comparison_dice.png
        model_comparison_precision_recall.png
        model_comparison_area_ratio.png
        model_comparison_no_flood_fpr.png
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from ai.satellite.src.dataset import (
    Sen1Floods11Dataset
)

from ai.satellite.models.unet import (
    UNet
)


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "unet_baseline_best.pt"
)

PW5_CHECKPOINT = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "positive_weight_5"
    / "best.pt"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Your baseline checkpoint may have a different actual path
# depending on the earlier checkpoint naming.
#
# If the baseline path above does not exist, the script will
# automatically search for a suitable baseline *best.pt
# checkpoint while avoiding the PW=5 checkpoint.
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "outputs"
    / "canonical_scene_comparison"
)


THRESHOLDS = [
    0.50,
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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("ResQTwin - Canonical Scene-Level Model Comparison")
print("=" * 80)

print()
print("Models:")
print("  1. Baseline U-Net")
print("  2. Positive-Weight=5 U-Net")

print()
print("Thresholds:")
print("  0.50, 0.70, 0.80, 0.90")

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# CHECKPOINT RESOLUTION
# ============================================================

def find_baseline_checkpoint():
    """
    Finds the original baseline checkpoint if the expected
    path does not exist.

    The Positive-Weight=5 checkpoint is explicitly excluded.
    """

    if BASELINE_CHECKPOINT.exists():
        return BASELINE_CHECKPOINT

    checkpoint_root = (
        PROJECT_ROOT
        / "ai"
        / "satellite"
        / "checkpoints"
    )

    candidates = []

    for path in checkpoint_root.rglob(
        "*best.pt"
    ):

        path_string = str(path).lower()

        if (
            "positive_weight_5"
            in path_string
        ):
            continue

        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "\nCould not find the baseline checkpoint.\n"
            f"Expected:\n{BASELINE_CHECKPOINT}\n\n"
            "Please check the checkpoint directory."
        )

    if len(candidates) > 1:

        print(
            "Multiple possible baseline checkpoints found:"
        )

        for candidate in candidates:
            print(
                f"  {candidate}"
            )

        print()

        # Prefer paths containing "baseline".
        baseline_candidates = [
            p
            for p in candidates
            if "baseline" in str(p).lower()
        ]

        if len(baseline_candidates) == 1:
            return baseline_candidates[0]

        raise RuntimeError(
            "\nMultiple baseline checkpoints found "
            "and the script cannot safely choose one.\n"
            "Please set BASELINE_CHECKPOINT explicitly."
        )

    return candidates[0]


BASELINE_CHECKPOINT = (
    find_baseline_checkpoint()
)


# ============================================================
# CHECK CHECKPOINTS
# ============================================================

if not BASELINE_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nBaseline checkpoint not found:\n"
        f"{BASELINE_CHECKPOINT}"
    )


if not PW5_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nPositive-Weight=5 checkpoint not found:\n"
        f"{PW5_CHECKPOINT}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("CHECKPOINTS")
print("=" * 80)

print()
print(
    f"Baseline : {BASELINE_CHECKPOINT}"
)

print(
    f"PW=5     : {PW5_CHECKPOINT}"
)

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

print()

dataset = Sen1Floods11Dataset(
    split="test"
)

print(
    f"Total test samples : {len(dataset)}"
)

print()


# ============================================================
# HELPER: EXTRACT SAMPLE
# ============================================================

def extract_input_and_target(sample):
    """
    Handles tuple/list and dictionary dataset formats.
    """

    if isinstance(
        sample,
        (tuple, list)
    ):

        image = sample[0]
        target = sample[1]

    elif isinstance(
        sample,
        dict
    ):

        image_keys = [
            "image",
            "images",
            "x",
            "input",
        ]

        target_keys = [
            "mask",
            "target",
            "label",
            "labels",
            "y",
        ]

        image = None
        target = None

        for key in image_keys:

            if key in sample:

                image = sample[key]
                break

        for key in target_keys:

            if key in sample:

                target = sample[key]
                break

        if (
            image is None
            or target is None
        ):

            raise ValueError(
                "Could not identify image/target "
                "keys in dataset sample."
            )

    else:

        raise TypeError(
            f"Unsupported sample type: "
            f"{type(sample)}"
        )

    return image, target


# ============================================================
# HELPER: PREPARE IMAGE
# ============================================================

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


# ============================================================
# HELPER: PREPARE TARGET
# ============================================================

def prepare_target(target):
    """
    Converts target to H x W NumPy array.
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
# HELPER: SAMPLE NAME
# ============================================================

def get_sample_name(
    dataset,
    index
):
    """
    Attempts to recover a useful scene name.
    """

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

    return name


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    model_name
):
    """
    Creates the U-Net and loads a checkpoint.
    """

    print(
        f"Loading {model_name}..."
    )

    model = UNet(
        in_channels=2,
        out_channels=1,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Resolve state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        clean_state_dict[
            key
        ] = value


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        clean_state_dict
    )

    model.to(DEVICE)

    model.eval()


    # --------------------------------------------------------
    # Print checkpoint information
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            print(
                f"  Epoch           : "
                f"{checkpoint['epoch']}"
            )

        if "validation_loss" in checkpoint:

            print(
                f"  Validation loss : "
                f"{checkpoint['validation_loss']:.6f}"
            )

        if "validation_dice" in checkpoint:

            print(
                f"  Validation Dice : "
                f"{checkpoint['validation_dice']:.6f}"
            )

        if "validation_iou" in checkpoint:

            print(
                f"  Validation IoU  : "
                f"{checkpoint['validation_iou']:.6f}"
            )

    print(
        f"  {model_name} loaded successfully."
    )

    print()

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

baseline_model = load_model(
    BASELINE_CHECKPOINT,
    "Baseline U-Net"
)

pw5_model = load_model(
    PW5_CHECKPOINT,
    "Positive-Weight=5 U-Net"
)


# ============================================================
# RUN INFERENCE
# ============================================================

print("=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)

print()
print(
    "Each model is evaluated once."
)

print(
    "Probability maps are cached in memory."
)

print()


baseline_probability_maps = []
pw5_probability_maps = []

target_maps = []
sample_names = []


with torch.no_grad():

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        image, target = (
            extract_input_and_target(
                sample
            )
        )

        image_tensor = (
            prepare_image(
                image
            )
            .to(DEVICE)
        )


        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_logits = (
            baseline_model(
                image_tensor
            )
        )

        baseline_probabilities = (
            torch.sigmoid(
                baseline_logits
            )
        )

        baseline_probabilities = (
            baseline_probabilities
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # ----------------------------------------------------
        # PW=5
        # ----------------------------------------------------

        pw5_logits = (
            pw5_model(
                image_tensor
            )
        )

        pw5_probabilities = (
            torch.sigmoid(
                pw5_logits
            )
        )

        pw5_probabilities = (
            pw5_probabilities
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target_np = (
            prepare_target(
                target
            )
        )


        baseline_probability_maps.append(
            baseline_probabilities
        )

        pw5_probability_maps.append(
            pw5_probabilities
        )

        target_maps.append(
            target_np
        )


        sample_names.append(
            get_sample_name(
                dataset,
                index
            )
        )


        if (
            (index + 1) % 10 == 0
            or index == 0
        ):

            print(
                f"Processed "
                f"{index + 1:3d}/"
                f"{len(dataset)} scenes"
            )


print()

print(
    "Inference complete."
)

print()


# ============================================================
# SCENE CLASSIFICATION
# ============================================================

print("=" * 80)
print("CANONICAL SCENE ACCOUNTING")
print("=" * 80)

print()

total_scenes = len(
    target_maps
)

valid_scenes = 0
all_ignore_scenes = 0
flood_scenes = 0
no_flood_scenes = 0

scene_metadata = []


for index, target in enumerate(
    target_maps
):

    valid = (
        target != -1
    )

    valid_pixel_count = int(
        np.sum(valid)
    )


    # --------------------------------------------------------
    # All-ignore scene
    # --------------------------------------------------------

    if valid_pixel_count == 0:

        all_ignore_scenes += 1

        scene_type = (
            "all_ignore"
        )


    else:

        valid_scenes += 1

        gt = (
            target[valid] > 0
        )

        gt_flood_pixels = int(
            np.sum(gt)
        )


        if gt_flood_pixels > 0:

            flood_scenes += 1

            scene_type = (
                "flood"
            )

        else:

            no_flood_scenes += 1

            scene_type = (
                "no_flood"
            )


    scene_metadata.append(
        {
            "index": index,
            "name": sample_names[index],
            "scene_type": scene_type,
            "valid_pixels": valid_pixel_count,
        }
    )


print(
    f"Total scenes        : "
    f"{total_scenes}"
)

print(
    f"Valid scenes        : "
    f"{valid_scenes}"
)

print(
    f"All-ignore scenes   : "
    f"{all_ignore_scenes}"
)

print(
    f"Flood scenes        : "
    f"{flood_scenes}"
)

print(
    f"No-flood scenes     : "
    f"{no_flood_scenes}"
)

print()

print(
    "These scene counts will be identical "
    "for both models and all thresholds."
)

print()


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_scene_metrics(
    probabilities,
    target,
    threshold,
    scene_type
):
    """
    Calculates metrics for a single scene.
    """

    valid = (
        target != -1
    )


    if not np.any(valid):

        return {
            "iou": np.nan,
            "dice": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "fpr": np.nan,
            "fnr": np.nan,
            "gt_flood_percent": np.nan,
            "pred_flood_percent": np.nan,
            "area_ratio": np.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "valid_pixels": 0,
            "scene_type": "all_ignore",
        }


    gt = (
        target[valid] > 0
    )

    pred = (
        probabilities[valid]
        >= threshold
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = int(
        np.sum(
            pred & gt
        )
    )

    fp = int(
        np.sum(
            pred & ~gt
        )
    )

    fn = int(
        np.sum(
            ~pred & gt
        )
    )

    tn = int(
        np.sum(
            ~pred & ~gt
        )
    )


    valid_pixels = (
        tp
        + fp
        + fn
        + tn
    )


    gt_flood_pixels = (
        tp + fn
    )

    pred_flood_pixels = (
        tp + fp
    )

    gt_non_flood_pixels = (
        tn + fp
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        tp + fp + fn
    )

    if union > 0:

        iou = (
            tp / union
        )

    else:

        iou = 1.0


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            / (tp + fp)
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            / (tp + fn)
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------

    if gt_non_flood_pixels > 0:

        fpr = (
            fp
            / gt_non_flood_pixels
        )

    else:

        fpr = 0.0


    # --------------------------------------------------------
    # FNR
    # --------------------------------------------------------

    if gt_flood_pixels > 0:

        fnr = (
            fn
            / gt_flood_pixels
        )

    else:

        fnr = 0.0


    # --------------------------------------------------------
    # Flood percentages
    # --------------------------------------------------------

    if valid_pixels > 0:

        gt_flood_percent = (
            100.0
            * gt_flood_pixels
            / valid_pixels
        )

        pred_flood_percent = (
            100.0
            * pred_flood_pixels
            / valid_pixels
        )

    else:

        gt_flood_percent = 0.0
        pred_flood_percent = 0.0


    # --------------------------------------------------------
    # Area ratio
    # --------------------------------------------------------

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
        "fpr": fpr,
        "fnr": fnr,

        "gt_flood_percent":
            gt_flood_percent,

        "pred_flood_percent":
            pred_flood_percent,

        "area_ratio":
            area_ratio,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "valid_pixels":
            valid_pixels,

        "scene_type":
            scene_type,
    }


# ============================================================
# GLOBAL RESULT STORAGE
# ============================================================

scene_results = []
summary_results = []


# ============================================================
# PROCESS EACH MODEL
# ============================================================

models = [

    (
        "baseline",
        baseline_probability_maps
    ),

    (
        "pw5",
        pw5_probability_maps
    ),

]


for model_name, probability_maps in models:

    print("=" * 80)

    if model_name == "baseline":

        print(
            "EVALUATING BASELINE U-NET"
        )

    else:

        print(
            "EVALUATING POSITIVE-WEIGHT=5 U-NET"
        )

    print("=" * 80)

    print()


    for threshold in THRESHOLDS:

        print(
            f"Threshold {threshold:.2f}"
        )


        # ----------------------------------------------------
        # Global counters
        # ----------------------------------------------------

        global_tp = 0
        global_fp = 0
        global_fn = 0
        global_tn = 0


        # ----------------------------------------------------
        # Scene-level lists
        # ----------------------------------------------------

        positive_ious = []
        positive_dices = []
        positive_precisions = []
        positive_recalls = []
        positive_fprs = []
        positive_fnrs = []
        positive_area_ratios = []

        no_flood_pred_percentages = []
        no_flood_fp_flags = []


        # ----------------------------------------------------
        # Process scenes
        # ----------------------------------------------------

        for index, (
            probabilities,
            target,
            metadata
        ) in enumerate(
            zip(
                probability_maps,
                target_maps,
                scene_metadata,
            )
        ):

            metrics = (
                calculate_scene_metrics(
                    probabilities,
                    target,
                    threshold,
                    metadata[
                        "scene_type"
                    ],
                )
            )


            # ------------------------------------------------
            # Skip all-ignore scenes
            # ------------------------------------------------

            if (
                metrics["scene_type"]
                == "all_ignore"
            ):

                continue


            # ------------------------------------------------
            # Global counters
            # ------------------------------------------------

            global_tp += (
                metrics["tp"]
            )

            global_fp += (
                metrics["fp"]
            )

            global_fn += (
                metrics["fn"]
            )

            global_tn += (
                metrics["tn"]
            )


            # ------------------------------------------------
            # Flood-containing scene
            # ------------------------------------------------

            if (
                metrics["scene_type"]
                == "flood"
            ):

                positive_ious.append(
                    metrics["iou"]
                )

                positive_dices.append(
                    metrics["dice"]
                )

                positive_precisions.append(
                    metrics["precision"]
                )

                positive_recalls.append(
                    metrics["recall"]
                )

                positive_fprs.append(
                    metrics["fpr"]
                )

                positive_fnrs.append(
                    metrics["fnr"]
                )

                positive_area_ratios.append(
                    metrics["area_ratio"]
                )


            # ------------------------------------------------
            # No-flood scene
            # ------------------------------------------------

            elif (
                metrics["scene_type"]
                == "no_flood"
            ):

                no_flood_pred_percentages.append(
                    metrics[
                        "pred_flood_percent"
                    ]
                )

                if metrics["fp"] > 0:

                    no_flood_fp_flags.append(
                        1
                    )

                else:

                    no_flood_fp_flags.append(
                        0
                    )


            # ------------------------------------------------
            # Save scene-level row
            # ------------------------------------------------

            scene_results.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "scene_index": index,
                    "scene_name": metadata["name"],
                    "scene_type": metadata[
                        "scene_type"
                    ],
                    "valid_pixels": metrics[
                        "valid_pixels"
                    ],
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "fpr": metrics["fpr"],
                    "fnr": metrics["fnr"],
                    "gt_flood_percent": metrics[
                        "gt_flood_percent"
                    ],
                    "pred_flood_percent": metrics[
                        "pred_flood_percent"
                    ],
                    "area_ratio": metrics[
                        "area_ratio"
                    ],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                }
            )


        # ====================================================
        # GLOBAL METRICS
        # ====================================================

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


        if (
            global_tp
            + global_fp
        ) > 0:

            global_precision = (
                global_tp
                / (
                    global_tp
                    + global_fp
                )
            )

        else:

            global_precision = 0.0


        if (
            global_tp
            + global_fn
        ) > 0:

            global_recall = (
                global_tp
                / (
                    global_tp
                    + global_fn
                )
            )

        else:

            global_recall = 0.0


        # ====================================================
        # GLOBAL AREA
        # ====================================================

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


        if total_gt_flood > 0:

            global_area_ratio = (
                total_pred_flood
                / total_gt_flood
            )

        else:

            global_area_ratio = np.nan


        if total_valid_pixels > 0:

            global_gt_flood_percent = (
                100.0
                * total_gt_flood
                / total_valid_pixels
            )

            global_pred_flood_percent = (
                100.0
                * total_pred_flood
                / total_valid_pixels
            )

        else:

            global_gt_flood_percent = 0.0
            global_pred_flood_percent = 0.0


        # ====================================================
        # POSITIVE-SCENE AGGREGATES
        # ====================================================

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


        # ====================================================
        # NO-FLOOD METRICS
        # ====================================================

        if len(
            no_flood_pred_percentages
        ) > 0:

            no_flood_mean_pred = (
                float(
                    np.mean(
                        no_flood_pred_percentages
                    )
                )
            )

            no_flood_false_alarm_rate = (
                float(
                    np.mean(
                        no_flood_fp_flags
                    )
                )
            )

        else:

            no_flood_mean_pred = 0.0

            no_flood_false_alarm_rate = (
                0.0
            )


        # ====================================================
        # SUMMARY ROW
        # ====================================================

        summary_row = {

            "model": model_name,

            "threshold": threshold,

            "total_scenes":
                total_scenes,

            "valid_scenes":
                valid_scenes,

            "all_ignore_scenes":
                all_ignore_scenes,

            "flood_scenes":
                flood_scenes,

            "no_flood_scenes":
                no_flood_scenes,


            # Global metrics
            "global_iou":
                global_iou,

            "global_dice":
                global_dice,

            "global_precision":
                global_precision,

            "global_recall":
                global_recall,


            # Counts
            "tp":
                global_tp,

            "fp":
                global_fp,

            "fn":
                global_fn,

            "tn":
                global_tn,


            # Flood area
            "global_gt_flood_percent":
                global_gt_flood_percent,

            "global_pred_flood_percent":
                global_pred_flood_percent,

            "global_area_ratio":
                global_area_ratio,


            # Positive scenes
            "mean_scene_iou":
                safe_mean(
                    positive_ious
                ),

            "median_scene_iou":
                safe_median(
                    positive_ious
                ),

            "mean_scene_dice":
                safe_mean(
                    positive_dices
                ),

            "median_scene_dice":
                safe_median(
                    positive_dices
                ),

            "mean_scene_precision":
                safe_mean(
                    positive_precisions
                ),

            "mean_scene_recall":
                safe_mean(
                    positive_recalls
                ),

            "mean_scene_fpr":
                safe_mean(
                    positive_fprs
                ),

            "mean_scene_fnr":
                safe_mean(
                    positive_fnrs
                ),

            "mean_scene_area_ratio":
                safe_mean(
                    positive_area_ratios
                ),

            "median_scene_area_ratio":
                safe_median(
                    positive_area_ratios
                ),


            # No-flood
            "no_flood_mean_pred_percent":
                no_flood_mean_pred,

            "no_flood_scene_false_alarm_rate":
                no_flood_false_alarm_rate,
        }


        summary_results.append(
            summary_row
        )


        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            f"  Global IoU       : "
            f"{global_iou * 100:.2f}%"
        )

        print(
            f"  Global Dice      : "
            f"{global_dice * 100:.2f}%"
        )

        print(
            f"  Precision        : "
            f"{global_precision * 100:.2f}%"
        )

        print(
            f"  Recall           : "
            f"{global_recall * 100:.2f}%"
        )

        print(
            f"  Mean scene IoU   : "
            f"{safe_mean(positive_ious) * 100:.2f}%"
        )

        print(
            f"  Median scene IoU : "
            f"{safe_median(positive_ious) * 100:.2f}%"
        )

        print(
            f"  Area ratio       : "
            f"{global_area_ratio:.2f}x"
        )

        print(
            f"  No-flood FPR     : "
            f"{no_flood_false_alarm_rate * 100:.2f}% "
            f"(scene-level)"
        )

        print()


# ============================================================
# SAVE SCENE-LEVEL CSV
# ============================================================

scene_csv_path = (
    OUTPUT_DIR
    / "scene_level_metrics.csv"
)


if scene_results:

    scene_fieldnames = list(
        scene_results[0].keys()
    )

    with open(
        scene_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=scene_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            scene_results
        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv_path = (
    OUTPUT_DIR
    / "summary_metrics.csv"
)


if summary_results:

    summary_fieldnames = list(
        summary_results[0].keys()
    )

    with open(
        summary_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary_results
        )


# ============================================================
# PREPARE PLOT DATA
# ============================================================

baseline_results = [
    r
    for r in summary_results
    if r["model"] == "baseline"
]

pw5_results = [
    r
    for r in summary_results
    if r["model"] == "pw5"
]


baseline_results.sort(
    key=lambda x: x["threshold"]
)

pw5_results.sort(
    key=lambda x: x["threshold"]
)


threshold_values = [
    r["threshold"]
    for r in baseline_results
]


# ============================================================
# PLOT HELPER
# ============================================================

def create_comparison_plot(
    baseline_key,
    pw5_key,
    title,
    ylabel,
    filename,
    multiply_by_100=False,
):

    baseline_values = [
        r[baseline_key]
        for r in baseline_results
    ]

    pw5_values = [
        r[pw5_key]
        for r in pw5_results
    ]


    if multiply_by_100:

        baseline_values = [
            value * 100
            for value in baseline_values
        ]

        pw5_values = [
            value * 100
            for value in pw5_values
        ]


    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(
        threshold_values,
        baseline_values,
        marker="o",
        label="Baseline U-Net",
    )


    plt.plot(
        threshold_values,
        pw5_values,
        marker="o",
        label="PW=5 U-Net",
    )


    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
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


    path = (
        OUTPUT_DIR
        / filename
    )


    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT 1: IOU
# ============================================================

create_comparison_plot(
    "global_iou",
    "global_iou",
    "Baseline vs PW=5 - Global IoU",
    "Global IoU (%)",
    "model_comparison_iou.png",
    True,
)


# ============================================================
# PLOT 2: DICE
# ============================================================

create_comparison_plot(
    "global_dice",
    "global_dice",
    "Baseline vs PW=5 - Global Dice",
    "Global Dice (%)",
    "model_comparison_dice.png",
    True,
)


# ============================================================
# PLOT 3: PRECISION / RECALL
# ============================================================

plt.figure(
    figsize=(9, 6)
)


baseline_precision = [
    r["global_precision"] * 100
    for r in baseline_results
]

baseline_recall = [
    r["global_recall"] * 100
    for r in baseline_results
]

pw5_precision = [
    r["global_precision"] * 100
    for r in pw5_results
]

pw5_recall = [
    r["global_recall"] * 100
    for r in pw5_results
]


plt.plot(
    threshold_values,
    baseline_precision,
    marker="o",
    label="Baseline Precision",
)

plt.plot(
    threshold_values,
    baseline_recall,
    marker="o",
    label="Baseline Recall",
)

plt.plot(
    threshold_values,
    pw5_precision,
    marker="o",
    label="PW=5 Precision",
)

plt.plot(
    threshold_values,
    pw5_recall,
    marker="o",
    label="PW=5 Recall",
)


plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Baseline vs PW=5 - Precision and Recall"
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


plt.savefig(
    OUTPUT_DIR
    / "model_comparison_precision_recall.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 4: AREA RATIO
# ============================================================

plt.figure(
    figsize=(9, 6)
)


baseline_area = [
    r["global_area_ratio"]
    for r in baseline_results
]

pw5_area = [
    r["global_area_ratio"]
    for r in pw5_results
]


plt.plot(
    threshold_values,
    baseline_area,
    marker="o",
    label="Baseline U-Net",
)

plt.plot(
    threshold_values,
    pw5_area,
    marker="o",
    label="PW=5 U-Net",
)


plt.axhline(
    1.0,
    linestyle="--",
    label="Ideal Area Ratio = 1.0x",
)


plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "Predicted / Ground-Truth Flood Area"
)

plt.title(
    "Baseline vs PW=5 - Flood Area Ratio"
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


plt.savefig(
    OUTPUT_DIR
    / "model_comparison_area_ratio.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 5: NO-FLOOD FPR
# ============================================================

plt.figure(
    figsize=(9, 6)
)


baseline_no_flood = [
    r[
        "no_flood_scene_false_alarm_rate"
    ] * 100
    for r in baseline_results
]

pw5_no_flood = [
    r[
        "no_flood_scene_false_alarm_rate"
    ] * 100
    for r in pw5_results
]


plt.plot(
    threshold_values,
    baseline_no_flood,
    marker="o",
    label="Baseline U-Net",
)

plt.plot(
    threshold_values,
    pw5_no_flood,
    marker="o",
    label="PW=5 U-Net",
)


plt.xlabel(
    "Probability Threshold"
)

plt.ylabel(
    "No-Flood Scenes with False Alarm (%)"
)

plt.title(
    "Baseline vs PW=5 - No-Flood Scene False Alarms"
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


plt.savefig(
    OUTPUT_DIR
    / "model_comparison_no_flood_fpr.png",
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("CANONICAL COMPARISON COMPLETE")
print("=" * 80)

print()

print(
    "Canonical scene accounting:"
)

print(
    f"  Total scenes      : "
    f"{total_scenes}"
)

print(
    f"  Valid scenes      : "
    f"{valid_scenes}"
)

print(
    f"  All-ignore scenes : "
    f"{all_ignore_scenes}"
)

print(
    f"  Flood scenes      : "
    f"{flood_scenes}"
)

print(
    f"  No-flood scenes   : "
    f"{no_flood_scenes}"
)

print()


# ============================================================
# COMPARISON TABLE
# ============================================================

print("=" * 80)
print("GLOBAL COMPARISON")
print("=" * 80)

print()

print(
    f"{'Threshold':<12}"
    f"{'Model':<12}"
    f"{'IoU':>10}"
    f"{'Dice':>10}"
    f"{'Prec.':>10}"
    f"{'Recall':>10}"
    f"{'Area':>10}"
)

print("-" * 80)


for threshold in THRESHOLDS:

    baseline = next(
        r
        for r in baseline_results
        if r["threshold"]
        == threshold
    )

    pw5 = next(
        r
        for r in pw5_results
        if r["threshold"]
        == threshold
    )


    print(
        f"{threshold:<12.2f}"
        f"{'Baseline':<12}"
        f"{baseline['global_iou'] * 100:>9.2f}%"
        f"{baseline['global_dice'] * 100:>9.2f}%"
        f"{baseline['global_precision'] * 100:>9.2f}%"
        f"{baseline['global_recall'] * 100:>9.2f}%"
        f"{baseline['global_area_ratio']:>9.2f}x"
    )


    print(
        f"{'':<12}"
        f"{'PW=5':<12}"
        f"{pw5['global_iou'] * 100:>9.2f}%"
        f"{pw5['global_dice'] * 100:>9.2f}%"
        f"{pw5['global_precision'] * 100:>9.2f}%"
        f"{pw5['global_recall'] * 100:>9.2f}%"
        f"{pw5['global_area_ratio']:>9.2f}x"
    )

    print()


# ============================================================
# BEST CONFIGURATIONS
# ============================================================

print("=" * 80)
print("BEST CONFIGURATIONS")
print("=" * 80)

print()


for model_name, results in [
    ("Baseline", baseline_results),
    ("PW=5", pw5_results),
]:

    best_iou = max(
        results,
        key=lambda x:
            x["global_iou"]
    )

    best_dice = max(
        results,
        key=lambda x:
            x["global_dice"]
    )

    best_scene_iou = max(
        results,
        key=lambda x:
            x["mean_scene_iou"]
    )


    print(
        f"{model_name}:"
    )

    print(
        f"  Best global IoU : "
        f"{best_iou['global_iou'] * 100:.2f}% "
        f"@ threshold "
        f"{best_iou['threshold']:.2f}"
    )

    print(
        f"  Best global Dice: "
        f"{best_dice['global_dice'] * 100:.2f}% "
        f"@ threshold "
        f"{best_dice['threshold']:.2f}"
    )

    print(
        f"  Best mean scene IoU: "
        f"{best_scene_iou['mean_scene_iou'] * 100:.2f}% "
        f"@ threshold "
        f"{best_scene_iou['threshold']:.2f}"
    )

    print()


# ============================================================
# OUTPUT FILES
# ============================================================

print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print()

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR}"
)

print()

print(
    f"  - {scene_csv_path.name}"
)

print(
    f"  - {summary_csv_path.name}"
)

print(
    "  - model_comparison_iou.png"
)

print(
    "  - model_comparison_dice.png"
)

print(
    "  - model_comparison_precision_recall.png"
)

print(
    "  - model_comparison_area_ratio.png"
)

print(
    "  - model_comparison_no_flood_fpr.png"
)

print()

print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print()

print(
    "This is an evaluation-only experiment."
)

print(
    "No model was retrained."
)

print(
    "Do NOT select the final production threshold "
    "until the scene-level results have been reviewed."
)

print(
    "The purpose of this experiment is to determine "
    "whether PW=5 improves performance consistently "
    "across individual scenes and operating thresholds."
)

print()

print("=" * 80)