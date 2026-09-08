"""
ResQTwin — Complete Test-Set Failure Analysis

Runs the frozen epoch-40 baseline U-Net over the complete
Sen1Floods11 hand-labeled test set and produces:

1. Per-chip segmentation metrics
2. False-positive / false-negative analysis
3. Flood-area overprediction analysis
4. Best / representative / worst cases
5. TP / FP / FN / Ignore diagnostic visualizations

No training or dataset modification is performed.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai.satellite.models.unet import UNet
from ai.satellite.src.dataset import Sen1Floods11Dataset


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "outputs"
    / "test_analysis"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------

def find_checkpoint():
    """Find the baseline .pt checkpoint."""

    checkpoints = list(
        CHECKPOINT_DIR.rglob("*.pt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint found under "
            f"{CHECKPOINT_DIR}"
        )

    baseline = [
        path
        for path in checkpoints
        if "baseline" in path.name.lower()
    ]

    if baseline:
        return baseline[0]

    return checkpoints[0]


def load_model(checkpoint_path):
    """Load the frozen U-Net checkpoint."""

    model = UNet(
        in_channels=2,
        out_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        epoch = checkpoint.get(
            "epoch",
            None
        )

    else:

        model.load_state_dict(
            checkpoint
        )

        epoch = None

    model.eval()

    return model, epoch


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(
    target,
    prediction
):
    """
    Calculate per-chip segmentation metrics.

    Target values:
        -1 = ignore
         0 = non-flood
         1 = flood

    Prediction values:
         0 = non-flood
         1 = flood

    Metrics:
        IoU
        Dice
        Precision
        Recall
        FPR
        FNR
        Flood-area ratio
    """

    # Only valid pixels participate in evaluation.
    valid = target != -1

    target_valid = target[valid]
    prediction_valid = prediction[valid]

    # Confusion matrix components.
    tp = int(
        (
            (prediction_valid == 1)
            & (target_valid == 1)
        ).sum()
    )

    fp = int(
        (
            (prediction_valid == 1)
            & (target_valid == 0)
        ).sum()
    )

    fn = int(
        (
            (prediction_valid == 0)
            & (target_valid == 1)
        ).sum()
    )

    tn = int(
        (
            (prediction_valid == 0)
            & (target_valid == 0)
        ).sum()
    )

    valid_pixels = int(
        valid.sum()
    )

    # Flood pixel counts.
    gt_flood_pixels = int(
        (target_valid == 1).sum()
    )

    predicted_flood_pixels = int(
        (prediction_valid == 1).sum()
    )

    # Flood percentages.
    gt_flood_pct = (
        100.0
        * gt_flood_pixels
        / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    predicted_flood_pct = (
        100.0
        * predicted_flood_pixels
        / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # IoU
    # ---------------------------------------------------------------

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator > 0:

        iou = (
            tp
            / iou_denominator
        )

    elif (
        gt_flood_pixels == 0
        and predicted_flood_pixels == 0
    ):

        # Perfect no-flood prediction.
        iou = 1.0

    else:

        iou = 0.0

    # ---------------------------------------------------------------
    # Dice
    # ---------------------------------------------------------------

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator > 0:

        dice = (
            2 * tp
            / dice_denominator
        )

    elif (
        gt_flood_pixels == 0
        and predicted_flood_pixels == 0
    ):

        dice = 1.0

    else:

        dice = 0.0

    # ---------------------------------------------------------------
    # Precision
    # ---------------------------------------------------------------

    precision_denominator = (
        tp + fp
    )

    if precision_denominator > 0:

        precision = (
            tp
            / precision_denominator
        )

    elif gt_flood_pixels == 0:

        # No predicted flood and no ground-truth flood.
        precision = 1.0

    else:

        precision = 0.0

    # ---------------------------------------------------------------
    # Recall
    # ---------------------------------------------------------------

    recall_denominator = (
        tp + fn
    )

    if recall_denominator > 0:

        recall = (
            tp
            / recall_denominator
        )

    elif gt_flood_pixels == 0:

        # Recall is not meaningful for a no-flood scene,
        # but 1.0 is convenient for the perfect no-flood case.
        recall = 1.0

    else:

        recall = 0.0

    # ---------------------------------------------------------------
    # False Positive Rate
    # ---------------------------------------------------------------

    fpr_denominator = (
        fp + tn
    )

    fpr = (
        fp
        / fpr_denominator
        if fpr_denominator > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # False Negative Rate
    # ---------------------------------------------------------------

    fnr_denominator = (
        fn + tp
    )

    fnr = (
        fn
        / fnr_denominator
        if fnr_denominator > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # Flood-area ratio
    # ---------------------------------------------------------------

    if gt_flood_pixels > 0:

        flood_area_ratio = (
            predicted_flood_pixels
            / gt_flood_pixels
        )

    elif predicted_flood_pixels > 0:

        # Ground truth has zero flood but model predicts flood.
        flood_area_ratio = float("inf")

    else:

        # Both are zero.
        flood_area_ratio = 1.0

    return {
        "valid_pixels": valid_pixels,
        "gt_flood_pixels": gt_flood_pixels,
        "predicted_flood_pixels": predicted_flood_pixels,
        "gt_flood_pct": gt_flood_pct,
        "predicted_flood_pct": predicted_flood_pct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "flood_area_ratio": flood_area_ratio,
        "has_ground_truth_flood": (
            gt_flood_pixels > 0
        ),
    }


# ---------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------

def country_from_filename(
    image_name
):
    """
    Extract the first filename component.

    Example:
        Ghana_313799_S1Hand.tif
        -> Ghana
    """

    return image_name.split(
        "_"
    )[0]


# ---------------------------------------------------------------------
# FULL TEST-SET INFERENCE
# ---------------------------------------------------------------------

def run_analysis(
    model,
    dataset
):
    """Run frozen-model inference over all test chips."""

    results = []

    print()
    print(
        "Running inference over complete test set..."
    )
    print()

    with torch.no_grad():

        for index in range(
            len(dataset)
        ):

            image, target = dataset[index]

            image_tensor = (
                image
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(
                image_tensor
            )

            probability = torch.sigmoid(
                logits
            )[0, 0]

            prediction = (
                probability >= THRESHOLD
            ).cpu().numpy().astype(
                np.uint8
            )

            target_np = (
                target
                .cpu()
                .numpy()
            )

            image_name, label_name = (
                dataset.samples[index]
            )

            metrics = calculate_metrics(
                target_np,
                prediction
            )

            result = {
                "index": index,
                "image_name": image_name,
                "label_name": label_name,
                "country": country_from_filename(
                    image_name
                ),
                **metrics,
            }

            results.append(
                result
            )

            print(
                f"[{index:02d}/"
                f"{len(dataset) - 1:02d}] "
                f"{image_name} | "
                f"IoU="
                f"{metrics['iou']:.4f} | "
                f"Dice="
                f"{metrics['dice']:.4f} | "
                f"Precision="
                f"{metrics['precision']:.4f} | "
                f"Recall="
                f"{metrics['recall']:.4f}"
            )

    return results


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def save_csv(
    results
):
    """Save all per-chip metrics to CSV."""

    output_path = (
        OUTPUT_DIR
        / "test_set_metrics.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    return output_path


# ---------------------------------------------------------------------
# CASE SELECTION
# ---------------------------------------------------------------------

def select_cases(
    results
):
    """
    Select meaningful cases.

    Flood-containing scenes are analyzed separately
    from no-flood scenes.
    """

    positive_cases = [
        result
        for result in results
        if result[
            "has_ground_truth_flood"
        ]
    ]

    no_flood_cases = [
        result
        for result in results
        if not result[
            "has_ground_truth_flood"
        ]
    ]

    # ---------------------------------------------------------------
    # IoU rankings for flood-containing scenes
    # ---------------------------------------------------------------

    by_iou = sorted(
        positive_cases,
        key=lambda result:
        result["iou"]
    )

    worst = by_iou[:5]

    best = by_iou[
        -5:
    ][::-1]

    median_position = (
        len(by_iou) // 2
    )

    representative = by_iou[
        max(
            0,
            median_position - 2
        ):
        median_position + 3
    ]

    # ---------------------------------------------------------------
    # False positives
    # ---------------------------------------------------------------

    highest_fp = sorted(
        positive_cases,
        key=lambda result:
        result["fp"],
        reverse=True
    )[:5]

    highest_fpr = sorted(
        positive_cases,
        key=lambda result:
        result["fpr"],
        reverse=True
    )[:5]

    # ---------------------------------------------------------------
    # False negatives
    # ---------------------------------------------------------------

    highest_fn = sorted(
        positive_cases,
        key=lambda result:
        result["fn"],
        reverse=True
    )[:5]

    highest_fnr = sorted(
        positive_cases,
        key=lambda result:
        result["fnr"],
        reverse=True
    )[:5]

    # ---------------------------------------------------------------
    # No-flood scenes
    # ---------------------------------------------------------------

    no_flood_fp = sorted(
        no_flood_cases,
        key=lambda result:
        result[
            "predicted_flood_pct"
        ],
        reverse=True
    )[:5]

    return {
        "best": best,
        "representative": representative,
        "worst": worst,
        "highest_fp": highest_fp,
        "highest_fpr": highest_fpr,
        "highest_fn": highest_fn,
        "highest_fnr": highest_fnr,
        "no_flood_fp": no_flood_fp,
        "positive_cases": positive_cases,
        "no_flood_cases": no_flood_cases,
    }


# ---------------------------------------------------------------------
# PRINT CASES
# ---------------------------------------------------------------------

def print_cases(
    title,
    cases
):
    """Print selected cases."""

    print()
    print(
        "=" * 110
    )
    print(title)
    print(
        "=" * 110
    )

    for result in cases:

        ratio = result[
            "flood_area_ratio"
        ]

        if np.isinf(ratio):

            ratio_text = "INF"

        else:

            ratio_text = (
                f"{ratio:.2f}x"
            )

        print(
            f"Index "
            f"{result['index']:02d} | "
            f"{result['image_name']} | "
            f"IoU="
            f"{result['iou']:.4f} | "
            f"Dice="
            f"{result['dice']:.4f} | "
            f"P="
            f"{result['precision']:.4f} | "
            f"R="
            f"{result['recall']:.4f} | "
            f"FPR="
            f"{result['fpr']:.4f} | "
            f"FNR="
            f"{result['fnr']:.4f} | "
            f"GT="
            f"{result['gt_flood_pct']:.2f}% | "
            f"Pred="
            f"{result['predicted_flood_pct']:.2f}% | "
            f"AreaRatio="
            f"{ratio_text} | "
            f"FP="
            f"{result['fp']} | "
            f"FN="
            f"{result['fn']}"
        )


# ---------------------------------------------------------------------
# TP / FP / FN VISUALIZATION
# ---------------------------------------------------------------------

def create_error_visualization(
    model,
    dataset,
    result,
    category
):
    """
    Create a four-panel diagnostic image:

        VV | VH | Ground Truth | TP/FP/FN

    Error map:
        White = TP
        Red   = FP
        Blue  = FN
        Black = TN
        Gray  = Ignore
    """

    index = result[
        "index"
    ]

    image, target = dataset[
        index
    ]

    image_tensor = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0]

    prediction = (
        probability >= THRESHOLD
    ).cpu().numpy().astype(
        np.uint8
    )

    image_np = (
        image
        .cpu()
        .numpy()
    )

    target_np = (
        target
        .cpu()
        .numpy()
    )

    valid = (
        target_np != -1
    )

    # ---------------------------------------------------------------
    # Error map
    # ---------------------------------------------------------------

    error_map = np.zeros(
        (
            target_np.shape[0],
            target_np.shape[1],
            3
        ),
        dtype=np.float32
    )

    # TP
    tp_mask = (
        (prediction == 1)
        & (target_np == 1)
        & valid
    )

    # FP
    fp_mask = (
        (prediction == 1)
        & (target_np == 0)
        & valid
    )

    # FN
    fn_mask = (
        (prediction == 0)
        & (target_np == 1)
        & valid
    )

    # TP = white
    error_map[
        tp_mask
    ] = [1.0, 1.0, 1.0]

    # FP = red
    error_map[
        fp_mask
    ] = [1.0, 0.0, 0.0]

    # FN = blue
    error_map[
        fn_mask
    ] = [0.0, 0.4, 1.0]

    # TN remains black.

    # Ignore = gray
    error_map[
        ~valid
    ] = [0.35, 0.35, 0.35]

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    vv = image_np[0]
    vh = image_np[1]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    axes[0].imshow(
        vv,
        cmap="gray"
    )

    axes[0].set_title(
        "Sentinel-1 VV"
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        vh,
        cmap="gray"
    )

    axes[1].set_title(
        "Sentinel-1 VH"
    )

    axes[1].axis(
        "off"
    )

    axes[2].imshow(
        target_np,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Ground Truth"
    )

    axes[2].axis(
        "off"
    )

    axes[3].imshow(
        error_map
    )

    axes[3].set_title(
        "TP / FP / FN Diagnostic"
    )

    axes[3].axis(
        "off"
    )

    fig.suptitle(
        f"{category.upper()} | "
        f"Sample {index} | "
        f"{result['image_name']} | "
        f"IoU={result['iou']:.4f}",
        fontsize=14
    )

    fig.text(
        0.5,
        0.02,
        "White = TP    "
        "Red = FP    "
        "Blue = FN    "
        "Black = TN    "
        "Gray = Ignore",
        ha="center",
        fontsize=11
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            0.95
        ]
    )

    output_path = (
        OUTPUT_DIR
        / f"{category}_"
        f"{index:03d}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ---------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------

def print_summary(
    results
):
    """Print overall dataset statistics."""

    positive_cases = [
        result
        for result in results
        if result[
            "has_ground_truth_flood"
        ]
    ]

    no_flood_cases = [
        result
        for result in results
        if not result[
            "has_ground_truth_flood"
        ]
    ]

    positive_ious = np.array(
        [
            result["iou"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    positive_dice = np.array(
        [
            result["dice"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    positive_precision = np.array(
        [
            result["precision"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    positive_recall = np.array(
        [
            result["recall"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    positive_fpr = np.array(
        [
            result["fpr"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    positive_fnr = np.array(
        [
            result["fnr"]
            for result in positive_cases
        ],
        dtype=np.float64
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TEST-SET SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        f"Total test scenes: "
        f"{len(results)}"
    )

    print(
        f"Flood-containing scenes: "
        f"{len(positive_cases)}"
    )

    print(
        f"No-flood scenes: "
        f"{len(no_flood_cases)}"
    )

    if len(positive_cases) > 0:

        print()
        print(
            "Flood-containing scenes "
            "(mean per-chip metrics):"
        )

        print(
            f"Mean IoU:       "
            f"{positive_ious.mean():.4f}"
        )

        print(
            f"Median IoU:     "
            f"{np.median(positive_ious):.4f}"
        )

        print(
            f"Mean Dice:      "
            f"{positive_dice.mean():.4f}"
        )

        print(
            f"Mean Precision: "
            f"{positive_precision.mean():.4f}"
        )

        print(
            f"Mean Recall:    "
            f"{positive_recall.mean():.4f}"
        )

        print(
            f"Mean FPR:       "
            f"{positive_fpr.mean():.4f}"
        )

        print(
            f"Mean FNR:       "
            f"{positive_fnr.mean():.4f}"
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=" * 80
    )

    print(
        "ResQTwin — Complete "
        "Test-Set Failure Analysis"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    checkpoint_path = (
        find_checkpoint()
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    model, epoch = (
        load_model(
            checkpoint_path
        )
    )

    if epoch is not None:

        print(
            f"Checkpoint epoch: "
            f"{epoch}"
        )

    # Use the existing dataset implementation.
    dataset = Sen1Floods11Dataset(
        split="test"
    )

    print(
        f"Test samples: "
        f"{len(dataset)}"
    )

    # ---------------------------------------------------------------
    # Run complete test-set inference.
    # ---------------------------------------------------------------

    results = run_analysis(
        model,
        dataset
    )

    # ---------------------------------------------------------------
    # Save CSV.
    # ---------------------------------------------------------------

    csv_path = save_csv(
        results
    )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------

    print_summary(
        results
    )

    # ---------------------------------------------------------------
    # Select cases.
    # ---------------------------------------------------------------

    cases = select_cases(
        results
    )

    # ---------------------------------------------------------------
    # Print rankings.
    # ---------------------------------------------------------------

    print_cases(
        "BEST 5 — Highest IoU "
        "(Flood-Containing Scenes)",
        cases["best"]
    )

    print_cases(
        "REPRESENTATIVE 5 — "
        "Around Median IoU",
        cases["representative"]
    )

    print_cases(
        "WORST 5 — Lowest IoU "
        "(Flood-Containing Scenes)",
        cases["worst"]
    )

    print_cases(
        "HIGHEST RAW FALSE-POSITIVE CASES",
        cases["highest_fp"]
    )

    print_cases(
        "HIGHEST FALSE-POSITIVE RATE CASES",
        cases["highest_fpr"]
    )

    print_cases(
        "HIGHEST RAW FALSE-NEGATIVE CASES",
        cases["highest_fn"]
    )

    print_cases(
        "HIGHEST FALSE-NEGATIVE RATE CASES",
        cases["highest_fnr"]
    )

    print_cases(
        "NO-FLOOD SCENES WITH "
        "HIGHEST PREDICTED FLOOD",
        cases["no_flood_fp"]
    )

    # ---------------------------------------------------------------
    # Diagnostic visualizations.
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "Creating diagnostic visualizations..."
    )

    print(
        "=" * 80
    )

    selected = {}

    categories_for_visualization = [
        "best",
        "representative",
        "worst",
        "highest_fpr",
        "highest_fnr",
        "no_flood_fp",
    ]

    for category in (
        categories_for_visualization
    ):

        for result in cases[
            category
        ]:

            index = result[
                "index"
            ]

            if index not in selected:

                selected[index] = category

    for index, category in (
        selected.items()
    ):

        result = next(
            result
            for result in results
            if result["index"] == index
        )

        output_path = (
            create_error_visualization(
                model=model,
                dataset=dataset,
                result=result,
                category=category
            )
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

    # ---------------------------------------------------------------
    # Final output.
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "Analysis complete."
    )

    print(
        "=" * 80
    )

    print(
        f"Metrics CSV: "
        f"{csv_path}"
    )

    print(
        f"Diagnostics: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()