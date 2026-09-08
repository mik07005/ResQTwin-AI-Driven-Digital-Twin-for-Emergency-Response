from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "ai" / "satellite" / "outputs" / "canonical_scene_comparison" / "scene_level_metrics.csv"
OUTPUT_DIR = ROOT / "ai" / "satellite" / "outputs" / "canonical_scene_comparison" / "final_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# 1. Compare PW=5 vs baseline at threshold 0.80
# ---------------------------------------------------------

base = df[
    (df["model"] == "baseline") &
    (df["threshold"] == 0.8)
].copy()

pw5 = df[
    (df["model"] == "pw5") &
    (df["threshold"] == 0.8)
].copy()

merged = base.merge(
    pw5,
    on=["scene_index", "scene_name", "scene_type"],
    suffixes=("_baseline", "_pw5")
)

# Only flood scenes for scene-level improvement analysis
flood = merged[merged["scene_type"] == "flood"].copy()

flood["iou_change"] = flood["iou_pw5"] - flood["iou_baseline"]
flood["dice_change"] = flood["dice_pw5"] - flood["dice_baseline"]
flood["precision_change"] = flood["precision_pw5"] - flood["precision_baseline"]
flood["recall_change"] = flood["recall_pw5"] - flood["recall_baseline"]
flood["area_ratio_change"] = (
    flood["area_ratio_pw5"] - flood["area_ratio_baseline"]
)

# ---------------------------------------------------------
# 2. Basic scene-level statistics
# ---------------------------------------------------------

improved = (flood["iou_change"] > 0).sum()
worsened = (flood["iou_change"] < 0).sum()
unchanged = (flood["iou_change"] == 0).sum()

median_iou_change = flood["iou_change"].median()
mean_iou_change = flood["iou_change"].mean()

print("\n" + "=" * 70)
print("FINAL SATELLITE SEGMENTATION ANALYSIS")
print("=" * 70)

print(f"\nFlood scenes analysed: {len(flood)}")
print(f"Improved IoU: {improved} ({improved / len(flood) * 100:.2f}%)")
print(f"Worsened IoU: {worsened} ({worsened / len(flood) * 100:.2f}%)")
print(f"Unchanged IoU: {unchanged}")
print(f"Mean IoU change: {mean_iou_change * 100:+.2f} pp")
print(f"Median IoU change: {median_iou_change * 100:+.2f} pp")

# ---------------------------------------------------------
# 3. Top 10 improvements
# ---------------------------------------------------------

top_improvements = flood.sort_values(
    "iou_change", ascending=False
).head(10)

top_improvements[
    [
        "scene_index",
        "scene_name",
        "iou_baseline",
        "iou_pw5",
        "iou_change",
        "recall_baseline",
        "recall_pw5",
        "area_ratio_baseline",
        "area_ratio_pw5",
    ]
].to_csv(
    OUTPUT_DIR / "top_10_improvements.csv",
    index=False
)

print("\n" + "-" * 70)
print("TOP 10 IoU IMPROVEMENTS")
print("-" * 70)

for _, r in top_improvements.iterrows():
    print(
        f"{int(r.scene_index):02d} {r.scene_name:15s} "
        f"{r.iou_baseline:.3f} -> {r.iou_pw5:.3f} "
        f"({r.iou_change * 100:+.2f} pp)"
    )

# ---------------------------------------------------------
# 4. Top 10 degradations
# ---------------------------------------------------------

top_degradations = flood.sort_values(
    "iou_change", ascending=True
).head(10)

top_degradations[
    [
        "scene_index",
        "scene_name",
        "iou_baseline",
        "iou_pw5",
        "iou_change",
        "recall_baseline",
        "recall_pw5",
        "area_ratio_baseline",
        "area_ratio_pw5",
    ]
].to_csv(
    OUTPUT_DIR / "top_10_degradations.csv",
    index=False
)

print("\n" + "-" * 70)
print("TOP 10 IoU DEGRADATIONS")
print("-" * 70)

for _, r in top_degradations.iterrows():
    print(
        f"{int(r.scene_index):02d} {r.scene_name:15s} "
        f"{r.iou_baseline:.3f} -> {r.iou_pw5:.3f} "
        f"({r.iou_change * 100:+.2f} pp)"
    )

# ---------------------------------------------------------
# 5. Catastrophic overprediction
# ---------------------------------------------------------

# PW5 predicts substantially more flooded area than GT
flood["pw5_overprediction_factor"] = (
    flood["pred_flood_percent_pw5"] /
    flood["gt_flood_percent_pw5"].replace(0, np.nan)
)

catastrophic_overprediction = flood.sort_values(
    "pw5_overprediction_factor",
    ascending=False
).head(10)

catastrophic_overprediction[
    [
        "scene_index",
        "scene_name",
        "gt_flood_percent_pw5",
        "pred_flood_percent_pw5",
        "area_ratio_pw5",
        "iou_pw5",
        "precision_pw5",
        "recall_pw5",
    ]
].to_csv(
    OUTPUT_DIR / "catastrophic_overprediction.csv",
    index=False
)

print("\n" + "-" * 70)
print("CATASTROPHIC OVERPREDICTION — TOP 10")
print("-" * 70)

for _, r in catastrophic_overprediction.iterrows():
    print(
        f"{int(r.scene_index):02d} {r.scene_name:15s} "
        f"GT={r.gt_flood_percent_pw5:.2f}% "
        f"Pred={r.pred_flood_percent_pw5:.2f}% "
        f"Ratio={r.area_ratio_pw5:.2f}x "
        f"IoU={r.iou_pw5:.3f}"
    )

# ---------------------------------------------------------
# 6. Severe underprediction
# ---------------------------------------------------------

flood["pw5_underprediction_factor"] = (
    flood["pred_flood_percent_pw5"] /
    flood["gt_flood_percent_pw5"].replace(0, np.nan)
)

severe_underprediction = flood.sort_values(
    "pw5_underprediction_factor",
    ascending=True
).head(10)

severe_underprediction[
    [
        "scene_index",
        "scene_name",
        "gt_flood_percent_pw5",
        "pred_flood_percent_pw5",
        "area_ratio_pw5",
        "iou_pw5",
        "precision_pw5",
        "recall_pw5",
    ]
].to_csv(
    OUTPUT_DIR / "severe_underprediction.csv",
    index=False
)

print("\n" + "-" * 70)
print("SEVERE UNDERPREDICTION — TOP 10")
print("-" * 70)

for _, r in severe_underprediction.iterrows():
    print(
        f"{int(r.scene_index):02d} {r.scene_name:15s} "
        f"GT={r.gt_flood_percent_pw5:.2f}% "
        f"Pred={r.pred_flood_percent_pw5:.2f}% "
        f"Ratio={r.area_ratio_pw5:.2f}x "
        f"IoU={r.iou_pw5:.3f}"
    )

# ---------------------------------------------------------
# 7. Overall metric comparison
# ---------------------------------------------------------

def global_metrics(model_name):
    d = df[
        (df["model"] == model_name) &
        (df["threshold"] == 0.8)
    ]

    # aggregate from confusion counts
    tp = d["tp"].sum()
    fp = d["fp"].sum()
    fn = d["fn"].sum()
    tn = d["tn"].sum()

    iou = tp / (tp + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return iou, dice, precision, recall, tp, fp, fn, tn


baseline_metrics = global_metrics("baseline")

pw5_name = "pw5"
pw5_metrics = global_metrics(pw5_name)

summary = pd.DataFrame([
    {
        "model": "baseline",
        "threshold": 0.80,
        "IoU": baseline_metrics[0],
        "Dice": baseline_metrics[1],
        "Precision": baseline_metrics[2],
        "Recall": baseline_metrics[3],
        "TP": baseline_metrics[4],
        "FP": baseline_metrics[5],
        "FN": baseline_metrics[6],
        "TN": baseline_metrics[7],
    },
    {
        "model": "PW5",
        "threshold": 0.80,
        "IoU": pw5_metrics[0],
        "Dice": pw5_metrics[1],
        "Precision": pw5_metrics[2],
        "Recall": pw5_metrics[3],
        "TP": pw5_metrics[4],
        "FP": pw5_metrics[5],
        "FN": pw5_metrics[6],
        "TN": pw5_metrics[7],
    }
])

summary.to_csv(
    OUTPUT_DIR / "final_metric_comparison.csv",
    index=False
)

print("\n" + "-" * 70)
print("GLOBAL METRICS @ THRESHOLD 0.80")
print("-" * 70)

print(
    f"Baseline : IoU={baseline_metrics[0]*100:.2f}% | "
    f"Dice={baseline_metrics[1]*100:.2f}% | "
    f"P={baseline_metrics[2]*100:.2f}% | "
    f"R={baseline_metrics[3]*100:.2f}%"
)

print(
    f"PW5      : IoU={pw5_metrics[0]*100:.2f}% | "
    f"Dice={pw5_metrics[1]*100:.2f}% | "
    f"P={pw5_metrics[2]*100:.2f}% | "
    f"R={pw5_metrics[3]*100:.2f}%"
)

# ---------------------------------------------------------
# 8. No-flood pixel-level false positive rate
# ---------------------------------------------------------

no_flood = merged[merged["scene_type"] == "no_flood"].copy()

def no_flood_fpr(suffix):
    fp = no_flood[f"fp_{suffix}"].sum()
    tn = no_flood[f"tn_{suffix}"].sum()

    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


baseline_nf_fpr = no_flood_fpr("baseline")
pw5_nf_fpr = no_flood_fpr("pw5")

print("\n" + "-" * 70)
print("NO-FLOOD PIXEL-LEVEL FALSE POSITIVE RATE")
print("-" * 70)

print(f"Baseline @0.80: {baseline_nf_fpr * 100:.4f}%")
print(f"PW5 @0.80     : {pw5_nf_fpr * 100:.4f}%")

# ---------------------------------------------------------
# 9. Save complete scene comparison
# ---------------------------------------------------------

flood.to_csv(
    OUTPUT_DIR / "all_flood_scene_differences.csv",
    index=False
)

# ---------------------------------------------------------
# 10. Final concise summary
# ---------------------------------------------------------

with open(OUTPUT_DIR / "FINAL_SUMMARY.txt", "w", encoding="utf-8") as f:

    f.write("ResQTwin Satellite Segmentation - Final Analysis\n")
    f.write("=" * 55 + "\n\n")

    f.write("Candidate configuration: Positive Weight = 5, threshold = 0.80\n\n")

    f.write(f"Flood scenes analysed: {len(flood)}\n")
    f.write(
        f"Scenes improved: {improved}/{len(flood)} "
        f"({improved / len(flood) * 100:.2f}%)\n"
    )
    f.write(
        f"Scenes worsened: {worsened}/{len(flood)} "
        f"({worsened / len(flood) * 100:.2f}%)\n"
    )
    f.write(f"Median IoU change: {median_iou_change * 100:+.2f} pp\n")
    f.write(f"Mean IoU change: {mean_iou_change * 100:+.2f} pp\n\n")

    f.write("Global metrics:\n")
    f.write(
        f"Baseline IoU: {baseline_metrics[0]*100:.2f}%\n"
        f"PW5 IoU: {pw5_metrics[0]*100:.2f}%\n"
        f"Baseline Dice: {baseline_metrics[1]*100:.2f}%\n"
        f"PW5 Dice: {pw5_metrics[1]*100:.2f}%\n"
        f"Baseline Precision: {baseline_metrics[2]*100:.2f}%\n"
        f"PW5 Precision: {pw5_metrics[2]*100:.2f}%\n"
        f"Baseline Recall: {baseline_metrics[3]*100:.2f}%\n"
        f"PW5 Recall: {pw5_metrics[3]*100:.2f}%\n\n"
    )

    f.write("No-flood pixel-level FPR:\n")
    f.write(f"Baseline: {baseline_nf_fpr*100:.4f}%\n")
    f.write(f"PW5: {pw5_nf_fpr*100:.4f}%\n")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to:\n{OUTPUT_DIR}")