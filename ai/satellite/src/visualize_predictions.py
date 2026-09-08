from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai.satellite.models.unet import UNet
from ai.satellite.src.dataset import Sen1Floods11Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data" / "sen1floods11"

CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "satellite" / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "ai" / "satellite" / "outputs" / "prediction_visualizations"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.5

# Test samples we want to inspect initially.
SAMPLE_INDICES = [0, 10, 20, 30, 40]


def find_checkpoint():
    checkpoints = list(CHECKPOINT_DIR.rglob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint found under {CHECKPOINT_DIR}"
        )

    # Prefer the baseline checkpoint.
    baseline = [
        p for p in checkpoints
        if "baseline" in p.name.lower()
    ]

    if baseline:
        return baseline[0]

    return checkpoints[0]


def load_model(checkpoint_path):
    model = UNet(
        in_channels=2,
        out_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", None)
    else:
        model.load_state_dict(checkpoint)
        epoch = None

    model.eval()

    return model, epoch


def prepare_dataset():
    return Sen1Floods11Dataset(split="test")


def create_figure(image, target, probability, prediction, sample_index, image_name):
    """
    Create a 4-panel diagnostic figure:

    1. Sentinel-1 VV
    2. Sentinel-1 VH
    3. Ground truth
    4. Prediction
    """

    vv = image[0]
    vh = image[1]

    valid = target != -1

    gt = np.ma.masked_where(~valid, target)

    pred = prediction.astype(np.float32)
    pred = np.ma.masked_where(~valid, pred)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    # VV
    axes[0].imshow(vv, cmap="gray")
    axes[0].set_title("Sentinel-1 VV")
    axes[0].axis("off")

    # VH
    axes[1].imshow(vh, cmap="gray")
    axes[1].set_title("Sentinel-1 VH")
    axes[1].axis("off")

    # Ground truth
    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction (threshold={THRESHOLD})")
    axes[3].axis("off")

    fig.suptitle(
        f"Test Sample {sample_index} — {image_name}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"test_sample_{sample_index:03d}.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("ResQTwin — Satellite Prediction Visualization")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = find_checkpoint()

    print(f"Checkpoint: {checkpoint_path}")

    model, epoch = load_model(checkpoint_path)

    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")

    dataset = prepare_dataset()

    print(f"Test samples available: {len(dataset)}")
    print(f"Visualizing: {SAMPLE_INDICES}")

    with torch.no_grad():

        for sample_index in SAMPLE_INDICES:

            if sample_index >= len(dataset):
                print(
                    f"Skipping {sample_index}: "
                    f"dataset contains only {len(dataset)} samples."
                )
                continue

            image, target = dataset[sample_index]


            image_name, _ = dataset.samples[sample_index]

            image_tensor = image.unsqueeze(0).to(DEVICE)

            logits = model(image_tensor)

            probability = torch.sigmoid(logits)[0, 0]

            prediction = (
                probability >= THRESHOLD
            ).cpu().numpy()

            image_np = image.cpu().numpy()
            target_np = target.cpu().numpy()
            probability_np = probability.cpu().numpy()

            

            output_path = create_figure(
                image=image_np,
                target=target_np,
                probability=probability_np,
                prediction=prediction,
                sample_index=sample_index,
                image_name=image_name
            )

            valid = target_np != -1

            predicted_flood = (
                prediction & valid
            ).sum()

            valid_pixels = valid.sum()

            flood_percentage = (
                100.0 * predicted_flood / valid_pixels
                if valid_pixels > 0
                else 0.0
            )

            print(
                f"[{sample_index}] "
                f"{image_name} | "
                f"Predicted flood: {flood_percentage:.2f}% | "
                f"Saved: {output_path}"
            )

    print()
    print("Visualization complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()