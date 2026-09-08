"""Run flood-segmentation inference using the trained ResQTwin U-Net."""

from pathlib import Path

import numpy as np
import torch

from ai.satellite.models.unet import UNet
from ai.satellite.src.dataset import Sen1Floods11Dataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "satellite" / "checkpoints"

checkpoint_files = list(CHECKPOINT_DIR.rglob("*.pt"))

if not checkpoint_files:
    raise FileNotFoundError(
        f"No checkpoint found under {CHECKPOINT_DIR}"
    )

CHECKPOINT_PATH = checkpoint_files[0]   

# Use a real Sen1Floods11 test sample.
SPLIT = "test"
SAMPLE_INDEX = 0

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    """Load the trained U-Net checkpoint."""
    model = UNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: UNet,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate flood probabilities and binary flood mask."""

    image = image.unsqueeze(0).to(device)

    logits = model(image)

    probabilities = torch.sigmoid(logits[:, 0])

    mask = probabilities >= THRESHOLD

    return probabilities[0].cpu(), mask[0].cpu()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    device = get_device()

    print("=== ResQTwin: Flood segmentation inference ===")
    print(f"Device:        {device}")

    if device.type == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint:    {CHECKPOINT_PATH}")
    print(f"Split:         {SPLIT}")
    print(f"Sample index:  {SAMPLE_INDEX}")
    print(f"Threshold:     {THRESHOLD}")
    print()

    # -----------------------------------------------------------------
    # Validate checkpoint
    # -----------------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    dataset = Sen1Floods11Dataset(SPLIT)

    if SAMPLE_INDEX >= len(dataset):
        raise IndexError(
            f"Sample index {SAMPLE_INDEX} is outside dataset "
            f"of size {len(dataset)}."
        )

    image, target = dataset[SAMPLE_INDEX]

    image_name, label_name = dataset.samples[SAMPLE_INDEX]

    print(f"Image:         {image_name}")
    print(f"Label:         {label_name}")
    print(f"Image shape:   {tuple(image.shape)}")
    print(f"Image dtype:   {image.dtype}")
    print(f"Target shape:  {tuple(target.shape)}")
    print()

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    model, checkpoint = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.6f}"
    )
    print()

    # -----------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------

    probabilities, prediction = predict(
        model,
        image,
        device,
    )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    valid_pixels = target != -1

    valid_prediction = prediction[valid_pixels]
    valid_target = target[valid_pixels]

    flood_pixels = int(valid_prediction.sum())

    valid_pixel_count = int(valid_pixels.sum())

    flood_percentage = (
        100.0 * flood_pixels / valid_pixel_count
        if valid_pixel_count > 0
        else 0.0
    )

    probability_min = float(probabilities[valid_pixels].min())
    probability_max = float(probabilities[valid_pixels].max())
    probability_mean = float(probabilities[valid_pixels].mean())

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    print("Prediction")
    print("-" * 50)
    print(f"Probability min:       {probability_min:.6f}")
    print(f"Probability max:       {probability_max:.6f}")
    print(f"Probability mean:      {probability_mean:.6f}")
    print(f"Flood pixels:           {flood_pixels:,}")
    print(f"Valid pixels:           {valid_pixel_count:,}")
    print(f"Predicted flood area:   {flood_percentage:.2f}%")
    print()

    print("Inference complete.")
    print("No training or data modification was performed.")


if __name__ == "__main__":
    main()