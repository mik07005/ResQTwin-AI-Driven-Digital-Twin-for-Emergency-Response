"""Visualize Sen1Floods11 VV, VH, and flood labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import Sen1Floods11Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "visualizations"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Sen1Floods11Dataset("train")

    # Find the first training sample containing usable labels.
    sample_index = 0

    while sample_index < len(dataset):
        image, mask = dataset[sample_index]

        if (mask != -1).any():
            break

        sample_index += 1

    if sample_index == len(dataset):
        raise RuntimeError("No usable labelled training sample found.")

    image = image.numpy()
    mask = mask.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # VV
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("VV (normalized)")
    axes[0].axis("off")

    # VH
    axes[1].imshow(image[1], cmap="gray")
    axes[1].set_title("VH (normalized)")
    axes[1].axis("off")

    # Flood label:
    # 0  = non-water
    # 1  = water
    # -1 = ignored / invalid
    label_display = np.full(mask.shape, np.nan, dtype=np.float32)

    label_display[mask == 0] = 0
    label_display[mask == 1] = 1
    label_display[mask == -1] = 2

    axes[2].imshow(
        label_display,
        cmap="viridis",
        vmin=0,
        vmax=2,
    )

    axes[2].set_title(
        "Flood Label\n"
        "0=Non-water | 1=Water | -1=Ignore"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Sen1Floods11 Training Sample {sample_index}",
        fontsize=14,
    )

    output_path = OUTPUT_DIR / "sen1floods11_sample.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()