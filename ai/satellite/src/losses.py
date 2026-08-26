"""Loss functions for Sen1Floods11 flood segmentation."""

import torch
import torch.nn as nn


class FloodSegmentationLoss(nn.Module):
    """
    Ignore-aware and class-imbalance-aware loss.

    Labels:
        -1 = ignore
         0 = non-water
         1 = water
    """

    def __init__(
        self,
        positive_weight: float = 9.52,
        dice_weight: float = 0.5,
    ):
        super().__init__()

        self.positive_weight = positive_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # logits:  [B, 1, H, W]
        # targets: [B, H, W]

        logits = logits.squeeze(1)

        valid_mask = targets != -1

        if not valid_mask.any():
            raise ValueError("Batch contains no valid labelled pixels.")

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask].float()

        # Weighted BCE handles the strong non-water/water imbalance.
        pos_weight = torch.tensor(
            self.positive_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

        bce = nn.functional.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=pos_weight,
        )

        # Dice component.
        probabilities = torch.sigmoid(valid_logits)

        intersection = (
            probabilities * valid_targets
        ).sum()

        dice_denominator = (
            probabilities.sum()
            + valid_targets.sum()
        )

        dice = (
            2.0 * intersection + 1.0
        ) / (
            dice_denominator + 1.0
        )

        dice_loss = 1.0 - dice

        return bce + self.dice_weight * dice_loss