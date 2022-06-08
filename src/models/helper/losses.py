import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IoU(nn.Module):
    def __init__(self, thresh: float = 0.5):
        """Intersection over Union"""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        smooth: float = 0,
    ):
        # Binarize prediction
        inputs = torch.where(inputs < self.thresh, 0, 1)
        batch_size = targets.shape[0]

        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        union = torch.logical_or(inputs, targets)
        union = union.view(batch_size, -1).sum(-1)
        IoU = (intersection + smooth) / (union + smooth)

        if weights is not None:
            assert (
                weights.shape == IoU.shape
            ), f'"weights" must be in shape of "{IoU.shape}"'
            return (IoU * weights).sum()

        return IoU.mean()


class Dice(nn.Module):
    def __init__(self, thresh: float = 0.5):
        """Dice coefficient."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        smooth: float = 0,
    ):
        # Binarize prediction
        inputs = torch.where(inputs < self.thresh, 0, 1)
        batch_size = targets.shape[0]

        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        targets_area = targets.view(batch_size, -1).sum(-1)
        inputs_area = inputs.view(batch_size, -1).sum(-1)
        dice = (2.0 * intersection + smooth) / (inputs_area + targets_area + smooth)

        if weights is not None:
            assert (
                weights.shape == dice.shape
            ), f'"weights" must be in shape of "{dice.shape}"'
            return (dice * weights).sum()

        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, thresh: float = 0.5):
        """Dice loss + binary cross-entropy loss."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh
        self.dice = Dice(self.thresh)
        self.__name__ = "DiceBCELoss"

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        smooth: float = 0,
    ):
        batch_size = inputs.shape[0]

        bce = F.binary_cross_entropy(inputs, targets, reduce=False)
        bce = bce.reshape(batch_size, -1).mean(-1)

        if weights is not None:
            assert (
                weights.shape == bce.shape
            ), f'"weights" must be in shape of "{bce.shape}"'
            bce = (bce * weights).sum()
        else:
            bce = bce.mean()

        dice_loss = 1 - self.dice(inputs, targets, weights, smooth)
        dice_bce = bce + dice_loss
        return dice_bce


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, alpha=0.8, gamma=2, smooth=1
    ):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE = nn.BCELoss()(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=0.5, eps=1e-9):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = -(
            ALPHA
            * (
                (targets * torch.log(inputs))
                + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))
            )
        )
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo
