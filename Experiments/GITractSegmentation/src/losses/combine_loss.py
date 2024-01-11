import torch
import torch.nn as nn

from losses.dice import DiceLoss
from losses.hausdorff import HausdorffLoss

DICE_WEIGHT = 0.4
HAUSDORFF_WEIGHT = 0.6

class DiceHausdorffLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceHausdorffLoss, self).__init__()
        
        self.dice = DiceLoss()
        self.hausdorff = HausdorffLoss()

    def forward(self, inputs, targets, smooth=1):

        hausdorff_loss = self.hausdorff(inputs, targets)        
        dice_loss = self.dice(inputs, targets)
        
        return (DICE_WEIGHT * dice_loss) + (HAUSDORFF_WEIGHT * hausdorff_loss)