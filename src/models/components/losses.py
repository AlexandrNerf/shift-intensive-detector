import torch
from torch import nn
from typing import Dict

class DetectionLoss(nn.Module):
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        self.loss_weights = loss_weights or {
            "loss_classifier": 1.0,
            "loss_box_reg": 1.0,
            "loss_objectness": 1.0,
            "loss_rpn_box_reg": 1.0,
        }

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += loss_value * weight
        return total_loss
