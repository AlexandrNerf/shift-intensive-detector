from torch import nn

from omegaconf import DictConfig
import hydra

class DetectionModel(nn.Module):
    def __init__(self, backbone: DictConfig, detector: DictConfig, num_classes: int) -> None:
        super().__init__()
        backbone = hydra.utils.instantiate(backbone)
        self.detector = hydra.utils.instantiate(detector, backbone=backbone, num_classes=num_classes)


    def forward(self, x, targets=None):
        out = self.detector(x, targets)
        return out
