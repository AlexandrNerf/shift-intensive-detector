from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch import nn
from torchvision.ops import MultiScaleRoIAlign
import hydra
from omegaconf import DictConfig

class FasterRCNNDecoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        anchor_generator = AnchorGenerator(
            sizes=(
                (32,),
                (64,),
                (128,),
                (256,),
            ),
            aspect_ratios=(
                (0.5, 1.0, 2.0),
            ) * 4,
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
