import timm
from torch import nn
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class TimmBackbone(nn.Module):
    def __init__(self,
                 model: str = 'mobilenetv3_large_100',
                 pretrained: bool = True,
                 out_indices=(1, 2, 3, 4),
                 fpn_out_channels: int = 256,):
        super().__init__()


        self.backbone = timm.create_model(
            model,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        in_channels_list = self.backbone.feature_info.channels()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
        )

        self.out_channels = fpn_out_channels

    def forward(self, x):
        features = self.backbone(x)

        features = OrderedDict(
            (str(i), feat)
            for i, feat in enumerate(features)
        )

        features = self.fpn(features)

        return features
