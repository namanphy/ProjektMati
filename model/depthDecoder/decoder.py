import torch
import torch.nn as nn

from model.base_model import BaseModel
from model.depthDecoder.blocks import FeatureFusionBlock, Interpolate, _make_decoder_scratch


class MidasDecoder(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
        """
        # print("Loading weights: ", path)

        super(MidasDecoder, self).__init__()

        self.scratch = _make_decoder_scratch(features=features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        # if path:
        #     self.load(path)

    def forward(self, x_array):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1, layer_2, layer_3, layer_4 = [activation for activation in x_array]

        layer_1_rn = self.scratch.layer1_rn(layer_1)  # 256 -> 256
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # 512 -> 256
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # 1024 -> 256
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # 2048 -> 256

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)
