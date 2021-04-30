import torch
import torch.nn as nn
from model.encoder import _make_resnet_encoder
from model.depthDecoder import MidasDecoder
from model.bboxDecoder import Darknet
from model.planeDecoder import MaskRCNN


class TriclopsNet(nn.Module):
    '''
      Network for detecting objects, generate depth map and identify plane surfaces
    '''

    def __init__(self):
        super(TriclopsNet, self).__init__()
        """
          Get required configuration for all the 3 models
  
        """

        # use_pretrained = False if path is None else True

        self.encoder = _make_resnet_encoder()

        self.depth_decoder = MidasDecoder()

        self.bbox_decoder = Darknet(cfg='path to config') # Fill

        self.plane_decoder = MaskRCNN(config='configuration') # Fill

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x, depth=False, bbox=False, planer=False):

        input_x = x

        # Encoder blocks
        layer_1 = self.encoder.layer1(x)
        layer_2 = self.encoder.layer2(layer_1)
        layer_3 = self.encoder.layer3(layer_2)
        layer_4 = self.encoder.layer4(layer_3)

        Yolo_0 = self.conv1(layer_2)
        Yolo_1 = self.conv2(layer_3)
        Yolo_2 = self.conv3(layer_4)

        if depth:
            depth_out = self.depth_decoder([layer_1, layer_2, layer_3, layer_4])
        else:
            depth_out = None

        if bbox:
            bbox_out = self.bbox_decoder(Yolo_2, Yolo_1, Yolo_0)
        else:
            bbox_out = None

        if planer:
            input = [input_x, [1, 1, 1, 1]]
            plane_out = self.plane_decoder(input, [layer_1, layer_2, layer_3, layer_4])
        else:
            plane_out = None

        return plane_out, bbox_out, depth_out
