

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from classifier.config import configurable
from .utils import *
from .build import ARCHITECTURE_REGISTRY

@ARCHITECTURE_REGISTRY.register()
class efficientnetb7mtl(nn.Module):
    @configurable
    def __init__(self, in_chan=3, out_chan=[2], pretrained=True, downsample=0):
        super(efficientnetb7mtl, self).__init__()
        if type(out_chan) is not list:
            out_chan = [out_chan]

        self.model = torchvision.models.efficientnet_b7(pretrained=pretrained)
        if in_chan != 3:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        heads = []
        for out_c in out_chan:
            heads.append(nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(2560, out_c), ))
        self.model.outs = nn.ModuleList(heads)

        if downsample >= 1:
            self.model.features[0][0] = nn.Conv2d(in_chan, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if downsample >= 2:
                self.model.features[2][0].block[1][0] = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192, bias=False)
                if downsample >= 3:
                    self.model.features[3][0].block[1][0] = nn.Conv2d(288, 288, kernel_size=5, stride=1, padding=2, groups=288, bias=False)
                    if downsample >= 4:
                        self.model.features[4][0].block[1][0] = nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, groups=480, bias=False)
                        if downsample >= 5:
                            self.model.features[5][0].block[1][0] = nn.Conv2d(960, 960, kernel_size=5, stride=1, padding=2, groups=960, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_chan": cfg.MODEL.CHANIN,
            "out_chan": cfg.MODEL.CHANOUT,
            "pretrained": cfg.MODEL.PRETRAINED,
            "downsample": cfg.MODEL.DOWNSAMPLE,
        }

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        outs = []
        for fc in self.model.outs:
            outs.append(fc(x))

        return outs
