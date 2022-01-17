
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from classifier.config import configurable
from .utils import *
from .build import ARCHITECTURE_REGISTRY

@ARCHITECTURE_REGISTRY.register()
class resnet18mtl(nn.Module):
    @configurable
    def __init__(self, in_chan=3, out_chan=[2], pretrained=True, downsample=0):
        super(resnet18mtl, self).__init__()
        if type(out_chan) is not list:
            out_chan = [out_chan]

        self.model = torchvision.models.resnet18(pretrained=pretrained)
        if in_chan != 3:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        heads = []
        for out_c in out_chan:
            heads.append(nn.Linear(512, out_c))
        self.model.outs = nn.ModuleList(heads)

        if downsample >= 1:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3, bias=False)
            if downsample >= 2:
                self.model.maxpool = Identity()
                if downsample >= 3:
                    self.model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
                    self.model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
                    if downsample >= 4:
                        self.model.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
                        self.model.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
                        if downsample >= 5:
                            self.model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
                            self.model.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_chan": cfg.MODEL.CHANIN,
            "out_chan": cfg.MODEL.CHANOUT,
            "pretrained": cfg.MODEL.PRETRAINED,
            "downsample": cfg.MODEL.DOWNSAMPLE,
        }

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        outs = []
        for fc in self.model.outs:
            outs.append(fc(x))

        return outs
