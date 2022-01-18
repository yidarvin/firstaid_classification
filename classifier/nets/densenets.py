
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from classifier.config import configurable
from .utils import *
from .build import ARCHITECTURE_REGISTRY


@ARCHITECTURE_REGISTRY.register()
class densenet121mtl(nn.Module):
    @configurable
    def __init__(self, in_chan=3, out_chan=[2], pretrained=True, downsample=0):
        super(densenet121mtl, self).__init__()
        if type(out_chan) is not list:
            out_chan = [out_chan]

        self.model = torchvision.models.densenet121(pretrained=pretrained)
        if in_chan != 3:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        heads = []
        for out_c in out_chan:
            heads.append(nn.Linear(1024, out_c))
        self.model.outs = nn.ModuleList(heads)

        if downsample >= 1:
            self.model.features.conv0 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3, bias=False)
            if downsample >= 2:
                self.model.features.pool0 = Identity()
                if downsample >= 3:
                    self.model.features.transition1.pool = Identity()
                    if downsample >= 4:
                        self.model.features.transition2.pool = Identity()
                        if downsample >= 5:
                            self.model.features.transition3.pool = Identity()

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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        outs = []
        for fc in self.model.outs:
            outs.append(fc(x))

        return outs

@ARCHITECTURE_REGISTRY.register()
class densenet169mtl(nn.Module):
    @configurable
    def __init__(self, in_chan=3, out_chan=[2], pretrained=True, downsample=0):
        super(densenet169mtl, self).__init__()
        if type(out_chan) is not list:
            out_chan = [out_chan]

        self.model = torchvision.models.densenet169(pretrained=pretrained)
        if in_chan != 3:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        heads = []
        for out_c in out_chan:
            heads.append(nn.Linear(1664, out_c))
        self.model.outs = nn.ModuleList(heads)

        if downsample >= 1:
            self.model.features.conv0 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3, bias=False)
            if downsample >= 2:
                self.model.features.pool0 = Identity()
                if downsample >= 3:
                    self.model.features.transition1.pool = Identity()
                    if downsample >= 4:
                        self.model.features.transition2.pool = Identity()
                        if downsample >= 5:
                            self.model.features.transition3.pool = Identity()

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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        outs = []
        for fc in self.model.outs:
            outs.append(fc(x))

        return outs

@ARCHITECTURE_REGISTRY.register()
class densenet201mtl(nn.Module):
    @configurable
    def __init__(self, in_chan=3, out_chan=[2], pretrained=True, downsample=0):
        super(densenet201mtl, self).__init__()
        if type(out_chan) is not list:
            out_chan = [out_chan]

        self.model = torchvision.models.densenet201(pretrained=pretrained)
        if in_chan != 3:
            self.model.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        heads = []
        for out_c in out_chan:
            heads.append(nn.Linear(1920, out_c))
        self.model.outs = nn.ModuleList(heads)

        if downsample >= 1:
            self.model.features.conv0 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3, bias=False)
            if downsample >= 2:
                self.model.features.pool0 = Identity()
                if downsample >= 3:
                    self.model.features.transition1.pool = Identity()
                    if downsample >= 4:
                        self.model.features.transition2.pool = Identity()
                        if downsample >= 5:
                            self.model.features.transition3.pool = Identity()

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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        outs = []
        for fc in self.model.outs:
            outs.append(fc(x))

        return outs
