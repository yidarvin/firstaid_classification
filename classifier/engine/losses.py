
import torch
import torch.nn as nn

from classifier.config import configurable
from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    @configurable
    def __init__(self, out_chan=[2]):
        super(CrossEntropyLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.out_chan = out_chan
    @classmethod
    def from_config(cls, cfg):
        return {
            "out_chan": cfg.MODEL.CHANOUT,
        }
    def forward(self, output, Ys):
        total_loss = 0.0
        for out,Y in zip(output,Ys):
            total_loss += self.CrossEntropyLoss(out, Y)
        return total_loss / len(output)

@LOSS_REGISTRY.register()
class HingeLoss(nn.Module):
    @configurable
    def __init__(self, out_chan=[2]):
        super(HingeLoss, self).__init__()
        self.HingeLoss = nn.MultiMarginLoss()
        self.out_chan = out_chan
    @classmethod
    def from_config(cls, cfg):
        return {
            "out_chan": cfg.MODEL.CHANOUT,
        }
    def forward(self, output, Ys):
        total_loss = 0.0
        for out,Y in zip(output,Ys):
            total_loss += self.HingeLoss(out, Y)
        return total_loss / len(output)
