
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
