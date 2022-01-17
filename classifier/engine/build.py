
import torch.optim as optim
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")

def build_loss(cfg):
    loss = cfg.HP.LOSS
    lossfx = LOSS_REGISTRY.get(loss)(cfg)
    return lossfx

def build_opt(model, cfg):
    opt = cfg.HP.OPTIMIZER
    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.HP.LR, weight_decay=cfg.HP.L2)
    return optimizer
