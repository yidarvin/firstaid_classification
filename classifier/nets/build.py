
import torch

from fvcore.common.registry import Registry

ARCHITECTURE_REGISTRY = Registry("ARCHITECTURE")

def build_model(cfg):
    arch = cfg.MODEL.ARCHITECTURE
    model = ARCHITECTURE_REGISTRY.get(arch)(cfg)
    if cfg.SAVE.MODELPATH and cfg.MODEL.LOADPREV:
        model.load_state_dict(torch.load(cfg.SAVE.MODELPATH, cfg.NAME + '_best.pth'))
    return model
