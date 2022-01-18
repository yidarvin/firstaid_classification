
from fvcore.common.config import CfgNode as CN



_C = CN()

_C.NAME = "default"

_C.SAVE = CN()
_C.SAVE.LOGPATH = ""
_C.SAVE.VISPATH = ""
_C.SAVE.MODELPATH = ""

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "resnet18mtl"
_C.MODEL.PRETRAINED = True
_C.MODEL.CHANIN = 3
_C.MODEL.CHANOUT = [2]
_C.MODEL.DOWNSAMPLE = 0
_C.MODEL.LOADPREV = False

_C.DATA = CN()
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.PATH = ""
_C.DATA.TRAIN.DATABLOAT = None
_C.DATA.TRAIN.FLIP = False
_C.DATA.TRAIN.ROTATE = False
_C.DATA.TRAIN.SHIFT = None
_C.DATA.TRAIN.ADDNOISE = None
_C.DATA.VAL = CN()
_C.DATA.VAL.PATH = ""
_C.DATA.TEST = CN()
_C.DATA.TEST.PATH = ""
_C.DATA.INFERENCE = CN()
_C.DATA.INFERENCE.PATH = ""

_C.INPUT = CN()
_C.INPUT.RESIZE = None
_C.INPUT.FILETYPE = "png" #"png", "h5", "nii"
_C.INPUT.NUMWORKERS = 1

_C.HP = CN()
_C.HP.BATCHSIZE = 8
_C.HP.NUMEPOCH = 10
_C.HP.LOSS = "CrossEntropyLoss"
_C.HP.OPTIMIZER = "Adam"
_C.HP.LR = 3e-4
_C.HP.L2 = 1e-6
