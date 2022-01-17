
import csv
import cv2
import imageio as io
import numpy as np

from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, models, transforms,utils

from .transforms import *
from .utils import *

class SimpleClassificationDataset(Dataset):
    def __init__(self, path_csv, resize=None, inference=True, databloat=None, transform=None):
        self.path_csv = path_csv
        self.resize = resize
        self.inference = inference
        self.databloat = databloat
        self.transform = transform

        self.path_imgs = []
        if not self.inference:
            self.labels = []
        with open(self.path_csv) as csvfile:
            pathreader = csv.reader(csvfile, delimiter=",")
            for row in pathreader:
                self.path_imgs.append(row[0])
                if not self.inference:
                    self.labels.append([int(r) for r in row[1:]])
        if databloat:
            self.path_imgs, self.labels = extend_data(self.path_imgs, self.labels, databloat)
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self, idx):
        img = io.imread(self.path_imgs[idx]).transpose([2,0,1])
        if not self.inference:
            lab = self.labels[idx]
        # Image Preprocessing
        img = rescale(img)
        if self.resize:
            img = resize(img, img.resize)
        sample = {'X': img, 'Y': lab}
        if self.transform:
            sample = self.transform(sample)
        return sample

def create_dataloaders(cfg):
    dataloaders = {}
    if cfg.DATA.TRAIN.PATH:
        augmentation = []
        if cfg.DATA.TRAIN.FLIP:
            augmentation.append(RandomFlip())
        if cfg.DATA.TRAIN.ROTATE:
            augmentation.append(RandomRotate())
        if cfg.DATA.TRAIN.SHIFT:
            augmentation.append(RandomShift(max_shift=cfg.DATA.TRAIN.SHIFT))
        if cfg.DATA.TRAIN.ADDNOISE:
            augmentation.append(AddNoise(scale=cfg.DATA.TRAIN.ADDNOISE))
        augmentation.append(ToTensor())
        data_tr = SimpleClassificationDataset(cfg.DATA.TRAIN.PATH,
                                              resize=cfg.INPUT.RESIZE,
                                              inference=False,
                                              databloat=cfg.DATA.TRAIN.DATABLOAT,
                                              transform=transforms.Compose(augmentation))
        loader_tr = DataLoader(data_tr, batch_size=cfg.HP.BATCHSIZE, shuffle=True, drop_last=True, num_workers=cfg.INPUT.NUMWORKERS)
        dataloaders["train"] = loader_tr
    if cfg.DATA.VAL.PATH:
        data_va = SimpleClassificationDataset(cfg.DATA.VAL.PATH,
                                              resize=cfg.INPUT.RESIZE,
                                              inference=False,
                                              transform=transforms.Compose([ToTensor()]))
        loader_va = DataLoader(data_va, batch_size=1, shuffle=False, drop_last=False)
        dataloaders["val"] = loader_va
    if cfg.DATA.TEST.PATH:
        data_te = SimpleClassificationDataset(cfg.DATA.TEST.PATH,
                                              resize=cfg.INPUT.RESIZE,
                                              inference=False,
                                              transform=transforms.Compose([ToTensor()]))
        loader_te = DataLoader(data_te, batch_size=1, shuffle=False, drop_last=False)
        dataloaders["test"] = loader_te
    if cfg.DATA.INFERENCE.PATH:
        data_in = SimpleClassificationDataset(cfg.DATA.INFERENCE.PATH,
                                              resize=cfg.INPUT.RESIZE,
                                              inference=True,
                                              transform=transforms.Compose([ToTensor()]))
        loader_in = DataLoader(data_in, batch_size=1, shuffle=False, drop_last=False)
        dataloaders["inference"] = loader_in
    return dataloaders
