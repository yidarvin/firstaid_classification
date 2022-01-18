
from os.path import join
import datetime

import numpy as np
import torch

from .build import build_loss, build_opt

def take_one_step(model, X, Y, criterion, phase="val", alpha=1):
    with torch.set_grad_enabled(phase == "train"):
        output = model(X)
        loss = criterion(output, Y)
        accs = [(torch.argmax(out,dim=1) == y).float().mean() for out,y in zip(output,Y)]
    return accs,loss

def train_classifier(model, dataloaders, logger, num_gpus, cfg):
    time_of_start = datetime.datetime.now()
    criterion = build_loss(cfg)
    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            model = nn.DataPrallel(model)
    optimizer = build_opt(model, cfg)
    num_epochs = cfg.HP.NUMEPOCH

    logger.super_print('----------TRAINING----------')
    best_acc = 0.0
    for epoch in range(num_epochs):
        logger.super_print('Epoch {}/{}:'.format(epoch, num_epochs - 1))
        for phase in dataloaders.keys():
            if phase == "train":
                model.train()
            elif phase == "test" or phase == "inference":
                continue
            else:
                model.eval()
            running_acc = [0.0 for ii in range(len(cfg.MODEL.CHANOUT))]
            running_loss = 0.0

            for ii,sample_batch in enumerate(dataloaders[phase]):
                X,Y = sample_batch['X'],sample_batch['Y']
                if num_gpus > 0:
                    X,Y = X.cuda(), [y.cuda() for y in Y]
                accs,loss = take_one_step(model, X, Y, criterion, phase=phase)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                running_acc = [r + acc.item() * X.size(0) for r,acc in zip(running_acc,accs)]
                running_loss += loss.item() * X.size(0)
            epoch_acc = [r / len(dataloaders[phase].dataset) for r in running_acc]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            logger.super_print('|-{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, '|'.join([str(e) for e in epoch_acc])))
            if phase == 'val':
                if np.mean(epoch_acc) >= best_acc:
                    best_acc = np.mean(epoch_acc)
                    if cfg.SAVE.MODELPATH:
                        if num_gpus > 1:
                            torch.save(model.module.state_dict(), join(cfg.SAVE.MODELPATH, cfg.NAME + '_best.pth'))
                        else:
                            torch.save(model.state_dict(), join(cfg.SAVE.MODELPATH, cfg.NAME + '_best.pth'))
        logger.super_print('|-Time {}'.format(datetime.datetime.now()-time_of_start))
    if cfg.SAVE.MODELPATH:
        if num_gpus > 1:
            torch.save(model.module.state_dict(), join(cfg.SAVE.MODELPATH, cfg.NAME + '_last.pth'))
        else:
            torch.save(model.state_dict(), join(cfg.SAVE.MODELPATH, cfg.NAME + '_last.pth'))
    logger.super_print('Time {}'.format(datetime.datetime.now()-time_of_start))
    logger.super_print('--------------------')

    return model

def test_classifier(model, dataloader, logger, num_gpus, cfg):
    time_of_start = datetime.datetime.now()
    criterion = build_loss(cfg)
    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            model = nn.DataPrallel(model)
    model.eval()
    logger.super_print('----------TESTING----------')
    conf_matrix = [np.zeros((c,c)) for c in cfg.MODEL.CHANOUT]
    running_loss = 0.0
    for ii,sample_batch in enumerate(dataloader):
        X,Y = sample_batch['X'],sample_batch['Y']
        if num_gpus > 0:
            X,Y = X.cuda(), [y.cuda() for y in Y]
        output = model(X)
        loss = criterion(output, Y)
        pred = [torch.argmax(out,dim=1).item() for out in output]
        for jj,p in enumerate(pred):
            conf_matrix[jj][Y[jj].item(), p] += 1
        running_loss += loss.item() * X.size(0)
    epoch_acc = [(cm * np.eye(cm.shape[0])).sum() / cm.sum() for cm in conf_matrix]
    epoch_loss = running_loss / len(dataloader.dataset)
    logger.super_print('Loss: {:.4f} Acc: {}'.format(epoch_loss, '|'.join([str(e) for e in epoch_acc])))
    for ii,acccm in enumerate(zip(epoch_acc,conf_matrix)):
        acc,cm = acccm
        logger.super_print('|-Class {}: Acc: {}'.format(ii,acc))
        for jj in range(cm.shape[0]):
            if jj == 0:
                logger.super_print('  |-[[{}],'.format(' '.join([str(c) for c in cm[jj]])))
            elif jj == cm.shape[0]-1:
                logger.super_print('  |- [{}]]'.format(' '.join([str(c) for c in cm[jj]])))
            else:
                logger.super_print('  |- [{}],'.format(' '.join([str(c) for c in cm[jj]])))
    logger.super_print('Time {}'.format(datetime.datetime.now()-time_of_start))
    logger.super_print('--------------------')

    return epoch_acc

def run_classifier(model, dataloader, logger, num_gpus, cfg):
    time_of_start = datetime.datetime.now()
    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            model = nn.DataPrallel(model)
    model.eval()
    logger.super_print('----------INFERENCE----------')
    for ii,sample_batch in enumerate(dataloader):
        X,path_file = sample_batch['X'],sample_batch['path']
        if num_gpus > 0:
            X = X.cuda()
        output = model(X)
        pred = [torch.argmax(out,dim=1).item() for out in output]
        logger.super_print('|-{} {}'.format(path_file[0], '|'.join([str(p) for p in pred])))
    logger.super_print('Time {}'.format(datetime.datetime.now()-time_of_start))
    logger.super_print('--------------------')

    return 0
