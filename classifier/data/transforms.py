
import numpy as np
import torch

class RandomFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        if np.random.rand() > self.flip_prob:
            X[:,:,:] = X[:,:,::-1]
        return {'X':X, 'Y':Y}

class RandomRotate(object):
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        rotnum = np.random.choice(4)
        for ii in range(X.shape[0]):
            X[ii,:,:] = np.rot90(X[ii,:,:],k=rotnum,axes=(0,1))
        return {'X':X, 'Y':Y}

class RandomShift(object):
    def __init__(self, max_shift=32):
        self.max_shift = int(max_shift)
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        h,w = X.shape[1:]
        X_shift = np.zeros((X.shape[0], X.shape[1]+2*self.max_shift, X.shape[2]+2*self.max_shift))
        for ii in range(X.shape[0]):
            X_shift[ii,:,:] = np.pad(X[ii,:,:], self.max_shift, mode='constant',constant_values=-1)
        top     = np.random.randint(0, 2*self.max_shift)
        left    = np.random.randint(0, 2*self.max_shift)
        X[:,:,:] = X_shift[:,top:(top+h), left:(left+w)]
        return {'X':X, 'Y':Y}

class AddNoise(object):
    def __init__(self, scale=0.01):
        self.scale = scale
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        X += np.random.randn(X.shape[0],X.shape[1],X.shape[2]) * self.scale
        return {'X':X, 'Y':Y}

class ToTensor(object):
    def __call__(self, sample):
        X,Y = sample['X'], sample['Y']
        sample['X'] = torch.from_numpy(X).float()
        return sample
