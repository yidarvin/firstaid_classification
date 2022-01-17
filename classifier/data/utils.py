
import numpy as np
import cv2

def rescale(img):
    img = img.astype(np.float32)
    new_img = img - img.min()
    new_img /= (new_img.max() + 1e-6)
    return new_img

def resize(img, resize=None):
    img = img.astype(np.float32)
    img = rescale(img)
    if resize is None or (resize == img.shape[1] and resize == img.shape[2]):
        return img
    new_img = np.zeros((img.shape[0], resize, resize))
    for ii in range(img.shape[0]):
        new_img[ii,:,:] = cv2.resize(img[ii,:,:], (resize,resize))
    img = new_img.astype(np.float32)
    img = rescale(img)
    return img

def extend_data(images, labels, databloat):
    images_tr = []
    labels_tr = []
    for ii in range(databloat):
        images_tr += images
        labels_tr += labels
    return images_tr, labels_tr
