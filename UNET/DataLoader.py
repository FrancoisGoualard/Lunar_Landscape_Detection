from config import DATAPATH
from tqdm import tqdm
import cv2 as cv
import os
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def create_dataset():
    render_path = str(DATAPATH)+'images/render/'
    mask_path = str(DATAPATH)+'images/ground/'
    images_ids = (sorted(os.listdir(render_path)),
                  sorted(os.listdir(mask_path)))

    X = np.zeros((len(images_ids[0]), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                 dtype=np.uint8)
    Y = np.zeros((len(images_ids[1]), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                 dtype=np.uint8)

    for n in tqdm(range(len(images_ids[0])), total=len(images_ids[0])):
        image_path = render_path + images_ids[0][n]
        img = cv.imread(image_path)[:, :, :IMG_CHANNELS]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256, 256))
        X[n] = img
        image_mask_path = mask_path + images_ids[1][n]
        mask_ = cv.imread(image_mask_path)[:, :, :IMG_CHANNELS]
        mask_ = cv.cvtColor(mask_, cv.COLOR_BGR2RGB)
        mask_ = cv.resize(mask_, (256, 256))
        Y[n] = mask_
    return X, Y
