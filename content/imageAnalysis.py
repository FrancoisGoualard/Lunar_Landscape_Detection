import os
import cv2 as cv

from pathlib import Path
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

from content.modelEnhancer import ModelEnhancer

from config import DATAPATH, KERASPATH, OUTPUT, GPU, NB_EPOCH


def parsing():
    """
    Loads the content of artificial-lunar-rocky-landscape-dataset, the
    location of this directory is set in config.py
    It modifies the images and writes them in artificial-lunar-rocky-landscape-dataset/images_cleaned

     """

    Path(DATAPATH + "images_cleaned/render/").mkdir(parents=True, exist_ok=True)
    Path(DATAPATH + "images_cleaned/ground/").mkdir(parents=True, exist_ok=True)

    '''Here we will not run this function if it has already run'''
    print(os.getcwd())
    len_render = len(os.listdir(DATAPATH + "images_cleaned/render/"))
    len_ground = len(os.listdir(DATAPATH + "images_cleaned/ground/"))
    if len_render > 200 & (len_render == len_ground) and GPU == 0:
        answer = input('Do you want to recreate the reshaped images directory? Answer with YES or NO and press enter \n')
        if answer != "YES":
            return

    SourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    TargetImg = sorted(os.listdir(DATAPATH + 'images/ground'))

    for i in range(len(SourceImg)):
        # if count < 2165:
        #     count = count + 1
            img_1 = cv.imread(DATAPATH + 'images/render/' + SourceImg[i])
            img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
            img_1 = cv.resize(img_1, (500, 500))
            cv.imwrite(f"{DATAPATH}images_cleaned/render/img_{i}.png", img_1)
            img_2 = cv.imread(DATAPATH + 'images/ground/' + TargetImg[i])
            img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
            img_2 = cv.resize(img_2, (500, 500))
            cv.imwrite(f"{DATAPATH}/images_cleaned/ground/img_{i}.png", img_2)


def load_images():
    SourceImg = sorted(os.listdir(DATAPATH + 'images_cleaned/render'))
    TargetImg = sorted(os.listdir(DATAPATH + 'images_cleaned/ground'))
    for i in range(len(SourceImg)):
        # if count < 2165:
        #     count = count + 1
        img_1 = cv.imread(DATAPATH + 'images_cleaned/render/' + SourceImg[i])
        render = img_1.reshape(1, 500, 500, 3)
        img_2 = cv.imread(DATAPATH + 'images_cleaned/ground/' + TargetImg[i])
        ground = img_2.reshape(1, 500, 500, 3)
        yield(render, ground)


def plot_layers(Model_):
    """
        Writes in a file named model_.png in the directory specified by the
        global variable OUTPUT. It draws a scheme of the network given as argument.
        """
    Model_.summary()
    path_file = OUTPUT + "model_.png"
    plot_model(Model_, to_file=path_file, show_shapes=True, show_layer_names=True)
    Image(retina=True, filename=OUTPUT + "model_.png")



def main_process():
    """Train transfer learning model with given hyperparameters and test against a given image."""

    parsing()
    input_shape = (500, 500, 3)
    # print('GPU available? %s' % (len(tf.config.list_physical_devices(device_type='GPU')) > 0))
    # print('Build with CUDA? %s' % (tf.test.is_built_with_cuda()))

    VGG16_weight = f"{KERASPATH}vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    VGG16_ = VGG16(include_top=False, weights=VGG16_weight, input_shape=input_shape)
    model_ = ModelEnhancer(VGG16_)
    plot_layers(model_)

    model_.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(f'{OUTPUT}model_TL_UNET.h5', verbose=1, mode='auto', monitor='loss', save_best_only=True)

    model_.fit(load_images(), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
                         steps_per_epoch=5, shuffle=True) # TODO Change epoch to hyperparameter constant
    transferLearningModel = load_model(f'{OUTPUT}model_TL_UNET.h5')
