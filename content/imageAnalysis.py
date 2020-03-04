import os
import cv2 as cv
import shutil
import tqdm
import imgaug

from pathlib import Path
from tensorflow.keras.utils import plot_model
from IPython.display import Image

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

from content.modelEnhancer import ModelEnhancer
from content.create_dataset import create_dataset

from config import DATAPATH, KERASPATH, OUTPUT, GPU, NB_EPOCH, SOURCEIMG, TARGETIMG


def treat_img(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (500, 500))
    return img


def scan_existing_folders(folder_path):
    try:
        nb_img = len(os.listdir(folder_path))
    except FileNotFoundError:
        return 0
    return nb_img


def parsing():
    """
    Loads the content of artificial-lunar-rocky-landscape-dataset, the
    location of this directory is set in config.py
    It modifies the images and writes them in artificial-lunar-rocky-landscape-dataset/images_cleaned

     """

    #  Here we will not run this function if it has already run
    nb_img = len(os.listdir(TARGETIMG))
    nb_render_img = scan_existing_folders(DATAPATH + "images_cleaned/render/")
    nb_ground_img = scan_existing_folders(DATAPATH + "images_cleaned/ground/")

    if nb_ground_img == nb_render_img & nb_img <= nb_ground_img:  # == if data augmentation, else <
        print(nb_img)
        if GPU == 1:  # if we are remote we automatically don't recreate the dataset
            return
        answer = input(
            'Do you want to recreate the reshaped images directory? Answer with YES or NO and press enter \n')
        if answer != "YES":
            return

    # recreate empty folders to write in
    shutil.rmtree(DATAPATH + "images_cleaned/", ignore_errors=True)

    Path(DATAPATH + "images_cleaned/").mkdir(parents=True, exist_ok=True)
    Path(DATAPATH + "images_cleaned/render/").mkdir(parents=True, exist_ok=True)
    Path(DATAPATH + "images_cleaned/ground/").mkdir(parents=True, exist_ok=True)

    SourceImg = sorted(os.listdir(SOURCEIMG))
    TargetImg = sorted(os.listdir(TARGETIMG))

    for i in tqdm.tqdm(range(len(SourceImg))):
        cv.imwrite(f"{DATAPATH}images_cleaned/render/" + SourceImg[i], treat_img(SOURCEIMG + SourceImg[i]))
        cv.imwrite(f"{DATAPATH}/images_cleaned/ground/" + TargetImg[i], treat_img(TARGETIMG + TargetImg[i]))


def load_images():
    SourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    TargetImg = sorted(os.listdir(DATAPATH + 'images/ground'))
    rotate3 = imgaug.augmenters.Affine(rotate=3)
    rotateinv = imgaug.augmenters.Affine(rotate=-3)
    flip_hr = imgaug.augmenters.Fliplr(p=1.0)

    for i in tqdm.tqdm(range(len(SourceImg))):
        img_1 = treat_img(DATAPATH + 'images/render/' + SourceImg[i])
        img_2 = treat_img(DATAPATH + 'images/ground/' + TargetImg[i])
        yield img_1, img_2
    load_augmented_images(rotate3)


def load_augmented_images(change):
    SourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    TargetImg = sorted(os.listdir(DATAPATH + 'images/ground'))
    for i in tqdm.tqdm(range(len(SourceImg))):
        img_1 = treat_img(DATAPATH + 'images/render/' + SourceImg[i])
        img_1 = change.augment_image(img_1)
        img_1 = img_1.reshape(1, 500, 500, 3)
        img_2 = treat_img(DATAPATH + 'images/ground/' + TargetImg[i])
        img_2 = change.augment_image(img_2)
        img_2 = img_2.reshape(1, 500, 500, 3)
        yield img_1, img_2

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

    # should be called if we want to activate data augmentation
    # try:
    #     create_dataset()
    # except AssertionError as e:
    #     print(repr(e))
    # parsing()
    input_shape = (500, 500, 3)

    VGG16_weight = f"{KERASPATH}vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG16_ = VGG16(include_top=False, weights=VGG16_weight, input_shape=input_shape)
    model_ = ModelEnhancer(VGG16_)
    plot_layers(model_)

    model_.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(f'{OUTPUT}model_TL_aug.h5', verbose=1, mode='auto', monitor='loss',
                                   save_best_only=True)
    rotate3 = imgaug.augmenters.Affine(rotate=3)
    rotateinv = imgaug.augmenters.Affine(rotate=-3)
    flip_hr = imgaug.augmenters.Fliplr(p=1.0)

    model_.fit(load_augmented_images(rotate3), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
               steps_per_epoch=5, shuffle=True)
    # model_.fit(load_augmented_images(rotateinv), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
    #            steps_per_epoch=5, shuffle=True)
    # model_.fit(load_augmented_images(flip_hr), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
    #            steps_per_epoch=5, shuffle=True)
    # model_.fit(load_images(), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
    #            steps_per_epoch=5, shuffle=True)


