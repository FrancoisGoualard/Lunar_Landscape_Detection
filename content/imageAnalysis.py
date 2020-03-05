import os
import cv2 as cv
import tqdm
import imgaug

from tensorflow.keras.utils import plot_model
from IPython.display import Image

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

from content.modelEnhancer import ModelEnhancer

from config import DATAPATH, KERASPATH, OUTPUT, GPU, NB_EPOCH, SOURCEIMG, TARGETIMG


def treat_img(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (500, 500))
    return img


def load_images():
    SourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    TargetImg = sorted(os.listdir(DATAPATH + 'images/ground'))

    rotate3 = imgaug.augmenters.Affine(rotate=3)
    rotateinv = imgaug.augmenters.Affine(rotate=-3)
    flip_hr = imgaug.augmenters.Fliplr(p=1.0)

    for i in tqdm.tqdm(range(len(SourceImg))):
        img_1 = treat_img(DATAPATH + 'images/render/' + SourceImg[i])
        img_rot1 = rotate3.augment_image(img_1)
        img_rotinv1 = rotateinv.augment_image(img_1)
        img_rotflip1 = flip_hr.augment_image(img_1)

        img_2 = treat_img(DATAPATH + 'images/ground/' + TargetImg[i])
        img_rot2 = rotate3.augment_image(img_2)
        img_rotinv2 = rotateinv.augment_image(img_2)
        img_rotflip2 = flip_hr.augment_image(img_2)

        img_1 = img_1.reshape(1, 500, 500, 3)
        img_rot1 = img_rot1.reshape(1, 500, 500, 3)
        img_rotinv1 = img_rotinv1.reshape(1, 500, 500, 3)
        img_rotflip1 = img_rotflip1.reshape(1, 500, 500, 3)
        img_2 = img_2.reshape(1, 500, 500, 3)
        img_rot2 = img_rot2.reshape(1, 500, 500, 3)
        img_rotinv2 = img_rotinv2.reshape(1, 500, 500, 3)
        img_rotflip2 = img_rotflip2.reshape(1, 500, 500, 3)

        yield img_1, img_2
        yield img_rot1, img_rot2
        yield img_rotinv1, img_rotinv2
        yield img_rotflip1, img_rotflip2


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

    input_shape = (500, 500, 3)

    VGG16_weight = f"{KERASPATH}vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG16_ = VGG16(include_top=False, weights=VGG16_weight, input_shape=input_shape)
    model_ = ModelEnhancer(VGG16_)
    plot_layers(model_)

    model_.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(f'{OUTPUT}model_TL_aug.h5', verbose=1, mode='auto', monitor='loss',
                                   save_best_only=True)

    model_.fit(load_images(), epochs=NB_EPOCH, verbose=1, callbacks=[checkpointer],
               steps_per_epoch=5, shuffle=True)
