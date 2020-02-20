import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from content.Displayer import Displayer
from keras.utils.vis_utils import plot_model
from IPython.display import Image
import matplotlib.pyplot as plt

from keras.applications import vgg16
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from content.modelEnhancer import ModelEnhancer
from config import DATAPATH, KERASPATH, HERE


np.random.seed(123)
tf.set_random_seed(123)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def parsing():
    InputPath = DATAPATH
    dataset_path = os.path.join(DATAPATH, "images", "dataset")

    SourceImg = sorted(os.listdir(dataset_path + 'render'))
    TargetImg = sorted(os.listdir(dataset_path + 'ground'))
    X_ = []
    y_ = []
    count = 0
    for i in range(len(SourceImg)):
        if count < 2165:
            count = count + 1
            # print(InputPath + 'images/render/' + SourceImg[i])
            img_1 = cv.imread(InputPath + 'images/render/' + SourceImg[i])
            disp = Displayer([], "")
            # disp.display_image_and_boxes(InputPath + 'images/render/' + SourceImg[i])
            img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
            img_1 = cv.resize(img_1, (500, 500))
            X_.append(img_1)
            # disp.display_image(img_1, f"image_1 {count}")
            img_2 = cv.imread(InputPath + 'images/ground/' + TargetImg[i])
            img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
            img_2 = cv.resize(img_2, (500, 500))
            # disp.display_image(img_2, f"image_2 {count}")
            # disp.clean()
            y_.append(img_2)
    X_ = np.array(X_)
    y_ = np.array(y_)
    return X_, y_


def plot_layers(Model_):
    Model_.summary()
    path_file = HERE + "model_.png"  # TODO : a changer avec os
    plot_model(Model_, to_file=path_file, show_shapes=True, show_layer_names=True)
    Image(retina=True, filename=HERE + "model_.png")


def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,500,500,3)
        y_input = y[i].reshape(1,500,500,3)
        yield (X_input,y_input)


def prediction_test(TransferLearningModel):
    img_x = cv.imread(DATAPATH + "images/render/render0001.png")
    img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
    img_x = cv.resize(img_x, (500, 500))
    img_x = img_x.reshape(1, 500, 500, 3)
    prediction = TransferLearningModel.predict(img_x)
    pred = prediction.reshape(500, 500, 3)
    pred_ = cv.resize(pred, (700, 450))
    plt.imshow(pred_)
    plt.show()


def main_process():
    X_, Y_ = parsing()
    input_shape = (500, 500, 3)
    tf.test.is_gpu_available(cuda_only=True,
    min_cuda_compute_capability=None)
    VGG16_weight = f"{KERASPATH}vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG16 = vgg16.VGG16(include_top=False, weights=VGG16_weight, input_shape=input_shape)
    Model_ = ModelEnhancer(VGG16)
    plot_layers(Model_)
    Model_.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(f'{HERE}model_TL_UNET.h5', verbose=1, mode='auto', monitor='loss', save_best_only=True)
    Model_.fit_generator(GenerateInputs(X_, Y_), epochs=433, verbose=1, callbacks=[checkpointer],
                         steps_per_epoch=5, shuffle=True)
    TransferLearningModel = load_model(f'{HERE}model_TL_UNET.h5')
    prediction_test(TransferLearningModel)
