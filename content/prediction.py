import matplotlib.pyplot as plt
import cv2 as cv
import os
import imgaug

from config import DATAPATH, OUTPUT
from tensorflow.keras.models import load_model

def prediction(transferLearningModel, file_path):
    """

    :param transferLearningModel: the trained network
    :param file_path: path of the image to give to the network
    :return: Nothing, prints the predicted image
    """
    plt.figure(figsize=(30, 30))
    img_x = cv.imread(file_path)
    if img_x.size < 1:
        return
    plt.subplot(5, 2, 1)
    plt.imshow(img_x)
    cv.imwrite(OUTPUT + "image_original", img_x)


    img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
    img_x = cv.resize(img_x, (500, 500))
    img_x = img_x.reshape(1, 500, 500, 3)
    prediction = transferLearningModel.predict(img_x)
    pred = prediction.reshape(500, 500, 3)
    pred_ = cv.resize(pred, (700, 450))
    cv.imwrite(OUTPUT + "prediction", pred_)

    plt.subplot(5, 2, 2)
    plt.title("Predicted segmentation", fontsize=20)
    plt.imshow(pred_)
    plt.show()


def display(img, title, number):
    plt.subplot(5, 3, number)
    plt.title(title, fontsize=20)
    try:
        img = img.reshape(500, 500, 3)
        img = cv.resize(img, (700, 450))
    except ValueError:
        pass
    plt.imshow(img)


def prediction_test(im_number):
    try:
         transferLearningModel1 = load_model(f'{OUTPUT}model_TL_UNET.h5')
         transferLearningModel2 = load_model(f'{OUTPUT}model_TL_aug.h5')
         transferLearningModel3 = load_model(f'{OUTPUT}model_TL_aug_150.h5')

    except ImportError:
        print(f"Error : please generate the weights first. {OUTPUT} + {sys.argv[i]} not found")
        return

    if int(im_number) < 0:
        print("Image number not correct")
        return

    plt.figure(figsize=(30, 30))

    sourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    targetImg = sorted(os.listdir(DATAPATH + 'images/ground'))

    img_x = cv.imread(DATAPATH + "images/render/" + sourceImg[int(im_number) - 1])
    display(img_x, f"Actual image, number {im_number}", 1)

    img_x = cv.imread(DATAPATH + "images/ground/" + targetImg[int(im_number) - 1])
    display(img_x, "Expected segmentation", 2)

    img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
    img_x = cv.resize(img_x, (500, 500))
    img_x = img_x.reshape(1, 500, 500, 3)

    prediction = transferLearningModel1.predict(img_x)
    display(prediction, "Predicted segmentation", 3)

    prediction = transferLearningModel2.predict(img_x)
    display(prediction, "Predicted segmentation with Data augmentation Epoch=2", 4)

    prediction = transferLearningModel3.predict(img_x)
    display(prediction, "Predicted segmentation with Data augmentation Epoch=150", 5)

    plt.show()

    return