import matplotlib.pyplot as plt
import cv2 as cv
import os

from config import DATAPATH

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

    img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
    img_x = cv.resize(img_x, (500, 500))
    img_x = img_x.reshape(1, 500, 500, 3)
    prediction = transferLearningModel.predict(img_x)
    pred = prediction.reshape(500, 500, 3)
    pred_ = cv.resize(pred, (700, 450))

    plt.subplot(5, 2, 2)
    plt.title("Predicted segmentation", fontsize=20)
    plt.imshow(pred_)
    plt.show()


def prediction_test(transferLearningModel, im_number):
    if int(im_number) < 0:
        print("Image number not correct")
        return

    plt.figure(figsize=(30, 30))

    sourceImg = sorted(os.listdir(DATAPATH + 'images/render'))
    targetImg = sorted(os.listdir(DATAPATH + 'images/ground'))

    img_x = cv.imread(DATAPATH + "images/render/" + sourceImg[int(im_number) - 1])
    plt.subplot(5, 3, 1)
    plt.title(f"Actual image, number {im_number}", fontsize=20)
    plt.imshow(img_x)

    img_x = cv.imread(DATAPATH + "images/ground/" + targetImg[int(im_number) - 1])
    plt.subplot(5, 3, 1 + 1)
    plt.title("Expected segmentation", fontsize=20)
    plt.imshow(img_x)

    img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
    img_x = cv.resize(img_x, (500, 500))
    img_x = img_x.reshape(1, 500, 500, 3)
    prediction = transferLearningModel.predict(img_x)
    pred = prediction.reshape(500, 500, 3)
    pred_ = cv.resize(pred, (700, 450))

    plt.subplot(5, 3, 1 + 2)
    plt.title("Predicted segmentation", fontsize=20)
    plt.imshow(pred_)
    plt.show()

    return