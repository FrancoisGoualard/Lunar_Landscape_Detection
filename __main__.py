import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from content.Displayer import Displayer
from content.imageAnalysis import main_process, prediction_test, load_model
from config import HERE
import tensorflow as tf

if __name__ == "__main__":
    print('tensorflow Version : ', tf.__version__)
    if len(sys.argv) == 2 :
        if sys.argv[1] == "display":
            Displayer().run_without_image()
    elif len(sys.argv) == 3:
        if sys.argv[1] == "test":
            TransferLearningModel = load_model(f'{HERE}model_TL_UNET.h5')
            prediction_test(TransferLearningModel, sys.argv[2])
    else:
        main_process()