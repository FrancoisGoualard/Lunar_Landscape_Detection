import sys
import os
from content.Displayer import Displayer
from content.imageAnalysis import main_process
from tensorflow.keras.models import load_model

from content.prediction import prediction_test
from content.create_dataset import create_dataset
from config import OUTPUT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    if len(sys.argv) == 1:
        return main_process()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "display":
            Displayer().run_without_image()
        else:
            return 0
    elif len(sys.argv) > 2:
        if sys.argv[1] == "test":
            prediction_test(sys.argv[2])


if __name__ == "__main__":
   main()
