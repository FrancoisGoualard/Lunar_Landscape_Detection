import sys
import os
from content.Displayer import Displayer
from content.imageAnalysis import main_process, load_model
from content.prediction import prediction_test
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
    elif len(sys.argv) == 3:
        if sys.argv[1] == "test":
            try:
                transferLearningModel = load_model(f'{OUTPUT}model_TL_UNET.h5')
            except ImportError:
                print(f"Error : please generate the weights first. {OUTPUT}model_TL_UNET.h5 not found")
            else:
                prediction_test(transferLearningModel, sys.argv[2])


if __name__ == "__main__":
    main()
