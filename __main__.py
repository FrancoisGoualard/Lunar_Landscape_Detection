import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from content.Displayer import Displayer
from content.imageAnalysis import main_process, prediction_test, load_model
from config import OUTPUT


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
                TransferLearningModel = load_model(f'{OUTPUT}model_TL_UNET.h5')
                prediction_test(TransferLearningModel, sys.argv[2])
            except:
                print(f"Error : please generate the weights first. {OUTPUT}model_TL_UNET.h5 not found")


if __name__ == "__main__":
    main()
