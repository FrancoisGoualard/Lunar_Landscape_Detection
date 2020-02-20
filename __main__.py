import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from content.Displayer import Displayer
from content.imageAnalysis import main_process, prediction_test, load_model
from config import HERE

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "display":
            Displayer().run_without_image()
    elif len(sys.argv) == 3:
        if sys.argv[1] == "test":
            try:
                TransferLearningModel = load_model(f'{HERE}output/model_TL_UNET.h5')
                prediction_test(TransferLearningModel, sys.argv[2])
            except:
                print(f"Error : please generate the weights first. {HERE}output/model_TL_UNET.h5 not found")
    else:
        main_process()