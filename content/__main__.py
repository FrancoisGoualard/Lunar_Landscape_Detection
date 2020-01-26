import sys

from content.Displayer import Displayer
from content.imageAnalysis import parsing, main_process

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "display":
            Displayer().run_without_image()
    else:
        main_process()