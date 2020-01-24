from content.Displayer import Displayer
import sys

if __name__ == "__main__":
    if sys.argv[1] == "display":
        Displayer().run()