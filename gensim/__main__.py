import platform
import sys
import numpy
import scipy
import gensim
import argparse

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("--info", action="store_true", help="Display version of Gensim and its dpendencies")
        opt = parser.parse_args()
        if opt.info:
                print(platform.platform())
                print("Python", sys.version)
                print("NumPy", numpy.__version__)
                print("SciPy", scipy.__version__)
                print("gensim", gensim.__version__)

