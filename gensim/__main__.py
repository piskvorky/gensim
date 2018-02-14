import argparse
from gensim import package_info

parser = argparse.ArgumentParser()
parser.add_argument("--info", action="store_true", help="Display version of Gensim and its dpendencies")
opt = parser.parse_args()

if opt.info:
    dict = package_info()
    print("\nGensim ", dict["gensim"][0], " from ", dict["gensim"][1], "\n")
    print("FAST_VERSION ", dict["fast_version"], "\n\n")
    print("python ", dict["python"], "\n\n")
    print("Platform \t", dict["platform"], "\n")
    print("NumPy \t\t", dict["NumPy"], "\n")
    print("SciPy \t\t", dict["SciPy"], "\n")
