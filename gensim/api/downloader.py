from __future__ import print_function
from __future__ import absolute_import
import argparse
from gensim.api import download
from gensim.api import catalogue
from gensim.api import info
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gensim console API")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--download", nargs=1, help="To download a corpus/model : python -m gensim -d corpus/model name")
    group.add_argument("-i", "--info", nargs=1, help="To get information about a corpus/model : python -m gensim -i model/corpus name")
    group.add_argument("-c", "--catalogue", help="To get the list of all models/corpus stored : python -m gensim -c", action="store_true")
    args = parser.parse_args()
    if args.download is not None:
        download(args.download[0])
    elif args.info is not None:
        info(args.info[0])
    elif args.catalogue is not None:
        catalogue(print_list=True)
