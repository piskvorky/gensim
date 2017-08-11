from __future__ import print_function
from __future__ import absolute_import
import sys
import argparse
import logging
from gensim.api import download
from gensim.api import catalogue
from gensim.api import info
if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s :%(name)s :%(levelname)s : %(message)s', filename="api.log", level=logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	logging.getLogger('').addHandler(console)
	parser = argparse.ArgumentParser(description="Gensim console API")
	group = parser.add_mutually_exclusive_group()
	group.add_argument(
        "-d", "--download", nargs=1,
        help="To download a corpus/model : python -m gensim -d "
        "corpus/model name")
	group.add_argument(
        "-i", "--info", nargs=1,
        help="To get information about a corpus/model : python -m gensim -i "
        "model/corpus name")
	group.add_argument(
        "-c", "--catalogue", help="To get the list of all models/corpus stored"
        " : python -m gensim -c", action="store_true")
	args = parser.parse_args()
	if sys.argv[1] == "-d" or sys.argv[1] == "--download":
		download(sys.argv[2])
	elif sys.argv[1] == "-i" or sys.argv[1] == "--info":
		info(sys.argv[2])
	elif sys.argv[1] == "-c" or sys.argv[1] == "--catalogue":
		catalogue()
