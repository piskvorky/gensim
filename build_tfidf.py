#!/usr/bin/env python2.5

"""
USAGE: %s LANGUAGE
    Build tfidf term-document matrix from all articles in the specified language, \
    eg. any, eng, fre, ita, ger, rus, ....

This script has to be run prior to running gensim.py. Its output is a matrix file \
which serves as input to gensim.py

Example: ./build_tfidf.py eng 2>&1 | tee ~/xrehurek/results/build_eng.log
"""


import logging
import sys
import os.path
import gc

import common
import iddb
import docsim


if __name__ == '__main__':
    logging.basicConfig(level = common.PRINT_LEVEL)
    logging.root.level = common.PRINT_LEVEL
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % (program)
        sys.exit(1)
    language = sys.argv[1]
    inputs = common.INPUT_PATHS
    prefix = common.PREFIX
    
    # merge databases into one, keeping only articles in the specified language (or 'any' to keep all languages)
    iddb.merge(inputs, prefix, language)
    
    # build and store tfidf matrix
    docsim.buildTFIDFMatrices(dbFile = common.dbFile(prefix, language), prefix = prefix + '_' + language, contentType = 'alphanum_nohtml', saveMatrices = False)
    
    logging.info("finished running %s" % program)
