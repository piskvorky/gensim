#!/usr/bin/env python2.5

"""
USAGE: %s LANGUAGE
    Build tfidf term-document matrix from all articles in directories specified \
in the common.py config file. The directories must be in DML-CZ format. Program \
searches for articles with fulltext.txt and meta.xml files. \
Each meta.xml is parsed for MSCs and language, which must match LANGUAGE (eg. \
any, eng, fre, ita, ger, rus, ..).

This script has to be run prior to running gensim.py. Its output is a matrix file \
which serves as input to gensim.py

Example: ./build_tfidf.py eng
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
    
    # build individual input databases
    # each input database contains ALL articles (not only those with the selected language)
    for id, path in inputs.iteritems():
        iddb.create_maindb(id, path) # create main article databases (all languages)
        gc.collect() # try giving a hint to garbage collector to clean up...
    
    # merge databases into one, keeping only articles in the specified language (or 'any')
    iddb.merge(inputs, prefix, language)
    
    # build and store tfidf matrix
    docsim.buildTFIDFMatrices(dbFile = common.dbFile(prefix, language), prefix = prefix + '_' + language, contentType = 'alphanum_nohtml', saveMatrices = False)
    
    logging.info("finished running %s" % program)
