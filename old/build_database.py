#!/usr/bin/env python2.5

"""
USAGE: %s
    Process all articles in directories specified \
in the common.py config file. The directories must be in DML-CZ format. Program \
searches for articles with fulltext.txt and meta.xml files. \

This script has to be run prior to running build_tfidf.py. Its output are \
database files, which serves as input to build_tfidf.py.

Example: ./build_database.py 2>&1 | tee ~/xrehurek/results/build_database.log
"""


import logging
import sys
import os.path
import gc

import common
import iddb


if __name__ == '__main__':
    logging.basicConfig(level = common.PRINT_LEVEL)
    logging.root.level = common.PRINT_LEVEL
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 1:
        print globals()['__doc__'] % (program)
        sys.exit(1)
    inputs = common.INPUT_PATHS
    prefix = common.PREFIX
    
    # build individual input databases
    # each input database contains ALL articles (not only those with the selected language)
    for id, path in inputs.iteritems():
        iddb.create_maindb(id, path) # create main article databases (all languages)
        gc.collect()
    
    logging.info("finished running %s" % program)
