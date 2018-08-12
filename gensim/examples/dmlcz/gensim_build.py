#!/usr/bin/env python
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s LANGUAGE
    Process the repository, accepting articles in LANGUAGE (or 'any').
    Store the word co-occurence matrix and id mappings, which are needed for subsequent processing.

Example: ./gensim_build.py eng
"""


import logging
import sys
import os.path

from gensim.corpora import sources, dmlcorpus

PREFIX = 'dmlcz'

AT_HOME = False

if AT_HOME:
    SOURCE_LIST = [
        sources.DmlCzSource('dmlcz', '/Users/kofola/workspace/dml/data/dmlcz/'),
        sources.DmlSource('numdam', '/Users/kofola/workspace/dml/data/numdam/'),
        sources.ArxmlivSource('arxmliv', '/Users/kofola/workspace/dml/data/arxmliv/'),
     ]

    RESULT_DIR = '/Users/kofola/workspace/dml/data/results'

else:

    SOURCE_LIST = [
        sources.DmlCzSource('dmlcz', '/data/dmlcz/data/share'),
        sources.DmlSource('numdam', '/data/dmlcz/data/numdam'),
        sources.ArxmlivSource('arxmliv', '/data/dmlcz/data/arxmliv'),
    ]

    RESULT_DIR = '/data/dmlcz/xrehurek/results'


def buildDmlCorpus(config):
    dml = dmlcorpus.DmlCorpus()
    dml.processConfig(config, shuffle=True)
    dml.buildDictionary()
    dml.dictionary.filterExtremes(noBelow=5, noAbove=0.3)  # ignore too (in)frequent words

    dml.save(config.resultFile('.pkl'))
    dml.saveAsText()  # save id mappings and documents as text data (matrix market format)
    return dml


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s", ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    language = sys.argv[1]

    # construct the config, which holds information about sources, data file filenames etc.
    config = dmlcorpus.DmlConfig('%s_%s' % (PREFIX, language), resultDir=RESULT_DIR, acceptLangs=[language])
    for source in SOURCE_LIST:
        config.addSource(source)
    buildDmlCorpus(config)

    logging.info("finished running %s", program)
