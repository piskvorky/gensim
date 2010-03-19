#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Standalone program that can estimate parameters for Latent Dirichlet Allocation \
(LDA) and/or perform inference.

To estimate (model will be written to datafile.model): 
./lda_estimate.py k datafile
example: ./lda_estimate.py 200 ~/ldadata/trndocs.low

To do inference (document likelihoods and gammas will be written to datafile.lda_inferred):
./lda_infer.py modelfile datafile
example: ./lda_infer.py ~/ldadata/trndocs.low.model ~/ldadata/newdocs.low

Data files may be either in list-of-words format (*.low) or matrix market \
format (*.mm).
"""


import logging
import sys
import os.path

from gensim.corpora import corpora # for input data i/o

from ldamodel import LdaModel # lda inference/estimation


PRINT_TOPICS = 10 # when printing model topics, how many top words to print out?



if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    logging.info("running %s" % " ".join(sys.argv))
    
    program = os.path.basename(sys.argv[0])
    
    if 'estimate' in program:
        # make sure we have enough cmd line parameters
        if len(sys.argv) < 3:
            print globals()["__doc__"]
            sys.exit(1)
        
        # parse cmd line
        k = int(sys.argv[1])
        datafile = sys.argv[2]
        
        # load corpus
        if datafile.endswith('.mm'):
            corpus = corpora.MmCorpus(datafile)
            # there is no word id mapping in MM format; use word=wordId
            id2word = dict((wordId, str(wordId)) for wordId in xrange(corpus.numTerms))
        else:
            corpus = corpora.CorpusLow(datafile)
            id2word = corpus.id2word
        #corpus.saveAsBlei()
        # run parameter estimation; this is the step that takes the most time
        model = LdaModel(id2word = id2word, numTopics = k)
        model.initialize(corpus)
        
        # store parameters, print topics info (for sanity check)
        model.save(datafile + '.model')
        if PRINT_TOPICS:
            logging.info("printing topics (top %i words)" % PRINT_TOPICS)
            model.printTopics(numWords = PRINT_TOPICS)
            print '=' * 40
    elif 'infer' in program:
        # make sure we have enough cmd line parameters
        if len(sys.argv) < 3:
            print globals()["__doc__"]
            sys.exit(1)
        
        # parse cmd line
        modelfile = sys.argv[1]
        datafile = sys.argv[2]
        
        # load model
        model = LdaModel.load(modelfile)
        if datafile.endswith('.mm'):
            corpus = corpora.MmCorpus(datafile)
        else:
            corpus = corpora.CorpusLow(datafile, id2word = model.id2word)
        # do the actual inference
        model.infer(corpus) # output is saved to datafile.lda_inferred
    else:
        print globals()["__doc__"]
        sys.exit(1)
    
    logging.info("finished running %s" % program)
