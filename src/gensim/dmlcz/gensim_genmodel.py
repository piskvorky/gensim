#!/usr/bin/env python
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s LANGUAGE METHOD
    Generate topic models for the specified subcorpus. METHOD is currently one \
of 'tfidf', 'lsi', 'lda', 'rp'.

Example: ./gensim_genmodel.py eng lsi
"""


import logging
import sys
import os.path
import re


from gensim.corpora import sources, dmlcorpus, MmCorpus
from gensim.models import lsimodel, ldamodel, tfidfmodel

import gensim_build


# internal method parameters
DIM_RP = 300 # dimensionality for the random projections
DIM_LSI = 200 # for lantent semantic indexing
DIM_LDA = 100 # for latent dirichlet allocation



if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    logging.root.level = logging.DEBUG
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    language = sys.argv[1]
    method = sys.argv[2].strip().lower()
    
    logging.info("loading corpus mappings")
    config = dmlcorpus.DmlConfig('gensim_%s' % language, resultDir = gensim_build.RESULT_DIR, acceptLangs = [language])

    logging.info("loading word id mapping from %s" % config.resultFile('wordids.txt'))
    id2word = dmlcorpus.DmlCorpus.loadDictionary(config.resultFile('wordids.txt'))
    logging.info("loaded %i word ids" % len(id2word))
    
    corpus = MmCorpus(config.resultFile('bow.mm'))

    if method == 'tfidf':
        model = tfidfmodel.TfidfModel(corpus, id2word = id2word, normalize = True)
        model.save(config.resultFile('tfidfmodel.pkl'))
    elif method == 'lda':
        model = ldamodel.LdaModel(corpus, id2word = id2word, numTopics = DIM_LDA)
        model.save(config.resultFile('ldamodel.pkl'))
    elif method == 'lsi':
        # first, transform word counts to tf-idf weights
        tfidf = tfidfmodel.TfidfModel(corpus, id2word = id2word, normalize = True)
        # then find the transformation from tf-idf to latent space
        model = lsimodel.LsiModel(tfidf[corpus], id2word = id2word, numTopics = DIM_LSI)
        model.save(config.resultFile('lsimodel.pkl'))
    elif method == 'rp':
        raise NotImplementedError("Random Projections not converted to the new interface yet")
    else:
        raise ValueError('unknown topic extraction method: %s' % repr(method))
    
    if True:
        MmCorpus.saveCorpus(config.resultFile('corpus_%s.mm' % method), model[corpus])
            
    logging.info("finished running %s" % program)

