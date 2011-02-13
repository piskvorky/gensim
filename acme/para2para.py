#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import
################################################################################
import os.path
import sys
import numpy as np

from gensim.corpora.dictionaryExistingCounts import DictionaryExistingCounts
from gensim.parsing.preprocessing import preprocess_documents
from gensim import utils, models, similarities, matutils

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level = logging.INFO)
logging.info("running %s" % ' '.join(sys.argv))
program = os.path.basename(sys.argv[0])


# configuration
###############################################################################
wordIdsExtension        ='_wordids.txt'
modelExtension          ='_tfidf.model'

# paths
basepath          = "/Users/dedan/projects/mpi/"        #this is where data will be, not code
corpusPath        = 'data/corpora/wiki-mar2008/'
corpusname        = 'head500.noblanks.cor'              # for testing
workingcorpus     = basepath + corpusPath + corpusname


# functions
#############################################################################
# TODO this should be moved to some library (textcorpus?)
def loadDictionary(fname):
    """
    Loads previously stored mapping between words and their ids, in a file
    with extension wordIdsExtension='_wordids.txt'.

    This function came from wikiCorpus, but it's so general that it'd be better here.
    Note that this is not a dictionary object and cannot do doc2bow. we create a DictionaryExistingCounts
    to wrap that functionality. the flow is

    id2word, word2id = loadDictionary('corpus.cor_wordids.txt')
    dictObject = DictionaryExistingCounts(id2word, word2id)
    The resulting object can be used as the `id2word` parameter for input to transformations.
    """
    id2word = {}
    word2id = {} # it's cheaper to create the two in this loop than to zip them later
    for lineNo, line in enumerate(open(fname)):
        cols = line[:-1].split('\t')
        if len(cols) == 2:
            wordId, word = cols
        elif len(cols) == 3:
            wordId, word, docFreq = cols
        else:
            continue
        id2word[int(wordId)] = word # docFreq not used
        word2id[word]        = int(wordId)
    return id2word, word2id


# load the model from last run
############################################################################
logging.debug("loading dictionary %s" % workingcorpus + wordIdsExtension)
id2word, word2id    = loadDictionary( workingcorpus + wordIdsExtension)
dictionary          = DictionaryExistingCounts( word2id, id2word )

# load the model
logging.debug("loading model from %s" % workingcorpus + modelExtension)
tfidf = models.TfidfModel.load(workingcorpus + modelExtension)


# read in the texts 
################################################################################
text1       = utils.get_txt('text1.txt')
text2       = utils.get_txt('text2.txt')
pre_text1   = preprocess_documents(text1)
pre_text2   = preprocess_documents(text2)

# remove words that appear only once
allTokens   = sum(pre_text1, [])
allTokens   = sum(pre_text2, allTokens)
tokensOnce  = set(word for word in set(allTokens) if allTokens.count(word) == 1)
pre_text1       = [[word for word in text if word not in tokensOnce]
                for text in pre_text1]
pre_text2       = [[word for word in text if word not in tokensOnce]
                for text in pre_text2]


# create bag of word representation and count words missin in the dictionary
corpus1     = []
corpus2     = []
all_missing = {}
for text in pre_text1:
    bow, missing = dictionary.doc2bow(text);
    corpus1.append(bow)
    for key, val in missing.iteritems():
        all_missing[key] = all_missing.get(key,0) + val
for text in pre_text2:
    bow, missing = dictionary.doc2bow(text);
    corpus2.append(bow)
    for key, val in missing.iteritems():
        all_missing[key] = all_missing.get(key,0) + val


logging.info("all missing words")
logging.info(all_missing)

# transform bow to tfidf space
t1      = tfidf[corpus1]
t2      = tfidf[corpus2]

# compute similarities
res = np.zeros((len(t1), len(t2)))
for i, par1 in enumerate(t1):
    for j, par2 in enumerate(t2):
        res[i,j] = matutils.cossim(par1, par2)
print res
print (res == res.max())
