#!/usr/bin/env python
# -*- coding: utf-8 -*-

# important,
# since testing this involves putting large structures into memory from disk, it's
# better to mount on a ramdisk. this is done with e.g. #mount -t tmpfs tmpfs /home/quesada/coding/gensim/data
# be careful that mounting will remove all contents of that folder!
import os.path
import sys



import gensim
from gensim.corpora.dictionaryExistingCounts import DictionaryExistingCounts
from gensim.corpora.wikiExternalParsingCorpus import WikiExternalParsingCorpus

from gensim.models.tfmodel import TfModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.parsing.preprocessing import *

import scipy
import scipy.io as sio
import scipy.stats as stats
print scipy.__version__

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level = logging.DEBUG)
logging.info("running %s" % ' '.join(sys.argv))
program = os.path.basename(sys.argv[0])
import numpy as np
from gensim import utils, models, similarities
import cPickle

# configuration
# we can save time by loading the objects from file. Set createCorpus to false
#sufixes
tfidfCpickleExtension   ='_tfidf.mm.Cpickle'
tfidfExtension          ='_tfidf.mm'
wordIdsExtension        ='_wordids.txt'

# flags
createCorpus =  True #False
# paths
corpusPath   = '/data/corpora/wiki-mar2008/'
corpusname = 'head500.noblanks.cor' # for testing
#corpusname =  'stemmedAllCleaned-fq10-cd10.noblanks.cor'
basepath       = "/home/quesada/coding/gensim/" #this is where data will be, not code
workingcorpus = basepath + corpusPath + corpusname
humanDataFilename = basepath + "data/humanTasks/lee-doc2doc/similarities0-1.txt"

# todo: ask to rewrite if there's already files of the same name
# todo: Write all config to a text file, read it from there. Maybe not in the class
# but on the file that uses the class and wraps it with command line args or text config file
# todo: make sure the logs go to the config file

# functions
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

# -1- load corpus plain text from file, creating a corpus object and index object, Save to file
# ------------------------------------------------------------------------------------
if (createCorpus==True):
    # creating a corpus produces the following files:
    # _wordids.txt (this one contains word freqs too)
    # _bow.mm
    # _tfidf.mm
    # _tfidf.mm.Cpickle
    input = workingcorpus + '.bz2'
    output = workingcorpus
    # build dictionary takes about 45 min
    logging.debug("-------calling .WikiExternParsingCorpus-------")
    logging.debug("input: " + input)
    wiki = WikiExternalParsingCorpus(input,keep_words = 200000)
    # save dictionary and bag-of-words, another ~45min
    wiki.saveAsText(output)
    del wiki

    # initialize corpus reader and word->id mapping
    from gensim.corpora import MmCorpus
    id2word, word2id = loadDictionary(output + wordIdsExtension)
    mm = MmCorpus(output + '_bow.mm') #abstract class; the only use is  to call the saveCorpus method

    # build tfidf ~20min
    logging.debug("-------calling TfidfModel-------")
    tfidf = TfidfModel(mm, id2word = id2word, normalize = True)

    # TODO serialize the entire tfidf model. We'll need it later  to create a sparseMatrixSimilarity object

    a = tfidf[mm]

    # save tfidf vectors in matrix market format
    # ~1.5h; result file is 14GB! bzip2'ed down to 4.5GB
    MmCorpus.saveCorpus(output + tfidfExtension, a, progressCnt = 10000)
    logging.info("finished running parsing using %s" % program)
#    ^^ All this is using streamed corpora. At no time the entire sparse matrix is in memory
#    Now we do put everything as a sparse matrix in memory. Skip this step if
#    you are running a laptop/computer with less than say 4 gb ram (it takes 3 to hold the sparse matrix)
#   we will read this into memory to write it as a cpickle, which will be read faster from disk
    # in future runs
    logging.debug("memory-loading tfidf matrix %s" % workingcorpus +tfidfExtension)
    index_tfidf = sio.mmread(output +tfidfExtension) # NOTE: not needed for the lee usecase
    logging.debug("pickling tfidf matrix to %s" % workingcorpus + tfidfCpickleExtension)
    try:
        f = open(workingcorpus + tfidfCpickleExtension, 'wb')
        cPickle.dump(index_tfidf, f, protocol = -1)
        f.close()
    except IOError as e:
        logging.debug(e)
    logging.debug("done pickling tfidf matrix %s" % workingcorpus + tfidfCpickleExtension)
else:
    #dictionary  = gensim.corpora.wikiExternParsingCorpus.loadDictionary( fname=corpusName + wordIdsExtension)
    logging.debug("loading dictionary %s" % workingcorpus + wordIdsExtension)
    id2word, word2id  = loadDictionary( workingcorpus + wordIdsExtension)
    logging.debug("loading pickled tfidf matrix to %s" % workingcorpus + tfidfCpickleExtension)
    try:
        f = open(workingcorpus + tfidfCpickleExtension, 'rb')
        index_tfidf = cPickle.load( f)
        f.close()
    except IOError as e:
        logging.debug(e)
    logging.debug("done loading pickled tfidf matrix %s" % workingcorpus + tfidfCpickleExtension)
    logging.info("finished running %s" % program)
#    print raw_input('holding here to see memory usage ') # todo measure time and memory


# / ------------------------------- /

# -2- load queries, that is, all docs in the lee dataset, and process them
#(remove stopwords etc)
leePath = basepath + '/data/corpora/lee/'
leeCorpusName = 'lee'
queries_filename = ( leePath + leeCorpusName + '.cor')
rawLeeTexts = utils.get_txt(queries_filename)

stoplist = utils.get_txt(basepath+"data/stoplistStoneDennisKwantes.txt")

DEFAULT_FILTERS = [ stem_text, strip_punctuation, remove_stopwords ]  # todo add these filters to log
preprocLeeTexts = preprocess_documents(rawLeeTexts)

# create a bow for each lee text. We need a Dictionary object first
# note that using DictionaryExistingCounts serves as a container for  word2id, id2word
# and also it takes cares of the bow conversion.
# todo: serializing this object would work  as well as having the '_wordids.txt' for ids
dictionary = DictionaryExistingCounts( word2id, id2word )

bow_queries_tfidf = [dictionary.doc2bow(text, allowUpdate=False, returnMissingWords=False) for text in preprocLeeTexts]

# ^^here the ids are  the ones in the larger corpus, because we use dictionary.
# note that these are raw freqs. Which is what we want. To keep them
# Simple checkout: similarities for the first lee doc, ordered by cosine
#sample_query_docid = 5
#bow_query_tfidf  = bow_queries_tfidf[sample_query_docid]
#testNN = utils.top_NN(bow_query_tfidf, rawCorpus, index_tfidf)
#print "raw query: " + rawLeeTexts[sample_query_docid]
#print "---------------------------------------------------------------------"
#print preprocLeeTexts[sample_query_docid]
#print "---------------------------------------------------------------------"
#ids, fqs = zip(*bow_query_tfidf)
#print ids
#print "---------------------------------------------------------------------"
#for result in testNN:
#    print result
#    print "\n"


# -3- compute cosine upper triangular matrix.

secondCorpusName = 'lee'
tfidf_lee   = models.TfidfModel(bow_queries_tfidf, dictionary.id2token, normalize=False )
tf_lee      = TfModel(bow_queries_tfidf, dictionary.id2token, normalize=False)


# note that the actual cell values we get only when using []
index_queries = similarities.SparseMatrixSimilarity(tfidf[bow_queries_tfidf])
# ^^ this is a very convoluted way of getting a csr_matrix from a bag of words!
# now, I could use vector by matrix operations to get an all vs all triang. matrix.
# Or I could do a nested loop. Since we'll need to do this only for the lee test,
# the less elegant, but easier solution (nested loop) is preferred

nqueries = index_queries.corpus.shape[0]
cosinesMat_model = np.zeros([nqueries, nqueries])

for sample_query_docid in range(nqueries):
    bow_query_tfidf  = bow_queries_tfidf[sample_query_docid]
    bow = tf_lee[bow_query_tfidf] # here's the actual tf values
    cosinesMat_model[sample_query_docid, :] = index_queries[bow]

print "foo"

# we need only the upper triangle, and we remove the diagonal
utri_list_model = utils.array2utrilst(cosinesMat_model)

# 4 - load human judgments
cosinesMat_human = np.loadtxt(humanDataFilename)
utri_list_human  = utils.array2utrilst(cosinesMat_human)

# 5 - compare model and human (corr)
print scipy.stats.pearsonr(np.array(utri_list_human), np.array(utri_list_model))
#(0.59600820805369992, 1.0840017782760449e-118)
# (0.54238525236444413, 1.2550974863613166e-94)
#(0.56247807052742493, 3.9434991804464149e-103) using tfidf_lee, but this is not conceptually right
# since the idf is computed on the query corpus of 50 queries. Not what we want.
# we could use the idf for the wiki corpus, if it was stored anywhere.

# note that we are not using any weighting for the queries
# we can try with weighting (tfidf) of different kinds

# --- finis ---

