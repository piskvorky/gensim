#!/usr/bin/env python2.5
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: gensim_build.py LANGUAGE
    Process the repository, accepting articles in LANGUAGE (or 'any'). 
    Store the word co-occurence matrix and id mappings, whic aren eeded for subsequent processing.
Example: ./gensim_build.py eng
    
-- or --

USAGE: gensim_genmodel.py LANGUAGE METHOD
    Generate topic models for the specified subcorpus. METHOD is currently one 
    of 'tfidf', 'lsi', 'lda', 'rp'.
Example: ./gensim_genmodel.py eng lsi

-- or --

USAGE: gensim.py LANGUAGE METHOD
    Generate similar.xml files, using a previously built model for METHOD.
Example: ./gensim.py eng lsi
"""


import logging
import sys
import os.path
import re

import sources
import corpora

import docsim
import lsimodel
import ldamodel
import tfidfmodel


#TODO:
# * prevest lda modely z blei na asteria01 na LdaModel objekt (trva tydny, nepoustet
#   znova...
# * prepsat random projections
# * logging per module -- ne vsechno pres logging.root, zlepseni modulovou prehlednost logu
# * rozdelit gensim.py na vic souboru, ne pres symlinky

SOURCE_LIST = [
               sources.DmlCzSource('dmlcz', '/Users/kofola/workspace/dml/data/dmlcz/'),
               sources.DmlSource('numdam', '/Users/kofola/workspace/dml/data/numdam/'),
               sources.ArxmlivSource('arxmliv', '/Users/kofola/workspace/dml/data/arxmliv/'),
               ]

RESULT_DIR = '/Users/kofola/gensim/results'

# set to True to do everything EXCEPT actually writing out similar.xml files to disk.
# similar.xml files are NOT written if DRY_RUN is true.
DRY_RUN = True # False

# how many 'most similar' documents to store in each similar.xml?
MIN_SCORE = 0.0 # prune based on similarity score (all below MIN_SCORE are ignored)
MAX_SIMILAR = 10 # prune based on rank (at most MAX_SIMILAR are stored). set to 0 to store all of them (no limit).

# internal method parameters
DIM_RP = 300 # dimensionality for the random projections
DIM_LSI = 200 # for lantent semantic indexing
DIM_LDA = 100 # for latent dirichlet allocation

# if there are no similar articles (after the pruning), do we still want to generate similar.xml?
SAVE_EMPTY = True

# xml template for similar article
ARTICLE = """
    <article weight="%(score)f">
        <authors>
            <author>%(author)s</author>
        </authors>
        <title>%(title)s</title>
        <suffix>%(suffix)s</suffix>
        <links>
            <link source="%(source)s" id="%(docId)s" />
        </links>
    </article>"""

# template for the whole similar.xml file (will be filled with multiple ARTICLE instances)
SIMILAR = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<related>%(articles)s
</related>
"""



def getMeta(fname):
    """
    Parse out author and title information from a meta.xml file, if possible.
    """
    author, title = "", ""
    try:
        meta = open(fname).read()
        title = re.findall('<title.*?>(.*?)</title>', meta, re.MULTILINE)[0]
        author = re.findall('<author.*?>(.*?)</author>', meta, re.MULTILINE)[0]
    except Exception, e:
        logging.warning("failed to parse meta at %s" % fname)
    return author, title


def buildDmlCorpus(config, language):
    dml = corpora.DmlCorpus()
    dml.processConfig(config, shuffle = True)
    dml.buildDictionary()
    dml.dictionary.filterExtremes(noBelow = 5, noAbove = 0.3) # ignore too (in)frequent words
    
    dml.save(config.resultFile('.pkl')) # save the mappings as binary data (actual documents are not saved, only their uris) 
    dml.saveAsText() # save id mappings and documents as text data (matrix market format)
    return dml


def generateSimilar(corpus, similarities, method):
    for docNo, topSims in enumerate(similarities): # for each document
        sourceId, (_, outPath) = corpus.documents[docNo]
        outfile = os.path.join(outPath, '_similar_%s.xml' % method) # similarities will be stored to this file
        articles = []
        for docNo2, score in topSims: # for each most similar
            source, (docId, docPath) = corpus.documents[docNo2]
            if score > MIN_SCORE and docNo != docNo2: # if similarity is above MIN_SCORE and not identity (=always maximum similarity, boring)
                suffix = ""
                author, title = getMeta(os.path.join(docPath, 'meta.xml')) # try to read metadata from meta.xml, if present
                articles.append(ARTICLE % locals()) # add the similar article to output
                if len(articles) >= MAX_SIMILAR:
                    break
        # now `articles` holds multiple strings in similar_*.xml format 
        if SAVE_EMPTY or articles:
            if not DRY_RUN: # only open output files for writing if DRY_RUN is false
                logging.info("generating %s (%i similars)" % (outfile, len(articles)))
                articles = ''.join(articles) # concat all similars to one string
                outfile = open(outfile, 'w')
                outfile.write(SIMILAR % locals()) # add xml headers and print to file
                outfile.close()
            else:
                logging.info("would be generating %s (%i similars)" % (outfile, len(articles)))
        else:
            logging.debug("skipping %s (no similar found)" % outfile)



#==============================================================================
if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    logging.root.level = logging.DEBUG
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__']
        sys.exit(1)
    language = sys.argv[1]
    
    # construct the config, which holds information about sources, data file filenames etc.
    config = corpora.DmlConfig('gensim_%s' % language, resultDir = RESULT_DIR, acceptLangs = [language])
    for source in SOURCE_LIST:
        config.addSource(source)
    
    if 'build' in program:
        buildDmlCorpus(config, language)
    elif 'genmodel' in program:
        if len(sys.argv) < 3:
            print globals()['__doc__']
            sys.exit(1)
        method = sys.argv[2].strip().lower()
        try:
            dml = corpora.DmlCorpus.load(config.resultFile('.pkl'))
        except IOError, e:
            raise IOError("no word-count corpus found at %s; you must first generate it through gensim_build.py")

        logging.info("loading word id mapping from %s" % config.resultFile('wordids.txt'))
        id2word = corpora.DmlCorpus.loadDictionary(config.resultFile('wordids.txt'))
        logging.info("loaded %i word ids" % len(id2word))

        if method == 'tfidf':
            corpus = corpora.MmCorpus(config.resultFile('bow.mm'))
            model = tfidfmodel.TfidfModel(corpus, id2word = id2word, normalize = True)
            model.save(config.resultFile('tfidfmodel.pkl'))
        elif method == 'lda':
            corpus = corpora.MmCorpus(config.resultFile('bow.mm'))
            model = ldamodel.LdaModel(corpus, id2word = id2word, numTopics = DIM_LDA)
            model.save(config.resultFile('ldamodel%i.pkl' % DIM_LDA))
        elif method == 'lsi' or method == 'lsa':
            # first, transform word counts to tf-idf weights
            corpus = corpora.MmCorpus(config.resultFile('bow.mm'))
            tfidf = tfidfmodel.TfidfModel(corpus, id2word = id2word, normalize = True)
            # then find the transformation from tf-idf to latent space
            model = lsimodel.LsiModel(tfidf.apply(corpus), id2word = id2word, numTopics = DIM_LSI)
            model.save(config.resultFile('lsimodel%i.pkl' % DIM_LSI))
        elif method == 'rp':
            raise NotImplementedError("Random Projections not converted to the new interface yet")
        else:
            raise ValueError('unknown topic extraction method: %s' % repr(method))
    else:
        if len(sys.argv) < 3:
            print globals()['__doc__']
            sys.exit(1)
        method = sys.argv[2].strip().lower()
        corpus = corpora.DmlCorpus.load(dml.pkl)
        input = corpora.MmCorpus(bow.mm)
        
        if method == 'tfidf':
            model = tfidfmodel.TfidfModel.load(modelfname('tfidf'))
        elif method == 'lsi':
            tfidf = tfidfmodel.TfidfModel.load(modelfname('tfidf'))
            input = corpora.TopicsCorpus(tfidf, input)
            model = lsimodel.LsiModel.load(modelfname('lsi'))
        elif method == 'lda':
            model = ldamodel.LdaModel.load(modelfname('lda'))
        else:
            raise ValueError('unknown method: %s' % repr(method))
        
        topics = corpora.TopicsCorpus(model, input) # documents from 'input' will be represented via 'model'
        sims = docsim.SparseMatrixSimilarity(topics, numBest = MAX_SIMILAR) # initialize structure which searches for similar documents
        generateSimilar(corpus, sims, method) # for each document, print MAX_SIMILAR nearest documents to a xml file, in dml-cz format
            
    logging.info("finished running %s" % program)

