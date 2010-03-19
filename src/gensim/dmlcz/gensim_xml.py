#!/usr/bin/env python
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s LANGUAGE METHOD
    Generate similar.xml files, using a previously built model for METHOD.

Example: ./gensim_xml.py eng lsi
"""


import logging
import sys
import os.path
import re


import docsim
from gensim.corpora import sources, corpora
from gensim.models import lsimodel, ldamodel, tfidfmodel


# set to True to do everything EXCEPT actually writing out similar.xml files to disk.
# similar.xml files are NOT written if DRY_RUN is true.
DRY_RUN = True # False

# how many 'most similar' documents to store in each similar.xml?
MIN_SCORE = 0.0 # prune based on similarity score (all below MIN_SCORE are ignored)
MAX_SIMILAR = 10 # prune based on rank (at most MAX_SIMILAR are stored). set to 0 to store all of them (no limit).

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



if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    logging.root.level = logging.DEBUG
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])
    
    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % program
        sys.exit(1)
    language = sys.argv[1]
    method = sys.argv[2].strip().lower()
    
    logging.info("loading corpus mappings")
    try:
        dml = corpora.DmlCorpus.load(config.resultFile('.pkl'))
    except IOError, e:
        raise IOError("no word-count corpus found at %s; you must first generate it through gensim_build.py")
    config = dml.config

    logging.info("loading word id mapping from %s" % config.resultFile('wordids.txt'))
    id2word = corpora.DmlCorpus.loadDictionary(config.resultFile('wordids.txt'))
    logging.info("loaded %i word ids" % len(id2word))


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

