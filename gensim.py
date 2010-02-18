#!/usr/bin/env python2.5

"""
USAGE: %s METHOD LANGUAGE
    Generate similar_method.xml files from the matrix created by the build_tfidf.py script.
    METHOD is currently either tfidf, lsi or rp.

Example: ./gensim.py tfidf eng
"""


import logging
import sys
import os.path
import re

import common
import docsim
import lsimodel
import ldamodel
import tfidfmodel


#TODO:
# 1) tfidf prepsat z matutils do samostatneho tfidfmodel.py (plus __getitem__, aby 
#    to byla transformace)
# 2) dodelat DmlCzSource, ktery dedi z DmlSource ale navic kontroluje existenci 
#    souboru dspace_id (a vraci jeho obsah jako id); uri = (docId, dirPath)
# 3) pospravovat cesty (common.py, cesty ke slovniku, bow)
# 4) pustit a odladit
# 5) prevest lda modely z blei na asteria01 na LdaModel objekt (trva tydny, nepoustet
#    znova...
# 5.5) dat do modelu id2word = None a dopocitat z korpusu jako default?


# set to True to do everything EXCEPT actually writing out similar.xml files to disk.
# similar.xml files are NOT written if DRY_RUN is true.
DRY_RUN = False

# how many 'most similar' documents to store in each similar.xml?
MIN_SCORE = 0.0 # prune based on similarity score (all below MIN_SCORE are ignored)
MAX_SIMILAR = 10 # prune based on rank (at most MAX_SIMILAR are stored). set to 0 to all of them (no limit).

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


def buildDmlCorpus(language):
    numdam = sources.DmlSource('numdam', sourcePath['numdam'])
    dmlcz = sources.DmlCzSource('dmlcz', sourcePath['dmlcz'])
    arxmliv = sources.ArxmlivSource('arxmliv', sourcePath['arxmliv'])
    
    config = corpora.DmlConfig('gensim', resultDir = sourcePath['results'], acceptLangs = [language])
    
    dml = corpora.DmlCorpus()
    dml.processConfig(config)
    dml.buildDictionary()
    dml.dictionary.filterExtremes(noBelow = 5, noAbove = 0.3) # ignore too (in)frequent words
    
    dml.save() # save the whole object as binary data
    dml.saveDebug() # save docNo->docId mapping, and termId->term mapping in text format
    dml.saveAsMatrix() # save word count matrix
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
    logging.basicConfig(level = common.PRINT_LEVEL)
    logging.root.level = common.PRINT_LEVEL
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % (program)
        sys.exit(1)
    method = sys.argv[1]
    language = sys.argv[2]
    
    if 'build' in program:
        buildDmlCorpus(language)
        sys.exit(0)
    if 'genmodel' in program:
        if 'tfidf' in program:
            corpus = corpora.MmCorpus(dml_bow)
            model = tfidfmodel.TfidfModel(corpus)
            model.save(modelfname('tfidf'))
        if 'lda' in program:
            corpus = corpora.MmCorpus(dml_bow)
            id2word = loadDictionary(dml_dict.txt)
            model = ldamodel.LdaModel(corpus, id2word, numTopics = DIM_LDA)
            model.save(modelfname('lda'))
        elif 'lsi' in program:
            # first, transform word counts to tf-idf weights
            corpus = corpora.MmCorpus(dml_bow)
            model = tfidfmodel.TfidfModel(corpus)
            tfidf = corpora.TopicsCorpus(model, corpus)
            # then find the transformation from tf-idf to latent space
            id2word = loadDictionary(dml_dict.txt)
            model = lsimodel.LsiModel(tfidf, id2word, numTopics = DIM_LSI)
            model.save(modelfname('lsi'))
        else:
            logging.critical('unknown topic extraction method in %s' % program)
        sys.exit(0)
    
    if 'gensim' in program:
        corpus = corpora.DmlCorpus.load(dml.pkl)
        input = corpora.MmCorpus(bow.mm)
        
        if model == 'tfidf':
            model = tfidfmodel.TfidfModel.load(modelfname('tfidf'))
        elif method == 'lsi':
            tfidf = tfidfmodel.TfidfModel.load(modelfname('tfidf'))
            input = corpora.TopicsCorpus(tfidf, input)
            model = lsimodel.LsiModel.load(modelfname('lsi'))
        elif method == 'lda':
            model = ldamodel.LdaModel.load(modelfname('lda'))
        
        topics = corpora.TopicsCorpus(model, input) # documents from 'input' will be represented via 'model'
        sims = docsim.SparseMatrixSimilarity(topics, numBest = MAX_SIMILAR) # initialize structure which searches for similar documents
        generateSimilar(corpus, sims, method) # for each document, print MAX_SIMILAR nearest documents to a xml file, in dml-cz format
            
    logging.info("finished running %s" % program)

