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

import scipy.sparse

import common
import docsim
import ipyutils
import matutils


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

# template for similar article
ARTICLE = """
    <article weight="%(score)f">
        <authors>
            <author>%(author)s</author>
        </authors>
        <title>%(title)s</title>
        <suffix>%(suffix)s</suffix>
        <links>
            <link source="%(source)s" id="%(id)s" />
        </links>
    </article>"""

# template for the whole similar.xml file (will be filled with multiple ARTICLE instances)
SIMILAR = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<related>%(articles)s
</related>
"""



def buildDmlCorpus(language):
    numdam = sources.DmlSource('numdam', sourcePath['numdam'])
    dmlcz = sources.DmlSource('dmlcz', sourcePath['dmlcz'])
    arxmliv = sources.ArxmlivSource('arxmliv', sourcePath['arxmliv'])
    
    config = corpora.DmlConfig('gensim', resultDir = sourcePath['results'], acceptLangs = [language])
    
    dml = corpora.DmlCorpus()
    dml.processConfig(config)
    dml.buildDictionary()
    dml.dictionary.filterExtremes(noBelow = 5, noAbove = 0.3)
    
    dml.save() # save the whole object as binary data
    dml.saveDebug() # save docNo->docId mapping, and termId->term mapping in text format
    dml.saveAsMatrix() # save word count matrix and tfidf matrix
    
                                                               
def generateSimilar(method, docSim):
    pass

    

def decomposeId(docId):
    """
    Decompose an article id back into (source, path from source base dir, full filesystem path) 3-tuple.
    """
    sep = docId.find(os.sep)
    assert sep >= 0, "failed to decompose docId into source and path, '%s'" % docId
    source, path = docId[ : sep], docId[sep + 1 : ]
    fullPath = os.path.join(common.INPUT_PATHS[source], path) # create full path by merging source base dir and the id
    return source, path, fullPath


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


def generateSimilarXML(method, docId, tops):
    inputSource, inputId, outdir = decomposeId(docId)
    outfile = os.path.join(outdir, 'similar_%s.xml' % method)
    articles = ""
    numArticles = 0
    for _simId, score in tops:
        simId = ipyutils.docids[_simId]
        if score > MIN_SCORE and simId != docId: # at least MIN_SCORE and not identity
            suffix = "" # FIXME k cemu byl vubec zamyslen tag 'suffix' v similar.xml?
            source, id, simdir = decomposeId(simId)
            author, title = getMeta(os.path.join(simdir, 'meta.xml'))
            articles += ARTICLE % locals()
            numArticles += 1
            if numArticles >= MAX_SIMILAR:
                break
    if SAVE_EMPTY or numArticles:
        logging.info("generating %s (%i similars)" % (outfile, numArticles))
        if not DRY_RUN: # only open output files for writing if DRY_RUN is false
            outfile = open(outfile, 'w')
            outfile.write(SIMILAR % locals())
            outfile.close()
    else:
        logging.debug("skipping %s (no similar found)" % outfile)


def loadTfIdf(language, asTensor = False):
    tfidfFile = common.matrixFile(common.PREFIX + '_' + language + 'TFIDF_T.mm')
    logging.info("loading TFIDF matrix from %s" % tfidfFile)
    if asTensor:
        import sparseSVD
        mat = sparseSVD.tensorFromMtx(tfidfFile, transposed = True)
    else:
        mat = matutils.loadMatrix(tfidfFile)
    logging.info("loaded TFIDF matrix of %i documents and %i terms" % (mat.shape[0], mat.shape[1]))
    return mat


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
    inputs = common.INPUT_PATHS
    
    # load document_id<->document_index, term<->term_index mappings
    ipyutils.loadDicts(prefix = common.PREFIX + '_' + language)
    
    # next, create a method matrix which will serve for cossim computations.
    # different methods create different matrices, but all return a sparse matrix of the shape (numDocs x N)
    if method == 'tfidf':
        mat = loadTfIdf(language)
    elif method == 'lsi':
        conceptFile = common.dataFile(common.PREFIX + '_' + language + '.lsa_concepts%i' % DIM_LSI)
        mat = docsim.buildLSIMatrices(language, factors = DIM_LSI, saveConcepts = conceptFile)
    elif method == 'rp':
        mat = docsim.buildRPMatrices(language, dimensions = DIM_RP)
    elif method == 'lda':
        mat = docsim.buildLDAMatrix(language, ipyutils.tokenids, numTopics = DIM_LDA)
    else:
        assert False, "unknown method '%s'" % method
    
    # make sure method matrix contains documents (=rows) of unit length
    logging.info("normalizing all vectors to unit length")
    mat = matutils.normalizeSparseRows(mat).tocsr()

    # and for each document, output its most similar documents to the file similar_METHOD.xml 
    for intId, docId in ipyutils.docids.iteritems():
        logging.info("processing document intId=%i, docId=%s" % (intId, docId))
        
        # get the document vector, of shape 1xN
        vec = mat[intId, :].T
        
        # compute cosine similarity against every other document in the collection
        tops = (mat * vec).todense() # multiply (=cossim for unit length  vectors) sparse matrix and sparse vector and convert the result to dense vector (normal numpy array)
        
        # sort the documents by their similarity, most similar come first
        sims = [tops.A[i, 0] for i in xrange(tops.shape[0])]
        tops = zip(xrange(tops.shape[0]), sims)
        tops.sort(key = lambda item: -item[1]) # sort by -sim => highest cossim first

        # and generate the files (unless DRY_RUN global var is set, in which case no file is actually written to disk)
        logging.debug("first ten most similar documents: %s" % (tops[:10]))
        generateSimilarXML(method, docId, tops)
        
    logging.info("finished running %s" % program)
