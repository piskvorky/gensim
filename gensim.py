#!/usr/bin/env python2.5

"""
USAGE: %s METHOD LANGUAGE
    Generate similar.xml files from the matrix created by the build_tfidf.py script.
    METHOD is currently either tfidf, lsi or rp.

Example: ./gensim.py tfidf eng
"""


import logging
import sys
import os.path

import scipy.sparse

import common
import docsim
import ipyutils
import matutils


DRY_RUN = False # set to True to do everything EXCEPT actually writing out similar.xml files to disk

MIN_SCORE = 0.0
MAX_SIMILAR = 10 # set to 0 to save *WHOLE* similarity list 

DIM_RP = 300 # dimensionality for the random projections
DIM_LSI = 200 # dimensionality for lantent semantic indexing

SAVE_EMPTY = True

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

SIMILAR = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<related>%(articles)s
</related>
"""

def decomposeId(docId):
    """Decompose an article id back into (source, path from source base dir) 2-tuple.
    """
    sep = docId.find(os.sep)
    assert sep >= 0, "failed to decompose docId into source and path, '%s'" % docId
    source, path = docId[ : sep], docId[sep + 1 : ]
    fullPath = os.path.join(common.INPUT_PATHS[source], path) # create full path by merging source base dir and the id
    return source, path, fullPath

def generateSimilarXML(method, docId, tops):
    inputSource, inputId, outdir = decomposeId(docId)
    outfile = os.path.join(outdir, 'similar_%s.xml' % method)
    articles = ""
    numArticles = 0
    for _simId, score in tops:
        simId = ipyutils.docids[_simId]
        if score > MIN_SCORE and simId != docId: # at least MIN_SCORE and not identity
            author = ""
            title = ""
            suffix = ""
            source, id, _ = decomposeId(simId)
            articles += ARTICLE % locals()
            numArticles += 1
            if numArticles >= MAX_SIMILAR:
                break
    if SAVE_EMPTY or numArticles:
        logging.info("generating %s (%i similars)" % (outfile, numArticles))
        if not DRY_RUN:
            outfile = open(outfile, 'w')
            outfile.write(SIMILAR % locals())
            outfile.close()
    else:
        logging.debug("skipping %s (no similar found)" % outfile)

def loadTfIdf(language):
    tfidfFile = common.matrixFile(common.PREFIX + '_' + language + 'TFIDF_T.mm')
    logging.info("loading TFIDF matrix from %s" % tfidfFile)
    mat = matutils.loadMatrix(tfidfFile)
    logging.info("loaded TFIDF matrix of %i documents and %i terms" % (mat.shape[0], mat.shape[1]))
    return mat


# main program entry point
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
        mat = docsim.buildLSIMatrices(language, factors = DIM_LSI)
    elif method == 'rp':
        mat = docsim.buildRPMatrices(language, dimensions = DIM_RP)
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
