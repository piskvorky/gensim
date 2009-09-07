#!/usr/bin/env python2.5

import logging
import sys
import os.path
import codecs
import cPickle
import gc

import numpy
import scipy

import lsi
from DocumentCollection import DocumentCollection
import document
import matutils
import common
import ArticleDB
import Article
import randomprojections

import gensim

def convertId(fpath):
    dir = os.path.split(fpath)[0]
    dir, artNum = os.path.split(dir)
    result = os.path.split(dir)[1] + '-' + str(artNum[1:])
    return result

def pruneCollection(coll):
    coll.filterExtremes(extremes = 2)

#    for doc in coll.getDocs():
#        doc.setId(convertId(doc.getId()))
    coll.removeEmptyDocuments()

def getCollectionFS(basedir):
    """
    Create DocumentCollection from file system directory (use all 'fulltxt.txt' files)
    """
    coll = DocumentCollection.fromDir(
        basedir,
        document.DocumentFactory(encoding = 'utf8', sourceType = 'file', contentType = 'alphanum_nohtml', lowercase = True, keepTexts = False, keepTokens = False, keepPositions = False),
        walk = True, 
        accept = lambda fname: fname.endswith('fulltext.txt'))
##     m*n = m*k x k*k x k*n

def getCollection(arts, encoding = 'utf8', contentType = 'alphanum_nohtml'):
    logging.info('creating collection of %i documents' % (len(arts)))
    df = document.DocumentFactory(lowercase = True, sourceType = 'string', keepTokens = False, keepTexts = False, keepPositions = False, contentType = contentType, encoding = encoding)
    result = DocumentCollection()
    dictionary = {}
    for artCnt, art in enumerate(arts):
        if artCnt % 1000 == 0:
            logging.info("PROGRESS: at article #%i" % artCnt)
        doc = df.createDocument(art.body, docid = art.id_int, dictionary = dictionary)
        result.addDocument(doc)
    result.dictionary = dictionary
    logging.info('%i tokens (%i unique) in %i documents' % (result.getNumTokens(), result.getNumUniqueTokens(), result.numDocs()))
    return result

def getArts(fname, acceptNoBody = False, acceptNoMsc = False):
    db = ArticleDB.ArticleDB(common.dataFile(fname), mode = 'open')
    artsall = [Article.Article(rec) for rec in db.db]
    if not acceptNoBody:
        artsnobody = [art.id_int for art in artsall if not art.body]
        artsall = [art for art in artsall if art.body]
    else:
        artsnobody = []
    if not acceptNoMsc:
        artsnomsc = [art.id_int for art in artsall if not art.msc]
        artsall = [art for art in artsall if art.msc]
    else:
        artsnomsc = []
    
    logging.info('loaded %i articles (ignoring %i empty articles and %i articles with no MSC)' % (len(artsall), len(artsnobody), len(artsnomsc)))
    return artsall

def buildTFIDFMatrices(dbFile, prefix, contentType = 'alphanum_nohtml', saveMatrices = True):
    if isinstance(dbFile, basestring): # we are given filename of database from which to read articles
        coll = getCollection(getArts(dbFile, acceptNoMsc = True), encoding = 'utf8', contentType = contentType)
    else:
        coll = getCollection(dbFile, encoding = 'utf8', contentType = contentType) # we are given the list of articles directly

    pruneCollection(coll)
    gc.collect()
    coll.saveLex(common.OUTPUT_PATH, prefix)
    gc.collect()
    matTFIDF = coll.getTFIDFMatrix(sparse = True).tocsc()
    gc.collect()
    if saveMatrices:
        matutils.saveMatrix(matTFIDF, common.matrixFile(prefix + 'TFIDFraw.mm'), sparse = True)
    logging.info("normalizing sparse TFIDF matrix")
    matutils.normalizeColumns(matTFIDF, inplace = True)
    if saveMatrices:
        matutils.saveMatrix(matTFIDF, common.matrixFile(prefix + 'TFIDF.mm'), sparse = True)
    matTFIDF_T = matTFIDF.T
    matTFIDF = None
    gc.collect()
    matutils.saveMatrix(matTFIDF_T, common.matrixFile(prefix + 'TFIDF_T.mm'), sparse = True)
    return matTFIDF_T

def buildLSIMatrices(language, factors = 200):
    # load up scipy.sparse matrix, convert it to divisi tensor, then forget the scipy.sparse (to save RAM)
    import sparseSVD
    tfidf = sparseSVD.toTensor(gensim.loadTfIdf(language).T)
    
    # perform SVD on the tensor
    lsie = lsi.LSIEngine(docmatrix = tfidf, cover = factors, useSparse = True)

    # free up memory
    del tfidf
    del lsie.U
    
    # normalize all documents vectors to unit length
    result = scipy.sparse.lil_matrix(lsie.VT.T)
    
    return result


def buildRPMatrices(language, dimensions = 200):
    tfidf = gensim.loadTfIdf(language)
    # get random projection matrix
    projection = randomprojections.getRPMatrix(tfidf.shape[1], dimensions)

    # perform the actual projection
    logging.info("computing %s through %s random projection" % (projection.shape, tfidf.shape))
    rp = matutils.normalized_sparse(tfidf.tocsr() * projection.T.tocsc()) # project and force unit length on document vectors
    return rp


def getMostSimilar(docId, mat, n):
    """
    Return n most similar documents, along with their similarities, for the document
    vector docId. Similarity is computed by cosine measure.
    If n evaluates to False, return all documents and their similarities.
    Return type is [(sim, docId1), (sim, docId2), ...]
    """
    assert scipy.sparse.issparse(mat)
    mat = mat.tocsr()
    vec = mat[docId, :].T
    mat = mat.tocsc()
    logging.debug("vec has shape %ix%i" % vec.shape)
    logging.debug("mat has shape %ix%i" % mat.shape)
    logging.debug('multiplying mat * vec.T = %ix%i * %ix%i' % (mat.shape + vec.shape))
    result = mat * vec
    result = zip(xrange(result.shape[1]), result)
    result.sort(key = lambda item: -item[1])
    if n:
        result = result[:n]
    return result

    
    
def buildSimMatrices(prefix, doLSI = False, doTFIDF = False, doRP = False, mat = None):
    if doLSI:
        if mat == None:
            matLSI = matutils.loadMatrix(common.matrixFile(prefix + 'LSI_VT.mm'))
            logging.info("loaded %ix%i LSI matrix" % matLSI.shape)
            matLSI = matutils.normalized(matLSI)
            logging.info("normalized %ix%i LSI matrix" % matLSI.shape)
            logging.debug("sqrlen of first column=%f" % numpy.sum(matLSI[:, 0] * matLSI[:, 0]))
            #matLSI = matLSI.T
            #logging.info("transposed back into a %ix%i LSI matrix" % matLSI.shape)
        else:
            matLSI = mat
        logging.info("creating cossim of %ix%i LSI matrix" % matLSI.shape)
        symmatLSI = numpy.abs(numpy.dot(matLSI.T, matLSI))
        matutils.saveMatrix(symmatLSI, common.matrixFile(prefix + 'LSIsim.mm'))
    if doTFIDF:
        if mat == None:
            matTFIDF_T = matutils.loadMatrix(common.matrixFile(prefix + 'TFIDF_T.mm'))
        else:
            matTFIDF_T = mat
        logging.info("creating cossim of %ix%i TFIDF matrix" % (matTFIDF_T.shape[1], matTFIDF_T.shape[0]))
        symmatTFIDF = matutils.symMatrix(matTFIDF_T, matutils.innerabs)
        
        for i in xrange(symmatTFIDF.shape[0]): # force symmetry (nonsymmetry can be caused by low numerical precision)
            symmatTFIDF[i, i] = 1.0
            for j in xrange(i + 1, symmatTFIDF.shape[1]):
                symmatTFIDF[j, i] = symmatTFIDF[i, j]
        matutils.saveMatrix(symmatTFIDF, common.matrixFile(prefix + 'TFIDFsim.mm'))
    if doRP:
        if mat == None:
            matRP = matutils.loadMatrix(common.matrixFile(prefix + 'RP.mm'))
            matRP = matutils.normalized(matRP)
            logging.info("normalized %ix%i RP matrix" % matRP.shape)
            logging.debug("sqrlen of first column=%f" % numpy.sum(matRP[:, 0] * matRP[:, 0]))
        else:
            matRP = mat
        logging.info("creating cossim of %ix%i RP matrix" % matRP.shape)
        symmatRP = numpy.abs(numpy.dot(matRP.T, matRP))
        matutils.saveMatrix(symmatRP, common.matrixFile(prefix + 'RPsim.mm'))
        
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    buildTFIDFMatrices(dbFile = 'main_cmj.pdl', prefix = 'cmj', contentType = 'alphanum_nohtml')
#    buildLSIMatrices(prefix = 'cmj', factors = 200)
#    buildSimMatrices(prefix = 'cmj')
#    buildTFIDFMatrices('tex_casopis.pdl', 'tex', contentType = 'math')
#    buildLSIMatrices('tex', factors = 100)
#    buildSimMatrices('tex')
