#!/usr/bin/env python2.5

import logging
import sys
import os.path
import codecs
import cPickle

import numpy
import scipy

import lsi # needed for Latent Semantic Indexing
import ldamodel # needed for Latent Dirichlet Allocation
import randomprojections # needed for Random Projections

from DocumentCollection import DocumentCollection
import document
import matutils
import common
import ArticleDB
import Article

import gensim

import sources
import corpus


def buildDmlCorpus(language):
    numdam = sources.DmlSource('numdam', sourcePath['numdam'])
    dmlcz = sources.DmlSource('dmlcz', sourcePath['dmlcz'])
    arxmliv = sources.ArxmlivSource('arxmliv', sourcePath['arxmliv'])
    
    config = corpus.DmlConfig('gensim', resultDir = sourcePath['results'], acceptLangs = [language])
    
    dml = corpus.DmlCorpus()
    dml.processConfig(config)
    dml.buildDictionary()
    dml.dictionary.filterExtremes()
    
    dml.save()
    dml.saveDebug()
    dml.saveAsBow()
    
                                                               
def generateSimilar(method, docSim):
    

dml-cz gensim workflow:
    for tfidf/lsi:
        make an incremental tfidf wrapper over bag-of-words (precompute idfs, make tfidf vectors from bow on the fly)
        for tfidf:
            mat = whole tfidf in memory (scipy.sparse)
        for lsi:
            mat = lsi(tfidf wrapper)
    for lda:
        mat = lda(bow wrapper)
    mat now contains in-memory representation of the corpus (dense doc x 200 for lsi/lda, sparse doc x terms for tfidf)
    interface for similarities?
    iterative getSim(mat, doc): return most similar documents (do not explicitly realize the whole doc x doc matrix!)
    
    for dml, iterate over corpus documents:
        call the general docsim.getSim(mat, docNo)
        store MAX_SIMILAR to the same dir as input in dml similar.xml format
    
    
    
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
    Create DocumentCollection from file system directory (use all 'fulltext.txt' files)
    """
    coll = DocumentCollection.fromDir(
        basedir,
        document.DocumentFactory(encoding = 'utf8', sourceType = 'file', contentType = 'alphanum_nohtml', lowercase = True, keepTexts = False, keepTokens = False, keepPositions = False),
        walk = True, 
        accept = lambda fname: fname.endswith('fulltext.txt'))

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
    coll.saveLex(common.OUTPUT_PATH, prefix)
    matTFIDF = coll.getTFIDFMatrix(sparse = True).tocsc()
    if saveMatrices:
        matutils.saveMatrix(matTFIDF, common.matrixFile(prefix + 'TFIDFraw.mm'), sparse = True)
    logging.info("normalizing sparse TFIDF matrix")
    matutils.normalizeColumns(matTFIDF, inplace = True)
    if saveMatrices:
        matutils.saveMatrix(matTFIDF, common.matrixFile(prefix + 'TFIDF.mm'), sparse = True)
    matTFIDF_T = matTFIDF.T
    matTFIDF = None
    matutils.saveMatrix(matTFIDF_T, common.matrixFile(prefix + 'TFIDF_T.mm'), sparse = True)
    return matTFIDF_T

def saveTopics(fname, topicMatrix, id2word):
    """
    Store topics from topicMatrix into a file, one topic per line. The line format
    is: 
    word1:value1[TAB]word2:value2[TAB]...
    where words are stored in decreasing order of value.
    
    topicMatrix is a (numWords x numTopics) dense matrix. Each entry in the topic
    matrix quantifies the important of a particular word for a particular topic.
     
    id2word is a mapping between word ids (matrix row indices) and the words 
    themselves (strings).
    
    Note that the order of topics matters, as the line number is later used as 
    topic id.
    """
    logging.info("storing %i topics (vocabulary of %i words) to %s" %
                 (topicMatrix.shape[1], topicMatrix.shape[0], fname))
    fout = open(fname, 'w')
    for topic in topicMatrix.T:
        sortedTopic = topic.argsort()[::-1] # words with the highest score come first
        wordStrings = ["%s:%f" % (id2word[i].encode('utf8'), topic[i]) for i in sortedTopic]
        fout.write("\t".join(wordStrings) + '\n')
    fout.close()

                
def buildLSIMatrices(language, factors = 200, saveConcepts = None):
    # load up divisi tensor directly from file
    tfidf = gensim.loadTfIdf(language, asTensor = True)
    
    # perform SVD on the tensor
    lsie = lsi.LSIEngine(docmatrix = tfidf, cover = factors, useSparse = True)

    # save the concepts as lists of words (human readable)
    del tfidf # free up memory
    if saveConcepts:
        import ipyutils
        assert len(ipyutils.tokenids) == lsie.U.shape[0] # make sure there is no word/index mismatch
        saveTopics(saveConcepts, lsie.U, ipyutils.tokenids)
    del lsie.U # free up memory
    
    # convert the matrix to sparse coo format, documents as rows
    result = scipy.sparse.coo_matrix(lsie.VT.T)
    
    return result


def buildRPMatrices(language, dimensions = 200):
    # load original tfidf matrix
    tfidf = gensim.loadTfIdf(language)

    # get random projection matrix
    projection = randomprojections.getRPMatrix(tfidf.shape[1], dimensions)

    # perform the actual projection
    logging.info("computing %s through %s random projection" % (projection.shape, tfidf.shape))
    rp = matutils.normalized_sparse(tfidf.tocsr() * projection.T.tocsc()) # project and force unit length on document vectors
    return rp


def buildLDAMatrix(language, id2word, numTopics = 200):
    # initialize the corpus
    tfidfFile = common.matrixFile(common.PREFIX + '_' + language + 'TFIDF_T.mm')
    corpus = matutils.MmCorpus(tfidfFile)
    
    # train the LDA model
    lda = ldamodel.LdaModel.fromCorpus(corpus, id2word = id2word, numTopics = numTopics, initMode = 'random')
#    lda = ldamodel.LdaModel.load('/Users/kofola/workspace/dml/data/results/cmj_topic100.lda_model')
    
    # store the trained LDA model
    ldaFile = common.dataFile(common.PREFIX + '_' + language + '.lda_model')
    lda.save(ldaFile)
    
    # store the topics in human-readable format (for inspection)
    saveTopics(ldaFile + '.topics', lda.getTopicsMatrix().T, lda.id2word)
    
    # now do another sweep over the corpus, estimating topic distribution for each 
    # corpus document
    logging.info("model trained; computing topic distributions")
    result = numpy.column_stack(lda.inference(doc)[2] for doc in corpus) # [2] because third result is the gamma
    
    # convert the result to scipy sparse format
    result = scipy.sparse.coo_matrix(result.T)
    
    logging.info("estimated topic distributions into a %sx%s matrix" % result.shape)
    return result
        
        
    

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
