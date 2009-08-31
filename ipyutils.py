import sys

import matutils
import numpy
import scipy.sparse
import os.path
import codecs
import logging

import common

logging.basicConfig(level = logging.DEBUG)

def loadDicts(prefix = 'cmj'):
    global docids, rdocids, tokenids
    docids = [item.strip().split('\t') for item in open(common.dataFile(prefix + '_docids.txt')).readlines()]
    docids = dict([(int(id), val) for id, val in docids]) # map int -> int_id
    rdocids = dict([(v, k) for k, v in docids.iteritems()]) # map int_id -> int
    tokenids = [item.strip().split('\t') for item in codecs.open(common.dataFile(prefix + '_tokenids.txt'), 'r', 'utf8').readlines()]
    tokenids = dict([(int(id), val) for id, val in tokenids])

def loadSimMatrices(prefix = 'cmj'):
    global m, l
    m = matutils.loadMatrix(common.matrixFile(prefix + 'TFIDFsim.mm'))
    l = matutils.loadMatrix(common.matrixFile(prefix + 'LSIsim.mm'))
    return m, l

def loadVSMatrices(prefix = 'cmj', loadTFIDF = True, loadLSI = True):
    """load and return vector space matrices of the documents"""
    global vecTFIDF, vecLSI
    vecTFIDF, vecLSI = None, None
    if loadTFIDF:
        vecTFIDF = matutils.loadMatrix(common.matrixFile(prefix + 'TFIDF.mm'))
    if loadLSI:
        vecLSI = matutils.loadMatrix(common.matrixFile(prefix + 'LSI_VT.mm'))
    return vecTFIDF, vecLSI

def loadConcepts(prefix = 'cmj'):
    """load and return the LSI_U matrix of concepts"""
    global concepts
    concepts = matutils.loadMatrix(common.matrixFile(prefix + 'LSI_U.mm'))
    return concepts

def head(*ids, **args):
    """print first 'topn' lines (default 6) for each file identified by id. Id may be either matrix column number (integer) or internal database id (string)
    Example: head(13, 22, '05-1955-4-4', topn = 10) prints first 10 lines for the three files"""
    type = args.get('type', 'final')
    if type == 'casopis':
        topn = args.get('topn', 50) # print first 6 lines by default for tex sources
    else:
        topn = args.get('topn', 6) # print first 6 lines by default for OCR
    for id in ids:
        if isinstance(id, basestring):
            id = rdocids[id]
        docid = docids[id]
        if type == 'final':
            path = docid[:docid.rfind('-')] + '/#' + docid[docid.rfind('-') + 1:]
            print id, ':\n' + u''.join(codecs.open(os.path.join('/home/radim/workspace/dml/data/dmlcz/cmj', path, 'fulltext.txt'), 'r', 'utf8').readlines()[:topn])
        elif type == 'casopis':
            print id, ':\n' + u''.join(codecs.open(os.path.join('/home/radim/workspace/dml/data/dmlcz', docid), 'U', 'latin2').readlines()[:topn])

def meta(*ids):
    for id in ids:
        if isinstance(id, basestring):
            id = rdocids[id]
        docid = docids[id]
        path = docid[:docid.rfind('-')] + '/#' + docid[docid.rfind('-') + 1:]
        print id, ':\n' + u''.join(codecs.open(os.path.join('/home/radim/workspace/dml/data/dmlcz/cmj', path, 'meta.xml'), 'r', 'utf8').readlines())

def top(id, mat, topn = 10):
    """Return a list of documents that most closely match document 'id', based on similarity matrix 'mat'.
    'id' may be either a column index (integer) or internal id (string)
    The list consists of (similarity, doc_column, doc_id) tuples.
    Only first 'topn' documents are returned (default 10)."""
    if isinstance(id, basestring):
        id = rdocids[id]
    bests = numpy.argsort(mat[id, :])[::-1][:topn]
    return [(mat[id, best], best, docids[best]) for best in bests]

def sim(id_x, id_y, mat):
    """Return similarity of id_x and id_y documents, based on similarity matrix 'mat'.
    Ids may be either column indices (integers) or internal ids (strings)"""
    if isinstance(id_x, basestring):
        id_x = rdocids[id_x]
    if isinstance(id_y, basestring):
        id_y = rdocids[id_y]
    return mat[id_x, id_y], id_x, id_y, docids[id_x], docids[id_y]

def topmatches(mat, topn = 20):
    """Return the most similar pairs of distinct documents from the whole matrix 'mat'.
    Return only the 'topn' best matches (20 by default)"""
    s = mat.argsort(axis = None)[::-1]
    result = []
    for pos in s:
        x = pos / mat.shape[0]
        y = pos - mat.shape[0] * x
        if x < y:
            result.append((mat.flat[pos], x, y, docids[x], docids[y]))
            if len(result) == topn:
                return result
    return result[:topn]

msim = lambda id_x, id_y: sim(id_x, id_y, m)

lsim = lambda id_x, id_y: sim(id_x, id_y, l)

def mtopmatches(topn = 20):
    """Syntactic sugar for topmatches() run on the TFIDF similarity matrix."""
    return topmatches(m, topn)

def ltopmatches(topn = 20):
    """Syntactic sugar for topmatches() run on the LSI similarity matrix."""
    return topmatches(l, topn)

def mtop(id, topn = 10):
    """Syntactic sugar for top() run on the TFIDF similarity matrix."""
    return top(id, m, topn)

def ltop(id, topn = 10):
    """Syntactic sugar for top() run on the LSI similarity matrix."""    
    return top(id, l, topn)

def printConcept(num, topn = 10):
    c = concepts.T[num]
    most = numpy.abs(c).argsort()[::-1][:topn]
    print ' + '.join(['%f*"%s"' % (c.flat[val], tokenids[val]) for val in most])

def loadArts(dbFile = 'main_cmj.pdl'):
    import ArticleDB
    import Article
    import common
    global db, arts
    db = ArticleDB.ArticleDB(common.dataFile(dbFile), mode = 'open')
    arts = [Article.Article(rec) for rec in db.db if rec['id_int'] in rdocids]

def getArtsWithMsc(mscCode):
    import mscs
    mscDict = dict([(art.id_int, [mscs.niceMSC(msc, prefix = len(mscCode))[0] for msc in art.msc]) for art in arts]) # create mapping of internal id -> list of top category MSCs    
    return [art.id_int for art in arts if mscCode in mscDict[art.id_int]]

def toNLTK(_mat, prefix = 2):
    import mscs
    if scipy.sparse.issparse(_mat):
        mat = _mat.tocsc()
    else:
        mat = _mat
    result = []
    for i in xrange(mat.shape[1]):
        if scipy.sparse.issparse(mat):
            a0 = mat[:, i].toarray()
        else:
            a0 = mat[:, i]
        nnind = a0.nonzero()[0]
        nnvals = a0.take(nnind)
        features = dict(zip(nnind, nnvals))
        id = docids[i]
        okarts = [a for a in arts if a.id_int == id]
        if len(okarts) != 1:
            raise Exception('%i articles with id=%s' % repr(id))
        labels = okarts[0].msc
        if len(labels) < 1:
            raise Exception('no msc for %s' % id)
        result.append((features, mscs.niceMSC(labels[0], prefix = prefix)[0]))
    return result
