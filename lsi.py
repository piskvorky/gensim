import logging
import math
import scipy
import numpy
import matutils
import sparseSVD

class LSIEngine:
    def __init__(self, docmatrix, cover = 0.5, useSparse = True):
        """cover: minimum data variance to keep: value from <0.0,1.0> is variance percentage; >1 is absolute number of factors"""
        logging.info("computing SVD of a %ix%i matrix, sparse = %s" % (docmatrix.shape[0], docmatrix.shape[1], useSparse))
        
        if useSparse:
            if cover <= 1.0:
                raise RuntimeError, "ratio SVD coverage not supported with sparse matrices!"
            result = sparseSVD.doSVD(docmatrix, num = cover)
        else:
            assert False, "SVD on dense matrices not supported (why do you need that anyway)?"
        
        if result == None:
            logging.critical("LSIEngine init failed")
            return
        
        U, S, VT = result
        if cover <= 1.0: # FIXME quadratic better? (% of length of S as a vector)
            eigencum = numpy.cumsum(S) / numpy.sum(S)
            self.maxFactors = 1 + len(numpy.compress(eigencum < cover, eigencum))
        else:
            self.maxFactors = min(int(cover), S.shape[0])

        self.U = U[:, :self.maxFactors]
        self.S1 = numpy.diag(1.0 / S[:self.maxFactors])
        self.S = numpy.diag(S[:self.maxFactors])
        self.VT = VT[:self.maxFactors, :]
        
        if cover <= 1.0: # print some stats; dense only
            coverage = 100.0 * numpy.sum(S[ : self.maxFactors]) / numpy.sum(S)
            logging.info("keeping %i factors out of %i; %.3f%% coverage" % (self.maxFactors, len(S), coverage))
            _docmatrix = numpy.dot(self.U, numpy.dot(self.S, self.VT))
            logging.debug("MSE between original and embedded matrix = %f" % matutils.mse(_docmatrix, docmatrix))

    def vis2d(self, idDocs, idTerms):
        import plotutils
        if (idDocs != None):
            plotutils.plotPnts(self.getLatentDocs(2).T, idDocs, newFigure = True)
        if (idTerms != None):
            plotutils.plotPnts(self.getLatentTerms(2), idTerms, marker = '^', color = 'g', trotation = 'vertical', newFigure = False)

    def getLatentDocs(self, f = -1):
        """return original documents projected into lsi space; return as matrix where document vectors are columns"""
        if (f < 0):
            f = self.maxFactors
        else:
            f = min(f, self.maxFactors)
        result = self.VT[:f, :].copy()
        return result

    def getLatentTerms(self, f = -1):
        """return original terms projected into lsi space; return as matrix where term vectors are rows"""
        if (f < 0):
            f = self.maxFactors
        else:
            f = min(f, self.maxFactors)
        result = numpy.dot(self.U[:, :f], self.S1[:f, :f]) # same as calling project() for each term separately
        return result

    def project(self, t = None, d = None, f = -1):
        """project term&doc vectors into the topic space
        t : document = terms vector (same order as in __init__)
        d : term = documents vector (same order as in __init__)
        f : number of factors to use; negative = use all available

        projected pseudo-query is formed by q = t.T * U(f) * S(f)^-1 + d.T * V(f)
        """
        if (f < 0):
            f = self.maxFactors
        else:
            f = min(f, self.maxFactors)
        q = numpy.zeros((f), dtype = float)
        if (t != None):
            q += numpy.dot(t, self.getDocTransform(f))
        if (d != None):
            q += numpy.dot(d, self.VT[:f, :])
        return q

    def getDocTransform(self, f = -1):
        if (f < 0):
            f = self.maxFactors
        else:
            f = min(f, self.maxFactors)
        return numpy.dot(self.U[:, :f], self.S1[:f, :f]) # return #terms x f matrix
    
    def foldin(self, docs):
        """
        fold in new documents into the collection; no recomputation is done (all old weights remain!)
        documents are represented as either standard collection matrix (ie array[term, doc]) or a single document vector
        """
        docs = numpy.array(docs)
        if docs.ndim == 1:
            docs = numpy.atleast_2d(docs)
        else:
            docs = numpy.transpose(docs)
        proj = numpy.transpose(numpy.array([self.project(doc) for doc in docs]))
        newVT = numpy.hstack([self.VT, proj])
        self.VT = newVT

if __name__ == '__main__':
    import pylab
    import plotutils
    logging.basicConfig(level = logging.DEBUG)
    
    d = [[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
         [1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1],
         [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]]
    idDocs = [u'doc1', 'doc2', 'doc3']
    idTerms = ['a','arrived','damaged','delivery','fire','gold','in','of','shipment','silver','truck']
    q  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    mat = numpy.transpose(numpy.array(d))
    lsi = LSIEngine(docmatrix = mat, cover = 2)
    print 'U: ', lsi.U
    print
    print 'VT: ', lsi.VT
    print
    nq = lsi.project(t = q, f = 2)
    lsi.vis2d(idDocs, idTerms)
    plotutils.plotPnt(nq, txt = 'query', color = 'r', tcolor = 'r')
    print 'proj q: ', nq
    nq = lsi.project(t = mat[:, 0], f = 2)
    plotutils.plotPnt(nq, 1, color = 'r', tcolor = 'r')
    nq = lsi.project(t = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], f = 2)
    plotutils.plotPnt(nq, 'query_a', color = 'r', tcolor = 'r')
    lsi.foldin(mat)
    idDocs.extend([doc + '_2' for doc in idDocs])
    lsi.foldin(q)
    idDocs.append('query')
    lsi.vis2d(idDocs, idTerms)
    symmat = matutils.symMatrix(lsi.getLatentDocs(2).T, matutils.cossim)
    print 'cossim matrix:', symmat
    plotutils.showMatrix(symmat)
    pylab.show() # display pylab figures
