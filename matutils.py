import logging
import math
import numpy
import scipy
import scipy.linalg
import scipy.sparse
from itertools import izip

def sqrLength(vec):
    return numpy.dot(vec, vec)

def length(vec):
    return math.sqrt(sqrLength(vec))

def unit(vec):
    return vec / length(vec)

def inner(v1, v2):
    if scipy.sparse.issparse(v1):
        v1 = v1.toarray()
        v1.shape = (v1.size,)
    if scipy.sparse.issparse(v2):
        v2 = v2.toarray()
        v2.shape = (v2.size,)
    return numpy.dot(v1, v2)

def innerabs(v1, v2):
    return numpy.abs(inner(v1, v2))

def _innerabs(v1, v2):
    return numpy.abs(numpy.dot(v1, v2))

def cossimUnit(v1, v2, doAbs = True):
    if doAbs:
        return numpy.abs(inner(v1, v2))  # FIXME abs? better rescale full [-1,1] -> [0,1]?
    else:
        return numpy.abs(inner(v1, v2))  # FIXME abs? better rescale full [-1,1] -> [0,1]?

def cossim(v1, v2, doAbs = True):
    if scipy.sparse.issparse(v1):
        v1 = v1.toarray()
        v1.shape = (v1.size,)
    if scipy.sparse.issparse(v2):
        v2 = v2.toarray()
        v2.shape = (v2.size,)
    return cossimUnit(unit(v1), unit(v2), doAbs)  # FIXME abs? better rescale full [-1,1] -> [0,1]?

def lengthSqr_sparse(vec):
    if isinstance(vec, scipy.sparse.lil_matrix):
        return sum([numpy.dot(vec.data[i], vec.data[i]) for i in xrange(vec.shape[0])])
    else:
        return numpy.dot(vec.data, vec.data)

def length_sparse(vec):
    return numpy.sqrt(lengthSqr_sparse(vec))

def unit_sparse(vec): # has to be in lil or csr sparse matrix format
    ilen = 1.0 / length_sparse(vec)
    return vec * ilen

def jacsim(v1, v2):
    i = inner(v1, v2)
    return i / (length(v1) * length(v2) - i)

def cosdist(v1, v2):
    c = cossim(v1, v2, doAbs = True)
    return (1.0  - c)

def innerdist(v1, v2):
    return 1.0 - abs(inner(v1, v2))

def euclidSqr(v1, v2):
    return numpy.sum((v2 - v1)**2)

def euclid(v1, v2):
    return math.sqrt(numpy.sum((v2 - v1)**2))

def euclidsim(v1, v2):
    return 1.0 / (1.0 + euclid(v1, v2))

def euclidsime(v1, v2):
    return numpy.power(numpy.e, -euclid(v1, v2)/8.0)

def symMatrix(vecs, symFn):
    """applies symFn on each pair from vecs, returns matrix of results (like numpy.fromfunction).
    symFn is a symmetric function taking 2 points of the same dimensionality
    !documents are rows!
    """
    if scipy.sparse.issparse(vecs):
        logging.debug('converting sparse %ix%i matrix to csr format' % vecs.shape)
        vecs = vecs.tocsr()
        logging.debug('multiplying mat * mat.T')
        result = (vecs * vecs.T)
        logging.debug('converting result to dense matrix')
        result = result.todense()
        return result
    else:
        ln = vecs.shape[0]
        result = numpy.empty((ln, ln), numpy.float32)
        for i in xrange(ln):
            for j in xrange(i, ln):
                result[i, j] = result[j, i] = symFn(vecs[i], vecs[j])
    return result

def euclidMatrix(points): # OPT better than direct distMatrix(points, euclid)?
    import numpy.matlib    
    numPoints = len(points)
    distMat = numpy.sqrt(numpy.sum((numpy.matlib.repmat(points, numPoints, 1) - numpy.matlib.repeat(points, numPoints, axis = 0))**2, axis = 1))
    return distMat.reshape((numPoints, numPoints))
    
def transform(data, transformation):
    return numpy.dot(data, transformation)

def normalized(mat):
    return numpy.column_stack([unit(vec) for vec in mat.T])

def normalized_sparse(_mat): # FIXME depends on scipy.sparse implementation details in an ugly way, but anything else is WAAAAy too slow..
    """normalize columns of a sparse matrix to unit length"""
    mat = _mat.copy().tocsc()
    for col in xrange(mat.shape[1]):
        dat = mat.data[mat.indptr[col]:mat.indptr[col+1]]
        len = numpy.dot(dat, dat)
        if len != 1:
            ilen = 1.0 / numpy.sqrt(len)
            mat.data[mat.indptr[col]:mat.indptr[col+1]] *= ilen
    return mat

def normalizeColumns(mat, inplace = False): # FIXME depends on scipy.sparse implementation details in an ugly way, but anything else is WAAAAy too slow..
    """normalize columns of a sparse matrix to unit length"""
    if inplace:
        if not isinstance(mat, scipy.sparse.csc_matrix):
            raise NotImplementedError("can only normalize CSC matrices inplace")
    else:
        # for not inplace normalization, just copy the matrix and convert it to csc
        mat = mat.copy().tocsc()
    for col in xrange(mat.shape[1]):
        dat = mat.data[mat.indptr[col]:mat.indptr[col+1]]
        len = numpy.dot(dat, dat)
        if len != 1:
            ilen = 1.0 / numpy.sqrt(len)
            mat.data[mat.indptr[col]:mat.indptr[col+1]] *= ilen
    return mat

def normalizeSparseRows(mat):
    return normalized_sparse(mat.T).T

def saveMatrix(mat, filename, sparse = False, format = 'MTX'):
    """save matrix to file. formats are:
        CLUTO - documents are rows
            dense is 1. line '#rows #cols', other lines sequences of exactly #cols floats
            sparse is 1. line '#rows #cols #nonzeros', other lines sequences of 'int float' document attribute pairs, with indexes starting at 1
        SVDLIB - documents are columns
            dense is the same as CLUTO
            sparse first line as CLUTO; next lines = row_nnz + int float pairs, starting at index 0
        MM = MTX = MATRIXMARKET - documents are columns
            1. line is typecode: '%%MatrixMarket matrix coordinate real general'
            2. line is: '#rows #cols #nonzeros'
            next line is one element per line: 'm n mat[m,n]', where indices are 1-based (as if A[1,1] was the first element)
        BMAT - binary format, documents are columns
            bytes 0-3 = n = number of rows
            4-7 = m = number of columns (documents)
            rest are m*n matrix elements, stored per column. size depends on mat.dtype.
        """
    logging.info('saving %ix%i matrix to %s' % (mat.shape[0], mat.shape[1], filename))
    try:
        format = format.upper()
        if format == 'BMAT':
            import scipy.sparse
            if scipy.sparse.issparse(mat):
                raise Exception("sparse binary not implemented")
            f = open(filename, 'wb')
            header = numpy.asarray(mat.shape, dtype = numpy.int32).tostring()
            f.write(header) # store header
            for vec in mat.T:
                f.write(vec.tostring()) # store vectors sequentially
            f.close()
            return

        f = open(filename, 'w')
        
        if format == 'MTX' or format == 'MM' or format == 'MATRIXMARKET':
            import scipy.io.mmio
            if sparse:
                if isinstance(mat, scipy.sparse.spmatrix):
                    tmp = mat
                else:
                    logging.debug('converting to sparse')
                    tmp = scipy.sparse.csr_matrix(mat)
            else:
                tmp = numpy.array(mat)
            logging.debug('writing file')
            scipy.io.mmio.mmwrite(f, tmp)
            return
        elif format == 'CLUTO':
            if scipy.sparse.issparse(mat):
                raise Exception("not implemented yet :F")
            else:
                if sparse:
                    nnz = mat.nonzero()
                    f.write("%i %i %i\n" % (mat.shape[1], mat.shape[0], len(nnz[0])))
                    for v in mat.T:
                        s = ' '.join([str(i + 1) + " " + str(v[i]) for i in numpy.where(v != 0)[0]]) # FIXME better test against numpy.allclose?
                        f.write(s + "\n")
                else:
                    f.write("%i %i\n" % (mat.shape[1], mat.shape[0]))
                    for v in mat.T:
                        s = ' '.join([`a` for a in v])
                        f.write(s + "\n")
        elif format == 'SVDLIB':
            if scipy.sparse.issparse(mat):
                if sparse:
                    f.write("%i %i %i\n" % (mat.shape[0], mat.shape[1], mat.getnnz()))
                    for i in xrange(mat.shape[0]):
                        v = mat[i, :].toarray()[0]
                        f.write("%i\n" % len(v.nonzero()[0]))
                        s = ' '.join([str(i) + " " + str(v[i]) for i in numpy.where(v != 0)[0]])
                        f.write(s + "\n")
                else:
                    f.write("%i %i\n" % (mat.shape[0], mat.shape[1]))
                    for i in xrange(mat.shape[0]):
                        v = mat[i, :].toarray()[0]
                        s = ' '.join([`a` for a in v])
                        f.write(s + "\n")
            else:
                if sparse:
                    nnz = mat.nonzero()
                    f.write("%i %i %i\n" % (mat.shape[0], mat.shape[1], len(nnz[0])))
                    for v in mat:
                        f.write("%i\n" % len(v.nonzero()[0]))
                        s = ' '.join([str(i) + " " + str(v[i]) for i in numpy.where(v != 0)[0]])
                        f.write(s + "\n")
                else:
                    f.write("%i %i\n" % (mat.shape[0], mat.shape[1]))
                    for v in mat:
                        s = ' '.join([`a` for a in v])
                        f.write(s + "\n")
        else:
            raise Exception("unrecognized matrix format %s" % format)
        f.close()
        logging.debug('done saving %s' % (filename))
    except Exception, detail:
        logging.error('failed to save into %s: %s' % (filename, str(detail)))
        raise

def loadMatrix(filename, format = 'MTX', bintype = numpy.float32):
    """load matrix"""

    format = format.upper()
    try:
        if format.startswith('BMAT'):
            f = open(filename, 'rb')
            shp = numpy.fromstring(f.read(8), dtype = numpy.int32)
            if len(format) > 4: # only return matrix shape, do not load data
                return tuple(shp)
            result = numpy.fromstring(f.read(), dtype = bintype)
            if result.size != shp[0] * shp[1]:
                raise Exception("size does not match: %ix%i header vs %i elements (wrong data type?)" % (shp[0], shp[1], len(result)))
            result.shape = shp[::-1]
            return result.T

        f = open(filename, 'r')        
        if format == 'MTX' or format == 'MM' or format == 'MATRIXMARKET':
            import scipy.io.mmio
            result = scipy.io.mmio.mmread(f)
        elif format == 'CLUTO':
            first = f.readline()
            params = first.split()
            if len(params) == 2:
                sparse = False
                shape = (int(params[1]), int(params[0]))
            elif len(params) == 3:
                sparse = True
                shape = (int(params[1]), int(params[0]))
                nnz = int(params[2])
            else:
                logging.error("cannot interpret %s" % filename)
                return None
            
            result = numpy.zeros(shape, float)
            i = 0
            for line in f:
                vals = line.split()
                j = 0
                while j < len(vals):
                    if sparse:
                        result[int(vals[j]) - 1, i] = float(vals[j + 1])
                        j += 1
                    else:
                        result[j, i] = float(vals[j])
                    j += 1
                
                i += 1
    except IOError, detail:
        logging.error('failed to load from %s: %s' % (filename, str(detail)))
        return None
    finally:
        f.close()
    return result

def eigenvectors(mat, matB = None):
    """return eigenvalues and eigenvectors (=matrix columns) in sorted order."""
    ZERO = 1e-6 # HAX
    e_val, e_vec = scipy.linalg.eig(mat, matB)
    e_val = e_val.astype(float)
    e_vec = e_vec.astype(float)
    e_sorted = numpy.argsort(e_val)[::-1] # sort in descending eigenvalue order
    e_nz = e_sorted # = e_sorted.take(numpy.where(numpy.abs(e_val.take(e_sorted, axis = 0)) > ZERO)[0], axis = 0)
    e_val = numpy.take(e_val, e_nz)
    e_vec = numpy.take(e_vec, e_nz, axis = 1)
    return e_val, e_vec

def eigenvectorsh(mat, eigvals_only = False):
    """return eigenvalues and eigenvectors (=hermitian matrix columns) in sorted order.
    set eigvals_only to only return eigenvalues (no eigenvectors)."""
    ZERO = 1e-6 # HAX
    if eigvals_only:
        e_val = scipy.linalg.eigh(mat, eigvals_only = True)
    else:
        e_val, e_vec = scipy.linalg.eigh(mat, eigvals_only = False)
    e_val = e_val.astype(float)
    e_sorted = numpy.argsort(e_val)[::-1] # sort in descending eigenvalue order
    e_nz = e_sorted # = e_sorted.take(numpy.where(numpy.abs(e_val.take(e_sorted, axis = 0)) > ZERO)[0], axis = 0)
    e_val = numpy.take(e_val, e_nz)
    if not eigvals_only:
        e_vec = e_vec.astype(float)
        e_vec = numpy.take(e_vec, e_nz, axis = 1)
        return e_val, e_vec
    else:
        return e_val

def mse(a, b): # MSE of two matrices
    c = a - b
    result = 1.0 * numpy.sum(numpy.multiply(c, c)) / c.size
    return result

def rmse(data1, data2):
    return numpy.sqrt(mse(data1, data2))

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    
    b = numpy.array([1,2,3, 4])
    c = numpy.array([b, b])
    print repr(c)
    saveMatrix(c, '/home/radim/workspace/plagiarism/data/tstmat2', format = 'bmat')
    
    d = loadMatrix('/home/radim/workspace/plagiarism/data/tstmat2', format = 'bmat', bintype = numpy.float32)
    print d


class MmWriter(object):
    def __init__(self, fname):
        self.fname = fname
    
    @staticmethod
    def determineNnz(corpus):
        logging.info("calculating matrix shape and density")
        numDocs = numTerms = numNnz = 0
        for docNo, bow in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: at document %i/%i" % 
                             (docNo, len(corpus)))
            if len(bow) > 0:
                numDocs = max(numDocs, docNo + 1)
                numTerms = max(wordId for wordId, _ in bow) + 1
                numNnz += len(bow)
        
        logging.info("BOW of %ix%i matrix, density=%.3f%% (%i/%i)" % 
                     (numDocs, numTerms,
                      100.0 * numNnz / (numDocs * numTerms),
                      numNnz,
                      numDocs * numTerms))
        return numDocs, numTerms, numNnz
    

    def writeHeaders(self, numDocs, numTerms, numNnz):
        logging.info("saving sparse %sx%s matrix with %i non-zero entries to %s" %
                     (numDocs, numTerms, numNnz, self.fname))
        self.fout = open(self.fname, 'w')
        self.fout.write('%%matrixmarket matrix coordinate real general\n')
        self.fout.write('%i %i %i\n' % (numDocs, numTerms, numNnz))
        self.lastDocNo = -1
    
    def __del__(self):
        """
        Automatic destructor which closes the underlying file. 
        
        There must be no circular references contained in the object for __del__
        to work! Closing the file explicitly via the close() method is preferred
        and safer.
        """
        self.close()
    
    def close(self):
        logging.debug("closing %s" % self.fname)
        self.fout.close()
        
    def writeBowVector(self, docNo, pairs):
        """
        Write a single bag-of-words vector to the file.
        """
        assert self.lastDocNo < docNo, "documents %i and %i not in sequential order!" % (self.lastDocNo, docNo)
        for termId, weight in sorted(pairs): # write term ids in sorted order
            if weight != 0.0:
                self.fout.write("%i %i %f\n" % (docNo + 1, termId + 1, weight)) # +1 because MM format starts counting from 1
        self.lastDocNo = docNo
    
    def writeCorpus(self, corpus):
        """
        Save bag-of-words representation of an entire corpus to disk.
        
        Note that the documents are processed one at a time, so the whole corpus 
        is allowed to be larger than the available RAM.
        """
        logging.info("saving %i BOW vectors to %s" % (len(corpus), self.fname))
        for docNo, bow in enumerate(corpus):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: saving document %i/%i" % 
                             (docNo, len(corpus)))
            self.writeBowVector(docNo, bow)
#endclass MmWriter


class MmReader(object):
    """
    Wrap a corpus represented as term-document matrix on disk in matrix-market 
    format, and present it as an object which supports iteration over documents. 
    A document = list of (word, weight) 2-tuples. This iterable format is used 
    internally in LDA inference.
    
    Note that the file is read into memory one document at a time, not whole 
    corpus at once. This allows for representing corpora which do not wholly fit 
    in RAM.
    """
    def __init__(self, fname):
        """
        Initialize the corpus reader. The fname is a path to a file on local 
        filesystem, which is expected to be sparse (coordinate) matrix
        market format. Documents are assumed to be rows of the matrix -- if 
        documents are columns, save the matrix transposed.
        """
        logging.info("initializing corpus reader from %s" % fname)
        self.fname = fname
        fin = open(fname)
        header = fin.next()
        if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
            raise ValueError("File %s not in Matrix Market format with coordinate real general" % fname)
        self.noRows = self.noCols = self.noElements = 0
        for lineNo, line in enumerate(fin):
            if not line.startswith('%'):
                self.noRows, self.noCols, self.noElements = map(int, line.split())
                break
        logging.info("accepted corpus with %i documents, %i terms, %i non-zero entries" %
                     (self.noRows, self.noCols, self.noElements))
    
    def __len__(self):
        return self.noRows
        
    def __iter__(self):
        fin = open(self.fname)
        
        # skip headers
        for line in fin:
            if line.startswith('%'):
                continue
            break
        
        prevId = None
        for line in fin:
            docId, termId, val = line.split()
            if docId != prevId:
                if prevId is not None:
                    yield prevId, document
                prevId = docId
                document = []
            # add (termId, weight) pair to the document
            document.append((int(termId) - 1, float(val),)) # -1 because matrix market indexes are 1-based => convert to 0-based
        if prevId is not None: # handle the last document, as a special case
            yield prevId, document
#endclass MmWriter


class MmCorpus(MmReader):
    def __iter__(self):
        for docId, doc in super(MmCorpus, self).__iter__():
            yield doc # get rid of docId, return the document only
#endclass MmCorpus
