#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for Latent Semantic Indexing.

It actually contains several algorithms for decomposition of extremely large 
corpora, a combination of which effectively and transparently allows building LSI
models for:

  * corpora much larger than RAM (only constant memory needed, independent of 
    corpus size)
  * corpora that are streamed (documents can only be accessed sequentially, not 
    random-accessed)
  * corpora that cannot even be temporarily stored, each document can only be 
    seen once and processed immediately (one-pass algorithm)
  * distributed computing for ultra large corpora, making use of a cluster of 
    machines

Performance on English Wikipedia (2G corpus positions, 3.2M documents, 200K features, 
0.5G non-zero entries in the final TF-IDF matrix), requesting top 400 LSI factors:


  ------------------------------------------------------------------------
  |                                           serial      distributed    |
  | one-pass update algo (chunks=factors)     215h        38h            |
  | one-pass merge algo (chunks=20K docs)     14h         4h             |
  | two-pass randomized algo (chunks=20K)     2.5h        N/A            |
  ------------------------------------------------------------------------

serial = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec
distributed = cluster of six logical nodes on four physical machines, each with dual core Xeon 2.0GHz, 4GB RAM, ATLAS

"""


import logging
import itertools

import numpy
import scipy.sparse
from scipy.sparse import sparsetools

# scipy is not a stable package yet, locations change, so try to work
# around differences (currently only concerns location of 'triu' in scipy 0.7 vs. 0.8)
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs, find_best_lapack_type
try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu


from gensim import interfaces, matutils, utils


logger = logging.getLogger('lsimodel')
logger.setLevel(logging.INFO)



def clipSpectrum(s, k, discard = 0.001):
    """
    Given eigenvalues `s`, return how many factors should be kept to avoid
    storing spurious (tiny, numerically instable) values. 
    
    This will ignore the tail of the spectrum with relative combined mass < min(`discard`, 1/k).
    
    The returned value is clipped against `k` (= at most `k` factors).
    """
    # compute relative contribution of eigenvalues towards the energy spectrum
    rel_spectrum = numpy.abs(1.0 - numpy.cumsum(s / numpy.sum(s)))
    # ignore the last `discard` (or 1/k, whichever is smaller) of the spectrum
    small = 1 + len(numpy.where(rel_spectrum > min(discard, 1.0 / k))[0]) 
    k = min(k, small) # clip against k
    logger.info("keeping %i factors (discarding %.3f%% of energy spectrum)" %
                (k, 100 * rel_spectrum[k - 1]))
    return k


class Projection(utils.SaveLoad):
    def __init__(self, m, k, docs = None, algo = 'onepass', chunks = None):
        """
        Construct the (U, S) projection from a corpus `docs`. 
        
        This is the class taking care of 'core math'; interfacing with corpora, 
        chunking, training etc. is done through the LsiModel class.
        
        `algo` is currently one of:
        
          * 'onepass'; only a single pass over `docs` is needed
          * 'twopass'; multiple passes over the input allowed => can use a 
            faster algorithm.
        """
        self.m, self.k = m, k
        if docs is not None:
            # base case decomposition: given a job `docs`, compute its decomposition in-core
            # results of several base case decompositions can be merged via `self.merge()`
            if algo == 'twopass':
                assert utils.isCorpus(docs)
                self.u, self.s = stochasticSvd(docs, k, chunks = chunks, num_terms = m, extra_dims = 0)
            elif algo == 'onepass':
                if utils.isCorpus(docs):
                    docs = matutils.corpus2csc(m, docs)
                if docs.shape[1] <= max(k, 100):
                    # For sufficiently small chunk size, update directly like `svd(now, docs)` 
                    # instead of `svd(now, svd(docs))`.
                    # This improves accuracy and is also faster for small chunks, because
                    # we need to perform one less svd.
                    # On larger chunks this doesn't work because we quickly run out of memory.
                    self.u = docs
                    self.s = None
                else:
                    try:
                        import sparsesvd
                    except ImportError:
                        raise ImportError("for LSA, the `sparsesvd` module is needed but not found; run `easy_install sparsesvd`")
                    logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
                    ut, s, vt = sparsesvd.sparsesvd(docs, k + 30) # ask for extra factors, because for some reason SVDLIBC sometimes returns fewer factors than requested
                    u = ut.T
                    del ut, vt
                    k = clipSpectrum(s ** 2, self.k)
                    self.u, self.s = u[:, :k], s[:k]
            else:
                raise NotImplementedError("unknown decomposition algorithm: '%s'" % algo)
        else:
            self.u, self.s = None, None
    
    
    def empty_like(self):
        return Projection(self.m, self.k)
    

    def merge(self, other, decay = 1.0):
        """
        Merge this Projection with another. 
        
        The content of `other` is destroyed in the process, so pass this function a 
        copy of `other` if you need it further.
        """
        if other.u is None:
            # the other projection is empty => do nothing
            return
        if self.u is None:
            # we are empty => result of merge is the other projection, whatever it is
            if other.s is None:
                # other.u contains a direct document chunk, not svd => perform svd
                docs = other.u
                assert scipy.sparse.issparse(docs)
                if self.m * self.k < 10000:
                    # SVDLIBC gives spurious results for small matrices.. run full
                    # LAPACK on them instead
                    logger.info("computing dense SVD of %s matrix" % str(docs.shape))
                    u, s, vt = numpy.linalg.svd(docs.todense(), full_matrices = False)
                else:
                    try:
                        import sparsesvd
                    except ImportError:
                        raise ImportError("for LSA, the `sparsesvd` module is needed but not found; run `easy_install sparsesvd`")
                    logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
                    ut, s, vt = sparsesvd.sparsesvd(docs, self.k + 30) # ask for a few extra factors, because for some reason SVDLIBC sometimes returns fewer factors than requested
                    u = ut.T
                    del ut
                del vt
                k = clipSpectrum(s ** 2, self.k)
                self.u = u[:, :k].copy('F')
                self.s = s[:k]
            else:
                self.u = other.u.copy('F')
                self.s = other.s.copy()
            return
        if self.m != other.m:
            raise ValueError("vector space mismatch: update has %s features, expected %s" %
                             (other.m, self.m))
        logger.info("merging projections: %s + %s" % (str(self.u.shape), str(other.u.shape)))
        m, n1, n2 = self.u.shape[0], self.u.shape[1], other.u.shape[1]
        if other.s is None:
            other.u = other.u.todense()
            other.s = 1.0 # broadcasting will promote this to eye(n2) where needed
        # TODO Maybe keep the bases as elementary reflectors, without 
        # forming explicit matrices with ORGQR.
        # The only operation we ever need is basis^T*basis ond basis*component.
        # But how to do that in scipy? And is it fast(er)?
        
        # find component of u2 orthogonal to u1
        # IMPORTANT: keep matrices in suitable order for matrix products; failing to do so gives 8x lower performance :(
        self.u = numpy.asfortranarray(self.u) # does nothing if input already fortran-order array
        other.u = numpy.asfortranarray(other.u)
        gemm, = get_blas_funcs(('gemm',), (self.u,))
        logger.debug("constructing orthogonal component")
        c = gemm(1.0, self.u, other.u, trans_a = True)
        gemm(-1.0, self.u, c, beta = 1.0, c = other.u, overwrite_c = True)
        
        # perform q, r = QR(component); code hacked out of scipy.linalg.qr
        logger.debug("computing QR of %s dense matrix" % str(other.u.shape))
        geqrf, = get_lapack_funcs(('geqrf',), (other.u,))
        qr, tau, work, info = geqrf(other.u, lwork = -1, overwrite_a = True) # sometimes segfaults with overwrite_a=True...?
        qr, tau, work, info = geqrf(other.u, lwork = work[0], overwrite_a = True) # sometimes segfaults with overwrite_a=True...?
        del other.u
        assert info >= 0
        r = triu(qr[:n2, :n2])
        if m < n2: # rare case, #features < #topics
            qr = qr[:, :m] # retains fortran order
        gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
        q, work, info = gorgqr(qr, tau, lwork = -1, overwrite_a = True)
        q, work, info = gorgqr(qr, tau, lwork = work[0], overwrite_a = True)
        assert info >= 0, "qr failed"
        assert q.flags.f_contiguous
        
        # find the rotation that diagonalizes r
        k = numpy.bmat([[numpy.diag(decay * self.s), c * other.s], [matutils.pad(numpy.matrix([]).reshape(0, 0), min(m, n2), n1), r * other.s]])
        logger.debug("computing SVD of %s dense matrix" % str(k.shape))
        try:
            # in numpy < 1.1.0, running SVD sometimes results in "LinAlgError: SVD did not converge'.
            # for these early versions of numpy, catch the error and try to compute
            # SVD again, but over k*k^T.
            # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg07224.html and
            # bug ticket http://projects.scipy.org/numpy/ticket/706
            u_k, s_k, _ = numpy.linalg.svd(k, full_matrices = False) # TODO *ugly overkill*!! only need first self.k SVD factors... but there is no LAPACK wrapper for partial svd/eigendecomp in numpy :(
        except numpy.linalg.LinAlgError:
            logging.error("SVD(A) failed; trying SVD(A * A^T)")
            u_k, s_k, _ = numpy.linalg.svd(numpy.dot(k, k.T), full_matrices = False) # if this fails too, give up
            s_k = numpy.sqrt(s_k)
        
        k = clipSpectrum(s_k ** 2, self.k)
        u_k, s_k = u_k[:, :k], s_k[:k]
        
        # update & rotate current basis U
        logger.debug("updating orthonormal basis U")
        self.u = gemm(1.0, self.u, u_k[:n1]) # TODO temporarily creates an extra (m,k) dense array in memory. find a way to avoid this!
        gemm(1.0, q, u_k[n1:], beta = 1.0, c = self.u, overwrite_c = True) # u = [u,u']*u_k
        self.s = s_k
#        diff = numpy.dot(self.u.T, self.u) - numpy.eye(self.u.shape[1])
#        logger.info('orth error after=%f' % numpy.sum(diff * diff))
#endclass Projection



class LsiModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model for Latent 
    Semantic Indexing (also known as Latent Semantic Analysis).
    
    The main methods are:
    
    1. constructor, which initializes the projection into latent topics space,
    2. the ``[]`` method, which returns representation of any input document in the 
       latent space,
    3. the `addDocuments()` method, which allows for incrementally updating the model with new documents. 

    Model persistency is achieved via its load/save methods.
    
    """
    def __init__(self, corpus = None, numTopics = 200, id2word = None, chunks = 20000, 
                 decay = 1.0, distributed = False, onepass = True):
        """
        `numTopics` is the number of requested factors (latent dimensions). 
        
        After the model has been trained, you can estimate topics for an
        arbitrary, unseen document, using the ``topics = self[document]`` dictionary 
        notation. You can also add new training documents, with ``self.addDocuments``,
        so that training can be stopped and resumed at any time, and the
        LSI transformation is available at any point.

        If you specify a `corpus`, it will be used to train the model. See the 
        method `addDocuments` for a description of the `chunks` and `decay` parameters.
        
        If your document stream is one-pass only (the stream cannot be repeated),
        turn on `onepass` to force a single pass SVD algorithm (slower).

        Turn on `distributed` to enforce distributed computing (only makes sense 
        if `onepass` is set at the same time, too).
        
        Example:
        
        >>> lsi = LsiModel(corpus, numTopics = 10)
        >>> print lsi[doc_tfidf]
        >>> lsi.addDocuments(corpus2) # update LSI on additional documents
        >>> print lsi[doc_tfidf]
        
        """
        self.id2word = id2word
        self.numTopics = int(numTopics)
        self.chunks = int(chunks)
        self.decay = float(decay)
        self.onepass = onepass
        
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')
        
        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dictFromCorpus(corpus)
            self.numTerms = len(self.id2word)
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())
        
        self.docs_processed = 0
        self.projection = Projection(self.numTerms, self.numTopics)
        
        if not distributed:
            logger.info("using serial LSI version on this node")
            self.dispatcher = None
        else:
            if not onepass:
                raise NotImplementedError("distributed randomized LSA not implemented yet; "
                                          "run either distributed one-pass, or serial randomized.")
            try:
                import Pyro
                ns = Pyro.naming.locateNS()
                dispatcher = Pyro.core.Proxy('PYRONAME:gensim.dispatcher@%s' % ns._pyroUri.location)
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word = self.id2word, numTopics = numTopics, 
                                      chunks = chunks, decay = decay,
                                      distributed = False, onepass = onepass)
                self.dispatcher = dispatcher
                logger.info("using distributed version with %i workers" % len(dispatcher.getworkers()))
            except Exception, err:
                # distributed version was specifically requested, so this is an error state
                logger.error("failed to initialize distributed LSI (%s)" % err)
                raise RuntimeError("failed to initialize distributed LSI (%s)" % err)

        if corpus is not None:
            self.addDocuments(corpus)
    
    
    def addDocuments(self, corpus, chunks = None, decay = None):
        """
        Update singular value decomposition to take into account a new 
        corpus of documents.
        
        Training proceeds in chunks of `chunks` documents at a time. The size of 
        `chunks` is a tradeoff between increased speed (bigger `chunks`) 
        vs. lower memory footprint (smaller `chunks`). If the distributed mode 
        is on, each chunk is sent to a different worker/computer.

        Setting `decay` < 1.0 causes re-orientation towards new data trends in the 
        input document stream, by giving less emphasis to old observations. This allows
        SVD to gradually "forget" old observations (documents) and give more 
        preference to new ones.
        """
        logger.info("updating SVD with new documents")
        
        # get computation parameters; if not specified, use the ones from constructor
        if chunks is None:
            chunks = self.chunks
        if decay is None:
            decay = self.decay
        
        if utils.isCorpus(corpus):
            if not self.onepass:
                # we are allowed multiple passes over input => use a faster, randomized algo
                update = Projection(self.numTerms, self.numTopics, corpus, algo = 'twopass', chunks = chunks)
                self.projection.merge(update, decay = decay)
            else:
                # the one-pass algo
                chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
                doc_no = 0
                for chunk_no, (key, group) in enumerate(chunker):
                    # construct the job as a sparse matrix, to minimize memory overhead
                    # definitely avoid materializing it as a dense matrix!
                    job = matutils.corpus2csc(self.numTerms, (doc for _, doc in group))
                    doc_no += job.shape[1]
                    if self.dispatcher:
                        # distributed version: add this job to the job queue, so workers can work on it
                        logger.debug("creating job #%i" % chunk_no)
                        self.dispatcher.putjob(job) # put job into queue; this will eventually block, because the queue has a small finite size
                        del job
                        logger.info("dispatched documents up to #%s" % doc_no)
                    else:
                        # serial version, there is only one "worker" (myself) => process the job directly
                        update = Projection(self.numTerms, self.numTopics, job)
                        del job
                        self.projection.merge(update, decay = decay)
                        del update
                        logger.info("processed documents up to #%s" % doc_no)
                        self.printTopics(5) # TODO see if printDebug works and remove one of these..
                
                if self.dispatcher:
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    import time
                    while self.dispatcher.jobsdone() <= chunk_no:
                        time.sleep(0.5) # check every half a second
                    logger.info("all jobs finished, downloading final projection")
                    del self.projection
                    self.projection = self.dispatcher.getstate()
                    logger.info("decomposition complete")
        else:
            assert not self.dispatcher, "must be in serial mode to receive jobs"
            assert isinstance(corpus, scipy.sparse.csc_matrix)
            assert self.onepass, "distributed two-pass algo not supported yet"
            update = Projection(self.numTerms, self.numTopics, corpus)
            self.projection.merge(update, decay = decay)
            logger.info("processed sparse job of %i documents" % (corpus.shape[1]))

        self.printTopics(5)

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s, decay=%s, chunks=%s)" % \
                (self.numTerms, self.numTopics, self.decay, self.chunks)


    def __getitem__(self, bow, scaled = False):
        """
        Return latent representation, as a list of (topic_id, topic_value) 2-tuples.
        
        This is done by folding input document into the latent topic space. 
        
        Note that this function returns the latent space representation **scaled by the
        singular values**. To return non-scaled embedding, set `scaled` to False.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        if utils.isCorpus(bow):
            return self._apply(bow)
        
        assert self.projection.u is not None, "decomposition not initialized yet"
        vec = numpy.asfortranarray(matutils.sparse2full(bow, self.numTerms), dtype = self.projection.u.dtype)
        vec.shape = (self.numTerms, 1)
        topicDist = scipy.linalg.fblas.dgemv(1.0, self.projection.u, vec, trans = True) # u^T * x
        if scaled:
            topicDist = (1.0 / self.projection.s) * topicDist # s^-1 * u^T * x
        return [(topicId, float(topicValue)) for topicId, topicValue in enumerate(topicDist)
                if numpy.isfinite(topicValue) and not numpy.allclose(topicValue, 0.0)]
    

    def printTopic(self, topicNo, topN = 10):
        """
        Return a specified topic (=left singular vector), 0 <= `topicNo` < `self.numTopics`, 
        as string.
        
        Return only the `topN` words which contribute the most to the direction 
        of the topic (both negative and positive).
        
        >>> lsimodel.printTopic(10, topN = 5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'
        
        """
        # size of the projection matrix can actually be smaller than `self.numTopics`,
        # if there were not enough factors (real rank of input matrix smaller than
        # `self.numTopics`). in that case, return empty string
        if topicNo >= len(self.projection.u.T):
            return ''
        c = numpy.asarray(self.projection.u.T[topicNo, :]).flatten()
        norm = numpy.sqrt(numpy.sum(c * c))
        most = numpy.abs(c).argsort()[::-1][:topN]
        return ' + '.join(['%.3f*"%s"' % (1.0 * c[val] / norm, self.id2word[val]) for val in most])


    def printTopics(self, numTopics = 5, numWords = 10):
        for i in xrange(min(numTopics, self.numTopics)):
            if i < len(self.projection.s):
                logger.info("topic #%i(%.3f): %s" % 
                            (i, self.projection.s[i], 
                             self.printTopic(i, topN = numWords)))


    def printDebug(self, numTopics = 5, numWords = 10):
        """
        Print (to log) the most salient words of the first `numTopics` topics.
        
        Unlike `printTopics()`, this looks for words that are significant for a 
        particular topic *and* not for others. This *should* result in a more
        human-interpretable description of topics.
        """
        # only wrap the module-level fnc
        printDebug(self.id2word, 
                   self.projection.u, self.projection.s,
                   range(min(numTopics, len(self.projection.u.T))), 
                   numWords = numWords)
#endclass LsiModel


def printDebug(id2token, u, s, topics, numWords = 10, numNeg = None):
    if numNeg is None:
        # by default, print half as many salient negative words as positive
        numNeg = numWords / 2
        
    logger.info('computing word-topic salience for %i topics' % len(topics))
    topics, result = set(topics), {}
    # TODO speed up by block computation
    for uvecno, uvec in enumerate(u):
        uvec = numpy.abs(numpy.asarray(uvec).flatten())
        udiff = uvec / numpy.sqrt(numpy.sum(uvec * uvec))
        for topic in topics:
            result.setdefault(topic, []).append((udiff[topic], uvecno))
    
    logger.debug("printing %i+%i salient words" % (numWords, numNeg))
    for topic in sorted(result.iterkeys()):
        weights = sorted(result[topic], reverse = True)
        _, most = weights[0]
        if u[most, topic] < 0.0: # the most significant word has negative sign => flip sign of u[most]
            normalize = -1.0
        else:
            normalize = 1.0
        
        # order features according to salience; ignore near-zero entries in u
        pos, neg = [], []
        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] > 0.0001:
                pos.append('%s(%.3f)' % (id2token[uvecno], normalize * u[uvecno, topic]))
                if len(pos) >= numWords:
                    break
        
        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] < -0.0001:
                neg.append('%s(%.3f)' % (id2token[uvecno], normalize * u[uvecno, topic]))
                if len(neg) >= numNeg:
                    break

        logger.info('topic #%s(%.3f): %s, ..., %s' % (topic, s[topic], ', '.join(pos), ', '.join(neg)))


def svdUpdate(U, S, V, a, b):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.
    
    The original matrix X is not needed at all, so this function implements flexible 
    *online* updates to an existing decomposition. 
    
    `a` and `b` are (m, 1) and (n, 1) matrices.
    
    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).
    
    This is the rank-1 update as described in
    *Brand, 2006: Fast low-rank modifications of the thin singular value decomposition*,
    but without separating the basis from rotations.
    """
    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = numpy.asmatrix(S)
    U = numpy.asmatrix(U)
    if V is not None:
        V = numpy.asmatrix(V)
    a = numpy.asmatrix(a).reshape(a.size, 1)
    b = numpy.asmatrix(b).reshape(b.size, 1)
    
    rank = S.shape[0]
    
    # eq (6)
    m = U.T * a
    p = a - U * m
    Ra = numpy.sqrt(p.T * p)
    if float(Ra) < 1e-10:
        logger.debug("input already contained in a subspace of U; skipping update")
        return U, S, V
    P = (1.0 / float(Ra)) * p
    
    if V is not None:
        # eq (7)
        n = V.T * b
        q = b - V * n
        Rb = numpy.sqrt(q.T * q)
        if float(Rb) < 1e-10:
            logger.debug("input already contained in a subspace of V; skipping update")
            return U, S, V
        Q = (1.0 / float(Rb)) * q
    else:
        n = numpy.matrix(numpy.zeros((rank, 1)))
        Rb = numpy.matrix([[1.0]])    
    
    if float(Ra) > 1.0 or float(Rb) > 1.0:
        logger.debug("insufficient target rank (Ra=%.3f, Rb=%.3f); this update will result in major loss of information"
                      % (float(Ra), float(Rb)))
    
    # eq (8)
    K = numpy.matrix(numpy.diag(list(numpy.diag(S)) + [0.0])) + numpy.bmat('m ; Ra') * numpy.bmat('n ; Rb').T
    
    # eq (5)
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = numpy.matrix(numpy.diag(s[: rank]))
    Up = numpy.bmat('U P') * tUp
    if V is not None:
        Vp = numpy.bmat('V Q') * tVp
    else:
        Vp = None
    Sp = tSp
    
    return Up, Sp, Vp


def iterSvd(corpus, numTerms, numFactors, numIter = 200, initRate = None, convergence = 1e-4):
    """
    Perform iterative Singular Value Decomposition on a streaming corpus, returning 
    `numFactors` greatest factors U,S,V^T (ie., not necessarily the full spectrum).
    
    The parameters `numIter` (maximum number of iterations) and `initRate` (gradient 
    descent step size) guide convergency of the algorithm. 
    
    The algorithm performs at most `numFactors*numIters` passes over the corpus.
    
    See **Genevieve Gorrell: Generalized Hebbian Algorithm for Incremental Singular 
    Value Decomposition in Natural Language Processing. EACL 2006.**
    
    Use of this function is deprecated; although it works, it is several orders of 
    magnitude slower than our own, direct (non-stochastic) version (which
    operates in a single pass, too, and can be distributed). 
    
    I keep this function here purely for backup reasons.
    """
    logger.info("performing incremental SVD for %i factors" % numFactors)

    # define the document/term singular vectors, fill them with a little random noise
    sdoc = 0.01 * numpy.random.randn(len(corpus), numFactors)
    sterm = 0.01 * numpy.random.randn(numTerms, numFactors)
    if initRate is None:
        initRate = 1.0 / numpy.sqrt(numTerms)
        logger.info("using initial learn rate of %f" % initRate)

    rmse = rmseOld = numpy.inf
    for factor in xrange(numFactors):
        learnRate = initRate
        for iterNo in xrange(numIter):
            errors = 0.0
            rate = learnRate / (1.0 + 9.0 * iterNo / numIter) # gradually decrease the learning rate to 1/10 of the initial value
            logger.debug("setting learning rate to %f" % rate)
            for docNo, doc in enumerate(corpus):
                vec = dict(doc)
                if docNo % 10 == 0:
                    logger.debug('PROGRESS: at document %i/%i' % (docNo, len(corpus)))
                vdoc = sdoc[docNo, factor]
                vterm = sterm[:, factor] # create a view (not copy!) of a matrix row
                
                # reconstruct one document, using all previous factors <0..factor-1>
                # we do not pre-cache the dot products because that takes insane amounts of memory (=doesn't scale with corpus size)
                recon = numpy.dot(sdoc[docNo, :factor], sterm[:, :factor].T) # numpy.dot is very fast anyway, this is not the bottleneck
                
                for termId in xrange(numTerms): # this loop is.
                    # error of one matrix element = real value - reconstructed value
                    error = vec.get(termId, 0.0) - (recon[termId] + vdoc * vterm[termId])
                    errors += error * error
                    
                    # update the singular vectors
                    tmp = vdoc
                    vdoc += rate * error * vterm[termId]
                    vterm[termId] += rate * error * tmp
                sdoc[docNo, factor] = vdoc
            
            # compute rmse = root mean square error of the reconstructed matrix
            rmse = numpy.exp(0.5 * (numpy.log(errors) - numpy.log(len(corpus)) - numpy.log(numTerms)))
            if not numpy.isfinite(rmse) or rmse > rmseOld:
                learnRate /= 2.0 # if we are not converging (oscillating), halve the learning rate
                logger.info("iteration %i diverged; halving the learning rate to %f" %
                             (iterNo, learnRate))
            
            # check convergence, looking for an early exit (but no sooner than 10% 
            # of numIter have passed)
            converged = numpy.divide(numpy.abs(rmseOld - rmse), rmseOld)
            logger.info("factor %i, finished iteration %i, rmse=%f, rate=%f, converged=%f" %
                          (factor, iterNo, rmse, rate, converged))
            if iterNo > numIter / 10 and numpy.isfinite(converged) and converged <= convergence:
                logger.debug("factor %i converged in %i iterations" % (factor, iterNo + 1))
                break
            rmseOld = rmse
        
        logger.info("PROGRESS: finished SVD factor %i/%i, RMSE<=%f" % 
                     (factor + 1, numFactors, rmse))
    
    # normalize the vectors to unit length; also keep the norms
    sdocLens = numpy.sqrt(numpy.sum(sdoc * sdoc, axis = 0))
    stermLens = numpy.sqrt(numpy.sum(sterm * sterm, axis = 0))
    sdoc /= sdocLens
    sterm /= stermLens
    
    # singular values are the norms
    svals = sdocLens * stermLens
    return sterm, svals, sdoc.T


def stochasticSvd(corpus, rank, chunks = 10000, num_terms = None, extra_dims = None, eps = 1e-6):
    """
    Return U, S -- the left singular vectors and the singular values of the streamed 
    input corpus.
    
    This may actually return less than the requested number of `rank` factors, 
    in case the input is of lower rank. Also note that the decomposition is unique
    up the the sign of the left singular vectors (columns of U).
    
    This is a streamed, two-pass algorithm. In case you can only afford a single
    pass over the input corpus, set onepass=True in LsiModel and avoid using this
    algorithm.

    The decomposition algorithm is based on stochastic approximation from::
    
    **Halko, Martinsson, Tropp. Finding structure with randomness, 2009.**
    
    """
    rank = int(rank)
    if extra_dims is None:
        samples = 2 * rank # use more samples than requested factors, to improve accuracy
    else:
        samples = rank + int(extra_dims)
    
    if num_terms is None:
        logger.warning("number of terms not provided; will scan the corpus (ONE EXTRA PASS, MAY BE SLOW) to determine it")
        num_terms = len(utils.dictFromCorpus(corpus))
        logger.info("found %i terms" % num_terms)
    else:
        num_terms = int(num_terms)
    
    eps = max(float(eps), 1e-9) # must ignore near-zero eigenvalues (probably numerical error); the associated eigenvectors are most likely garbage
    
    # first pass: construct the orthonormal action matrix Q
    # proceed in blocks of `chunks` documents (much faster than going one-by-one 
    # and more memory friendly than processing all documents at once)
    y = numpy.zeros(dtype = numpy.float64, shape = (num_terms, samples))
    logger.info("constructing %s orthonormal action matrix" % str(y.shape))
    
    chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
    for chunk_no, (key, group) in enumerate(chunker):
        if chunk_no % 1 == 0:
            logger.info('PROGRESS: at document #%i' % (chunk_no * chunks))
        # construct the chunk as a sparse matrix, to minimize memory overhead
        # definitely avoid materializing it as a dense (num_terms x chunks) matrix!
        chunk = matutils.corpus2csc(num_terms, (doc for _, doc in group)) # documents = columns of sparse CSC
        m, n = chunk.shape
        assert m == num_terms
        assert n <= chunks # the very last chunk may be smaller
        o = numpy.random.normal(0.0, 1.0, (n, samples)) # draw a random gaussian matrix
        sparsetools.csc_matvecs(num_terms, n, samples, chunk.indptr, # y = y + chunk * o
                                chunk.indices, chunk.data, o.ravel(), y.ravel())
        del chunk, o
    
    q, r = numpy.linalg.qr(y) # orthonormalize the range
    del y # Y not needed anymore, free up mem
    samples = clipSpectrum(numpy.diag(r), samples, discard = eps)
    q = q[:, :samples].copy() # discard bogus columns, in case Y was rank-deficient
    
    # second pass: construct the covariance matrix X = B^T * B, where B = A * Q
    # again, construct X incrementally, in chunks of `chunks` documents from the streaming 
    # input corpus A, to avoid using O(number of documents) memory
    x = numpy.zeros(shape = (samples, samples), dtype = numpy.float64, order = 'F')
    logger.info("constructing %s covariance matrix" % str(x.shape))
    chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
    for chunk_no, (key, group) in enumerate(chunker):
        if chunk_no % 1 == 0:
            logger.info('PROGRESS: at document #%i' % (chunk_no * chunks))
        chunk = matutils.corpus2csc(num_terms, (doc for _, doc in group)).transpose()
        b = chunk * q # sparse * dense matrix multiply
        del chunk
        x += numpy.dot(b.T, b) # TODO should call the BLAS routine SYRK, but there is no SYRK wrapper in scipy :(
        del b
    
    # now we're ready to compute decomposition of the small matrix X
    logger.info("computing decomposition of the %s covariance matrix" % str(x.shape))
    u, s, vt = numpy.linalg.svd(x) # could use linalg.eigh, but who cares... and svd returns the factors already sorted :)
    keep = clipSpectrum(s, rank, discard = eps)
    
    logger.info("computing the final decomposition")
    s = numpy.sqrt(s[:keep]) # sqrt to go back from singular values of B^T*B to singular values of B = singular values of the corpus
    u = numpy.dot(q, u[:, :keep]) # go back from left singular vectors of B to left singular vectors of the corpus
    return u, s
    
