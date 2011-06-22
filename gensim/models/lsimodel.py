#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for Latent Semantic Indexing.

This module actually contains several algorithms for decomposition of large corpora, a
combination of which effectively and transparently allows building LSI models for:

* corpora much larger than RAM: only constant memory is needed, independent of
  the corpus size (though still dependent on the feature set size)
* corpora that are streamed: documents are only accessed sequentially, no
  random-access
* corpora that cannot be even temporarily stored: each document can only be
  seen once and must be processed immediately (one-pass algorithm)
* distributed computing for very large corpora, making use of a cluster of
  machines

Wall-clock performance on the English Wikipedia (2G corpus positions, 3.2M
documents, 100K features, 0.5G non-zero entries in the final TF-IDF matrix),
requesting the top 400 LSI factors:


====================================================== ============ ==================
 algorithm                                             serial       distributed
====================================================== ============ ==================
 one-pass merge algorithm                              5h14m        1h41m
 multi-pass stochastic algo (with 2 power iterations)  5h39m        N/A [1]_
====================================================== ============ ==================


*serial* = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

*distributed* = cluster of four logical nodes on three physical machines, each
with dual core Xeon 2.0GHz, 4GB RAM, ATLAS

.. [1] The stochastic algo could be distributed too, but most time is already spent
   reading/decompressing the input from disk in its 4 passes. The extra network
   traffic due to data distribution across cluster nodes would likely make it
   *slower*.

"""


import logging
import itertools
import sys

import numpy
import scipy.sparse
from scipy.sparse import sparsetools

from gensim import interfaces, matutils, utils


logger = logging.getLogger('gensim.models.lsimodel')


# accuracy defaults for the multi-pass stochastic algo
P2_EXTRA_DIMS = 100 # set to `None` for dynamic P2_EXTRA_DIMS=k
P2_EXTRA_ITERS = 2


def clip_spectrum(s, k, discard=0.001):
    """
    Given eigenvalues `s`, return how many factors should be kept to avoid
    storing spurious (tiny, numerically instable) values.

    This will ignore the tail of the spectrum with relative combined mass < min(`discard`, 1/k).

    The returned value is clipped against `k` (= never return more than `k`).
    """
    # compute relative contribution of eigenvalues towards the energy spectrum
    rel_spectrum = numpy.abs(1.0 - numpy.cumsum(s / numpy.sum(s)))
    # ignore the last `discard` mass (or 1/k, whichever is smaller) of the spectrum
    small = 1 + len(numpy.where(rel_spectrum > min(discard, 1.0 / k))[0])
    k = min(k, small) # clip against k
    logger.info("keeping %i factors (discarding %.3f%% of energy spectrum)" %
                (k, 100 * rel_spectrum[k - 1]))
    return k


def asfarray(a, name=''):
    if not a.flags.f_contiguous:
        logger.debug("converting %s array %s to FORTRAN order" % (a.shape, name))
        a = numpy.asfortranarray(a)
    return a


def ascarray(a, name=''):
    if not a.flags.contiguous:
        logger.debug("converting %s array %s to C order" % (a.shape, name))
        a = numpy.ascontiguousarray(a)
    return a


class Projection(utils.SaveLoad):
    def __init__(self, m, k, docs=None, use_svdlibc=False):
        """
        Construct the (U, S) projection from a corpus `docs`. The projection can
        be later updated by merging it with another Projection via `self.merge()`.

        This is the class taking care of the 'core math'; interfacing with corpora,
        splitting large corpora into chunks and merging them etc. is done through
        the higher-level `LsiModel` class.
        """
        self.m, self.k = m, k
        if docs is not None:
            # base case decomposition: given a job `docs`, compute its decomposition,
            # *in-core*.
            if not use_svdlibc:
                u, s = stochastic_svd(docs, k, chunksize=sys.maxint, num_terms=m,
                    power_iters=P2_EXTRA_ITERS, extra_dims=P2_EXTRA_DIMS)
            else:
                try:
                    import sparsesvd
                except ImportError:
                    raise ImportError("`sparsesvd` module requested but not found; run `easy_install sparsesvd`")
                logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
                if not scipy.sparse.issparse(docs):
                    docs = matutils.corpus2csc(docs)
                ut, s, vt = sparsesvd.sparsesvd(docs, k + 30) # ask for extra factors, because for some reason SVDLIBC sometimes returns fewer factors than requested
                u = ut.T
                del ut, vt
                k = clip_spectrum(s**2, self.k)
            self.u = u[:, :k].copy()
            self.s = s[:k].copy()
        else:
            self.u, self.s = None, None


    def empty_like(self):
        return Projection(self.m, self.k)


    def merge(self, other, decay=1.0):
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
            self.u = other.u.copy()
            self.s = other.s.copy()
            return
        if self.m != other.m:
            raise ValueError("vector space mismatch: update is using %s features, expected %s" %
                             (other.m, self.m))
        logger.info("merging projections: %s + %s" % (str(self.u.shape), str(other.u.shape)))
        m, n1, n2 = self.u.shape[0], self.u.shape[1], other.u.shape[1]
        # TODO Maybe keep the bases as elementary reflectors, without
        # forming explicit matrices with ORGQR.
        # The only operation we ever need is basis^T*basis ond basis*component.
        # But how to do that in scipy? And is it fast(er)?

        # find component of u2 orthogonal to u1
        logger.debug("constructing orthogonal component")
        self.u = asfarray(self.u, 'self.u')
        c = numpy.dot(self.u.T, other.u)
        self.u = ascarray(self.u, 'self.u')
        other.u -= numpy.dot(self.u, c)

        other.u = [other.u] # do some reference magic and call qr_destroy, to save RAM
        q, r = matutils.qr_destroy(other.u) # q, r = QR(component)
        assert not other.u

        # find the rotation that diagonalizes r
        k = numpy.bmat([[numpy.diag(decay * self.s), numpy.multiply(c, other.s)],
                        [matutils.pad(numpy.array([]).reshape(0, 0), min(m, n2), n1), numpy.multiply(r, other.s)]])
        logger.debug("computing SVD of %s dense matrix" % str(k.shape))
        try:
            # in numpy < 1.1.0, running SVD sometimes results in "LinAlgError: SVD did not converge'.
            # for these early versions of numpy, catch the error and try to compute
            # SVD again, but over k*k^T.
            # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg07224.html and
            # bug ticket http://projects.scipy.org/numpy/ticket/706
            u_k, s_k, _ = numpy.linalg.svd(k, full_matrices=False) # TODO *ugly overkill*!! only need first self.k SVD factors... but there is no LAPACK wrapper for partial svd/eigendecomp in numpy :(
        except numpy.linalg.LinAlgError:
            logger.error("SVD(A) failed; trying SVD(A * A^T)")
            u_k, s_k, _ = numpy.linalg.svd(numpy.dot(k, k.T), full_matrices=False) # if this fails too, give up with an exception
            s_k = numpy.sqrt(s_k) # go back from eigen values to singular values

        k = clip_spectrum(s_k**2, self.k)
        u1_k, u2_k, s_k = numpy.array(u_k[:n1, :k]), numpy.array(u_k[n1:, :k]), s_k[:k]

        # update & rotate current basis U = [U, U']*[U1_k, U2_k]
        logger.debug("updating orthonormal basis U")
        self.s = s_k
        self.u = ascarray(self.u, 'self.u')
        self.u = numpy.dot(self.u, u1_k)

        q = ascarray(q, 'q')
        q = numpy.dot(q, u2_k)
        self.u += q
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
    3. `add_documents()` for incrementally updating the model with new documents.

    Model persistency is achieved via its load/save methods.

    """
    def __init__(self, corpus=None, num_topics=200, id2word=None, chunksize=20000,
                 decay=1.0, distributed=False, onepass=True,
                 power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS):
        """
        `num_topics` is the number of requested factors (latent dimensions).

        After the model has been trained, you can estimate topics for an
        arbitrary, unseen document, using the ``topics = self[document]`` dictionary
        notation. You can also add new training documents, with ``self.add_documents``,
        so that training can be stopped and resumed at any time, and the
        LSI transformation is available at any point.

        If you specify a `corpus`, it will be used to train the model. See the
        method `add_documents` for a description of the `chunksize` and `decay` parameters.

        Turn `onepass` off to force a multi-pass stochastic algorithm.

        `power_iters` and `extra_samples` affect the accuracy of the stochastic
        multi-pass algorithm, which is used either internally (`onepass=True`) or
        as the front-end algorithm (`onepass=False`). Increasing the number of
        power iterations improves accuracy, but lowers performance. See [2]_ for
        some hard numbers.

        Turn on `distributed` to enable distributed computing.

        Example:

        >>> lsi = LsiModel(corpus, num_topics=10)
        >>> print lsi[doc_tfidf] # project some document into LSI space
        >>> lsi.add_documents(corpus2) # update LSI on additional documents
        >>> print lsi[doc_tfidf]

        .. [2] http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf

        """
        self.id2word = id2word
        self.num_topics = int(num_topics)
        self.chunksize = int(chunksize)
        self.decay = float(decay)
        if distributed:
            if not onepass:
                logger.warning("forcing the one-pass algorithm for distributed LSA")
                onepass = True
        self.onepass = onepass
        self.extra_samples, self.power_iters = extra_samples, power_iters

        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 1 + max([-1] + self.id2word.keys())

        self.docs_processed = 0
        self.projection = Projection(self.num_terms, self.num_topics)

        self.numworkers = 1
        if not distributed:
            logger.info("using serial LSI version on this node")
            self.dispatcher = None
        else:
            if not onepass:
                raise NotImplementedError("distributed stochastic LSA not implemented yet; "
                                          "run either distributed one-pass, or serial randomized.")
            try:
                import Pyro
                ns = Pyro.naming.locateNS()
                dispatcher = Pyro.core.Proxy('PYRONAME:gensim.lsi_dispatcher@%s' % ns._pyroUri.location)
                dispatcher._pyroOneway.add("exit")
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word=self.id2word, num_topics = num_topics,
                                      chunksize=chunksize, decay=decay,
                                      distributed=False, onepass=onepass)
                self.dispatcher = dispatcher
                self.numworkers = len(dispatcher.getworkers())
                logger.info("using distributed version with %i workers" % self.numworkers)
            except Exception, err:
                # distributed version was specifically requested, so this is an error state
                logger.error("failed to initialize distributed LSI (%s)" % err)
                raise RuntimeError("failed to initialize distributed LSI (%s)" % err)

        if corpus is not None:
            self.add_documents(corpus)


    def add_documents(self, corpus, chunksize=None, decay=None):
        """
        Update singular value decomposition to take into account a new
        corpus of documents.

        Training proceeds in chunks of `chunksize` documents at a time. The size of
        `chunksize` is a tradeoff between increased speed (bigger `chunksize`)
        vs. lower memory footprint (smaller `chunksize`). If the distributed mode
        is on, each chunk is sent to a different worker/computer.

        Setting `decay` < 1.0 causes re-orientation towards new data trends in the
        input document stream, by giving less emphasis to old observations. This allows
        LSA to gradually "forget" old observations (documents) and give more
        preference to new ones.
        """
        logger.info("updating model with new documents")

        # get computation parameters; if not specified, use the ones from constructor
        if chunksize is None:
            chunksize = self.chunksize
        if decay is None:
            decay = self.decay

        if not scipy.sparse.issparse(corpus):
            if not self.onepass:
                # we are allowed multiple passes over the input => use a faster, randomized two-pass algo
                update = Projection(self.num_terms, self.num_topics, None)
                update.u, update.s = stochastic_svd(corpus, self.num_topics,
                    num_terms=self.num_terms, chunksize=chunksize,
                    extra_dims=self.extra_samples, power_iters=self.power_iters)
                self.projection.merge(update, decay=decay)
            else:
                # the one-pass algo
                doc_no = 0
                chunker = itertools.groupby(enumerate(corpus), key=lambda (docno, doc): docno / chunksize)
                for chunk_no, (key, group) in enumerate(chunker):
                    logger.info("preparing a new chunk of documents")
                    corpus = list(doc for _, doc in group)
                    nnz = sum(len(doc) for doc in corpus)
                    # construct the job as a sparse matrix, to minimize memory overhead
                    # definitely avoid materializing it as a dense matrix!
                    logger.debug("converting corpus to csc format")
                    job = matutils.corpus2csc(corpus, num_docs=len(corpus), num_terms=self.num_terms, num_nnz=nnz)
                    del corpus
                    doc_no += job.shape[1]
                    if self.dispatcher:
                        # distributed version: add this job to the job queue, so workers can work on it
                        logger.debug("creating job #%i" % chunk_no)
                        self.dispatcher.putjob(job) # put job into queue; this will eventually block, because the queue has a small finite size
                        del job
                        logger.info("dispatched documents up to #%s" % doc_no)
                    else:
                        # serial version, there is only one "worker" (myself) => process the job directly
                        update = Projection(self.num_terms, self.num_topics, job)
                        del job
                        self.projection.merge(update, decay=decay)
                        del update
                        logger.info("processed documents up to #%s" % doc_no)
                        self.print_topics(5)

                # wait for all workers to finish (distributed version only)
                if self.dispatcher:
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    self.projection = self.dispatcher.getstate()
#            logger.info("top topics after adding %i documents" % doc_no)
#            self.print_debug(10)
        else:
            assert not self.dispatcher, "must be in serial mode to receive jobs"
            assert self.onepass, "distributed two-pass algo not supported yet"
            update = Projection(self.num_terms, self.num_topics, corpus.tocsc())
            self.projection.merge(update, decay=decay)
            logger.info("processed sparse job of %i documents" % (corpus.shape[1]))


    def __str__(self):
        return "LsiModel(num_terms=%s, num_topics=%s, decay=%s, chunksize=%s)" % \
                (self.num_terms, self.num_topics, self.decay, self.chunksize)


    def __getitem__(self, bow, scaled=False, chunksize=256):
        """
        Return latent representation, as a list of (topic_id, topic_value) 2-tuples.

        This is done by folding input document into the latent topic space.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus and chunksize:
            # by default, transform 256 documents at once, when called as `lsi[corpus]`.
            # this chunking is completely transparent to the user, but it speeds
            # up internal computations (one mat * mat multiplication, instead of
            # 256 smaller mat * vec multiplications, better use of cache).
            return self._apply(bow, chunksize=chunksize)

        if is_corpus:
            vec = numpy.vstack(matutils.sparse2full(doc, self.num_terms).astype(self.projection.u.dtype) for doc in bow).T
        else:
            vec = matutils.sparse2full(bow, self.num_terms).astype(self.projection.u.dtype)

        assert self.projection.u is not None, "decomposition not initialized yet"
        # automatically convert U to memory order suitable for column slicing
        # this will ideally be done only once, at the very first lsi[query] transformation
        self.projection.u = asfarray(self.projection.u)
        topic_dist = numpy.dot(self.projection.u[:, :self.num_topics].T, vec) # u^-1 * x
        if scaled:
            topic_dist = (1.0 / self.projection.s[:self.num_topics]) * topic_dist # s^-1 * u^-1 * x

        # convert a numpy array to gensim sparse vector = tuples of (feature_id, feature_weight),
        # with no zero weights.
        if not is_corpus:
            # lsi[single_document]
            result = matutils.full2sparse(topic_dist)
        else:
            # lsi[chunk of documents]
            result = matutils.Dense2Corpus(topic_dist)
        return result


    def print_topic(self, topicno, topn=10):
        """
        Return a specified topic (=left singular vector), 0 <= `topicno` < `self.num_topics`,
        as string.

        Return only the `topn` words which contribute the most to the direction
        of the topic (both negative and positive).

        >>> lsimodel.print_topic(10, topn=5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
        # size of the projection matrix can actually be smaller than `self.num_topics`,
        # if there were not enough factors (real rank of input matrix smaller than
        # `self.num_topics`). in that case, return empty string
        if topicno >= len(self.projection.u.T):
            return ''
        c = numpy.asarray(self.projection.u.T[topicno, :]).flatten()
        norm = numpy.sqrt(numpy.sum(numpy.dot(c, c)))
        most = numpy.abs(c).argsort()[::-1][:topn]
        return ' + '.join(['%.3f*"%s"' % (1.0 * c[val] / norm, self.id2word[val]) for val in most])


    def print_topics(self, num_topics=5, num_words=10):
        for i in xrange(min(num_topics, self.num_topics)):
            if i < len(self.projection.s):
                logger.info("topic #%i(%.3f): %s" %
                            (i, self.projection.s[i],
                             self.print_topic(i, topn=num_words)))


    def print_debug(self, num_topics=5, num_words=10):
        """
        Print (to log) the most salient words of the first `num_topics` topics.

        Unlike `print_topics()`, this looks for words that are significant for a
        particular topic *and* not for others. This *should* result in a more
        human-interpretable description of topics.
        """
        # only wrap the module-level fnc
        print_debug(self.id2word, self.projection.u, self.projection.s,
                   range(min(num_topics, len(self.projection.u.T))),
                   num_words=num_words)


    def save(self, fname):
        """
        Override the default `save` (which uses cPickle), because that's
        too inefficient and cPickle has bugs. Instead, single out the large transformation
        matrix and store that separately in binary format (that can be directly
        mmap'ed back in `load()`), under `fname.npy`.
        """
        logger.info("storing %s object to %s and %s" % (self.__class__.__name__, fname, fname + '.npy'))
        if self.projection.u is None:
            # model not initialized: there is no projection
            utils.pickle(self, fname)

        # first, remove the projection from self.__dict__, so it doesn't get pickled
        u = self.projection.u
        del self.projection.u
        try:
            utils.pickle(self, fname) # store projection-less object
            numpy.save(fname + '.npy', u) # store projection
        finally:
            self.projection.u = u


    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see `save`).
        """
        logger.debug("loading %s object from %s" % (cls.__name__, fname))
        result = utils.unpickle(fname)
        ufname = fname + '.npy'
        try:
            result.projection.u = numpy.load(ufname, mmap_mode='r') # load back as read-only
        except:
            logger.debug("failed to load mmap'ed projection from %s" % ufname)
        return result
#endclass LsiModel


def print_debug(id2token, u, s, topics, num_words=10, num_neg=None):
    if num_neg is None:
        # by default, print half as many salient negative words as positive
        num_neg = num_words / 2

    logger.info('computing word-topic salience for %i topics' % len(topics))
    topics, result = set(topics), {}
    # TODO speed up by block computation
    for uvecno, uvec in enumerate(u):
        uvec = numpy.abs(numpy.asarray(uvec).flatten())
        udiff = uvec / numpy.sqrt(numpy.sum(numpy.dot(uvec, uvec)))
        for topic in topics:
            result.setdefault(topic, []).append((udiff[topic], uvecno))

    logger.debug("printing %i+%i salient words" % (num_words, num_neg))
    for topic in sorted(result.iterkeys()):
        weights = sorted(result[topic], key=lambda x: -abs(x[0]))
        _, most = weights[0]
        if u[most, topic] < 0.0: # the most significant word has a negative sign => flip sign of u[most]
            normalize = -1.0
        else:
            normalize = 1.0

        # order features according to salience; ignore near-zero entries in u
        pos, neg = [], []
        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] > 0.0001:
                pos.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(pos) >= num_words:
                    break

        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] < -0.0001:
                neg.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(neg) >= num_neg:
                    break

        logger.info('topic #%s(%.3f): %s, ..., %s' % (topic, s[topic], ', '.join(pos), ', '.join(neg)))



def stochastic_svd(corpus, rank, num_terms, chunksize=20000, extra_dims=None,
                  power_iters=0, dtype=numpy.float64, eps=1e-6):
    """
    Return (U, S): the left singular vectors and the singular values of the streamed
    input corpus `corpus` [3]_.

    This may actually return less than the requested number of top `rank` factors,
    in case the input is of lower rank. The `extra_dims` (oversampling) and especially
    `power_iters` (power iterations) parameters affect accuracy of the decomposition.

    This algorithm uses `2+power_iters` passes over the data. In case you can only
    afford a single pass over the input corpus, set `onepass=True` in :class:`LsiModel`
    and avoid using this algorithm directly.

    The decomposition algorithm is based on
    **Halko, Martinsson, Tropp. Finding structure with randomness, 2009.**

    .. [3] If `corpus` is a scipy.sparse matrix instead, it is assumed the whole
       corpus fits into core memory and a different (more efficient) code path is chosen.
    """
    rank = int(rank)
    if extra_dims is None:
        samples = max(10, 2 * rank) # use more samples than requested factors, to improve accuracy
    else:
        samples = rank + int(extra_dims)
    logger.info("using %i extra samples and %i power iterations" % (samples - rank, power_iters))

    num_terms = int(num_terms)

    eps = max(float(eps), 1e-9) # must ignore near-zero eigenvalues (probably numerical error); the associated eigenvectors are typically unstable/garbage

    # first phase: construct the orthonormal action matrix Q = orth(Y) = orth((A * A.T)^q * A * O)
    # build Y in blocks of `chunksize` documents (much faster than going one-by-one
    # and more memory friendly than processing all documents at once)
    y = numpy.zeros(dtype=dtype, shape=(num_terms, samples))
    logger.info("1st phase: constructing %s action matrix" % str(y.shape))

    if scipy.sparse.issparse(corpus):
        m, n = corpus.shape
        assert num_terms == m, "mismatch in number of features: %i in sparse matrix vs. %i parameter" % (m, num_terms)
        o = numpy.random.normal(0.0, 1.0, (n, samples)).astype(y.dtype) # draw a random gaussian matrix
        sparsetools.csc_matvecs(m, n, samples, corpus.indptr, corpus.indices,
                                corpus.data, o.ravel(), y.ravel()) # y = corpus * o
        del o

        # unlike numpy, scipy.sparse `astype()` copies everything, even if there is no change to dtype!
        # so check for equal dtype explicitly, to avoid the extra memory footprint if possible
        if y.dtype != dtype:
            y = y.astype(dtype)
        logger.debug("running %i power iterations" % power_iters)
        for power_iter in xrange(power_iters):
            y = corpus.T * y
            y = corpus * y
    else:
        chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunksize)
        num_docs = 0
        for chunk_no, (key, group) in enumerate(chunker):
            logger.info('PROGRESS: at document #%i' % (chunk_no * chunksize))
            # construct the chunk as a sparse matrix, to minimize memory overhead
            # definitely avoid materializing it as a dense (num_terms x chunksize) matrix!
            chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype) # documents = columns of sparse CSC
            m, n = chunk.shape
            assert m == num_terms
            assert n <= chunksize # the very last chunk of A is allowed to be smaller in size
            num_docs += n
            logger.debug("multiplying chunk * gauss")
            o = numpy.random.normal(0.0, 1.0, (n, samples)).astype(dtype) # draw a random gaussian matrix
            sparsetools.csc_matvecs(num_terms, n, samples, chunk.indptr, # y = y + chunk * o
                                    chunk.indices, chunk.data, o.ravel(), y.ravel())
            del chunk, o

        for power_iter in xrange(power_iters):
            logger.info("running power iteration #%i" % (power_iter + 1))
            yold = y.copy()
            y[:] = 0.0
            chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunksize)
            for chunk_no, (key, group) in enumerate(chunker):
                logger.info('PROGRESS: at document #%i/%i' % (chunk_no * chunksize, num_docs))
                chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype) # documents = columns of sparse CSC
                tmp = chunk.T * yold
                tmp = chunk * tmp
                del chunk
                y += tmp
            del yold

    logger.info("orthonormalizing %s action matrix" % str(y.shape))
    y = [y]
    q, r = matutils.qr_destroy(y) # orthonormalize the range
    del y
    samples = clip_spectrum(numpy.diag(r), samples, discard=eps)
    qt = q[:, :samples].T.copy()
    del q

    if scipy.sparse.issparse(corpus):
        b = qt * corpus
        logger.info("2nd phase: running dense svd on %s matrix" % str(b.shape))
        u, s, vt = numpy.linalg.svd(b, full_matrices=False)
        del b, vt
    else:
        # second phase: construct the covariance matrix X = B * B.T, where B = Q.T * A
        # again, construct X incrementally, in chunks of `chunksize` documents from the streaming
        # input corpus A, to avoid using O(number of documents) memory
        x = numpy.zeros(shape = (samples, samples), dtype=dtype)
        logger.info("2nd phase: constructing %s covariance matrix" % str(x.shape))
        chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunksize)
        for chunk_no, (key, group) in enumerate(chunker):
            logger.info('PROGRESS: at document #%i/%i' % (chunk_no * chunksize, num_docs))
            chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype)
            b = qt * chunk # dense * sparse matrix multiply
            x += numpy.dot(b, b.T) # TODO should call the BLAS routine SYRK, but there is no SYRK wrapper in scipy :(
            del chunk, b

        # now we're ready to compute decomposition of the small matrix X
        logger.info("running dense decomposition on %s covariance matrix" % str(x.shape))
        u, s, vt = numpy.linalg.svd(x) # could use linalg.eigh, but who cares... and svd returns the factors already sorted :)
        s = numpy.sqrt(s) # sqrt to go back from singular values of X to singular values of B = singular values of the corpus
    q = qt.T.copy()
    del qt

    logger.info("computing the final decomposition")
    keep = clip_spectrum(s**2, rank, discard=eps)
    u = u[:, :keep].copy()
    s = s[:keep]
    u = numpy.dot(q, u)
    return u, s

