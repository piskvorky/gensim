#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
**For a faster implementation of LDA (parallelized for multicore machines), see** :mod:`gensim.models.ldamulticore`.

Latent Dirichlet Allocation (LDA) in Python.

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents. The model can also be updated with new documents
for online training.

The core estimation code is based on the `onlineldavb.py` script by M. Hoffman [1]_, see
**Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**

The algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM, and
* is **distributed**: makes use of a cluster of machines, if available, to
  speed up model estimation.

.. [1] http://www.cs.princeton.edu/~mdhoffma

"""
import logging
import numbers
import os

import numpy as np
import six
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from six.moves import xrange
from collections import defaultdict

from gensim import interfaces, utils, matutils
from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference
)
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback


logger = logging.getLogger('gensim.models.ldamodel')

# Epsilon (very small) values used by each expected data type instead of 0, to avoid Arithmetic Errors.
DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


def update_dir_prior(prior, N, logphat, rho):
    """Updates a given prior using Newton's method, described in
    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"
    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.

    Parameters
    ----------
    prior : list of float
        The prior for each possible outcome at the previous iteration (to be updated).
    N : int
        Number of observations.
    logphat : list of float
        Log probabilities for the current estimation, also called "observed sufficient statistics".
    rho : float
        Learning rate.

    Returns
    -------
    list of float
        The updated prior.

    """
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior


class LdaState(utils.SaveLoad):
    """Encapsulate information for distributed computation of :class:`~gensim.models.ldamodel.LdaModel` objects.

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.

    """

    def __init__(self, eta, shape, dtype=np.float32):
        """

        Parameters
        ----------
        eta : list of float
            The prior probabilities assigned to each term.
        shape : tuple of (int, int)
            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).
        dtype : type
            Overrides the numpy array default types.

        """
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        """Prepare the state for a new EM iteration (reset sufficient stats). """
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        """Merge the result of an E step from one node with that of another node (summing up sufficient statistics).

        The merging is trivial and after merging all cluster nodes, we have the
        exact same result as if the computation was run on a single node (no
        approximation).

        Parameters
        ----------
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.

        """
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted average for the sufficient statistics.

        The number of documents is strected in both state objects, so that they are of comparable magnitude.
        This procedure corresponds to the stochastic gradient update from
        `Hoffman et al. :"Online Learning for Latent Dirichlet Allocation"
        <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted sum for the sufficient statistics.

        In contrast to :meth:`~gensim.models.ldamodel.LdaState.blend`, the sufficient statistics are not scaled
        prior to aggregation.

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # merge the two matrices by summing
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        """Get the parameters of the posterior over the topics, also reffered to as "the topics".

        Returns
        -------
        list of float
            Parameters of the posterior probability over topics.

        """
        return self.eta + self.sstats

    def get_Elogbeta(self):
        """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        list of float
            Posterior probabilities for each topic.
        """
        return dirichlet_expectation(self.get_lambda())

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Overrides :class:`~gensim.utils.SaveLoad.load` by enforcing the `dtype` parameter
        to ensure backwards compatibility.

        Parameters
        ----------
        fname : str
            Path to file that contains the needed object.
        args
            Positional parameters to be propagated to class:`~gensim.utils.SaveLoad.load`
        kwargs
            Key-word parameters to be propagated to class:`~gensim.utils.SaveLoad.load`

        Returns
        -------
        :class:`~gensim.models.ldamodel.LdaState`
            The model loaded from the given file.

        """
        result = super(LdaState, cls).load(fname, *args, **kwargs)

        # dtype could be absent in old models
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64  # float64 was implicitly used before (cause it's default in numpy)
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)

        return result


class LdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """This model implements online latent dirichlet allocation as presented in
    `Hoffman et al. :"Online Learning for Latent Dirichlet Allocation" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.

    Examples
    -------
    #. Initialize a model using a gensim corpus.

    >>> from gensim.test.utils import common_corpus
    >>>
    >>> lda = LdaModel(common_corpus, num_topics=10)

    #.You can then infer topic distributions on new, unseen documents.

    >>> doc_bow = [(1, 0.3), (2, 0.1), (0, 0.09)]
    >>> doc_lda = lda[doc_bow]

    The model can be updated (trained) with new documents.

    >>> # In practice (corpus =/= initial training corpus), but we use the same here for simplicity.
    >>> other_corpus = common_corpus
    >>>
    >>> lda.update(other_corpus)

    Model persistency is achieved through its `load`/`save` methods.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                 random_state=None, ns_conf=None, minimum_phi_value=0.01,
                 per_word_topics=False, callbacks=None, dtype=np.float32):
        """ The constructor estimates Latent Dirichlet Allocation model parameters based
        on a training corpus.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`).
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : dict of (int, str)
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        distributed : bool, optional
            Whether distributed computing should be used to accerelate training.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        alpha : {np.ndarray, str}, optional
            Can be set to an 1D array of length equal to the number of expected topics that expresses
            our a-priori belief for the each topics' probability.
            Alternatively default prior selecting strategies can be employed by supplying a string:

                * 'asymmetric': Uses a fixed normalized assymetric prior of `1.0 / topicno`.
                * 'default': Learns an assymetric prior from the corpus.
        eta : {float, np.array, str}, optional
            A-priori belief on word probability, this can be:

                * scalar for a symmetric prior over topic/word probability,
                * vector of length num_words to denote an asymmetric user defined probability for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,
                * the string 'auto' to learn the asymmetric prior from the data.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to Kappa from
            `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to Tau_0 from `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        ns_conf : dict of (str, object), optional
            Key word parameters propagated to :func:`~gensim.utils.getNS` to get a Pyro4 Nameserved.
            Only used if `distributed` is set to True.
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        callbacks : list of :class:`~gensim.models.callbacks.Callback`
            Metric callbacks to log and visualize evaluation metrics of the model during training.
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.

        Examples
        --------

        #. Train an LDA model using a gensim corpus:

        >>> from gensim.test.utils import common_corpus
        >>> from gensim.corpora.dictionary import Dictionary
        >>>
        >>> lda = LdaModel(common_corpus, num_topics=10)  # train model
        >>>

        #. Query, or update the model using new, unseen documents:

        >>> other_texts = [
        ...     ['computer', 'time', 'graph'],
        ...     ['survey', 'response', 'eps'],
        ...     ['human', 'system', 'computer']
        ... ]
        >>> other_dictionary = Dictionary(other_texts)
        >>> other_corpus = [other_dictionary.doc2bow(text) for text in other_texts]

        >>> # Query the model on an unseen document
        >>> unseen_doc = other_corpus[0]
        >>> repr = lda[unseen_doc] # get topic probability distribution for a document
        >>>
        >>> # Update the model by incrementally training on the new corpus.
        >>> lda.update(other_corpus) # update the LDA model with additional documents
        >>> repr = lda[unseen_doc]

        #. A lot of parameters can be tuned to optimize training for your specific case:

        >>> lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)  # train asymmetric alpha from data

        """
        if dtype not in DTYPE_TO_EPS:
            raise ValueError(
                "Incorrect 'dtype', please choose one of {}".format(
                    ", ".join("numpy.{}".format(tp.__name__) for tp in sorted(DTYPE_TO_EPS))))

        self.dtype = dtype

        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

        self.random_state = utils.get_random_state(random_state)

        assert self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms), (
            "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # set up distributed environment if necessary
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError("auto-optimizing alpha not implemented in distributed LDA")
            # set up distributed version
            try:
                import Pyro4
                if ns_conf is None:
                    ns_conf = {}

                with utils.getNS(**ns_conf) as ns:
                    from gensim.models.lda_dispatcher import LDA_DISPATCHER_PREFIX
                    self.dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug("looking for dispatcher at %s" % str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(
                        id2word=self.id2word, num_topics=self.num_topics, chunksize=chunksize,
                        alpha=alpha, eta=eta, distributed=False
                    )
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info("using distributed version with %i workers", self.numworkers)
            except Exception as err:
                logger.error("failed to initialize distributed LDA (%s)", err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        # Initialize the variational distribution q(beta|lambda)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms), dtype=self.dtype)
        self.state.sstats[...] = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))

        # Check that we haven't accidentally fallen back to np.float64
        assert self.eta.dtype == self.dtype
        assert self.expElogbeta.dtype == self.dtype

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunks_as_numpy=use_numpy)

    def init_dir_prior(self, prior, name):
        """Initialize our prio belief for the Dirichlet distribution.

        Parameters
        ----------
        prior : {str, list of float, np.ndarray of float, float}
            A-priori belief on word probability. If `name` == 'eta' then the prior can be:

                * scalar for a symmetric prior over topic/word probability,
                * vector of length num_words to denote an asymmetric user defined probability for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,
                * the string 'auto' to learn the asymmetric prior from the data.

            If `name` == 'alpha', then the prior can be:

                * an 1D array of length equal to the number of expected topics,
                * 'asymmetric': Uses a fixed normalized assymetric prior of `1.0 / topicno`.
                * 'default': Learns an assymetric prior from the corpus.
        name : {'alpha', 'eta'}
            Whether the `prior` is parameterized by the alpha vector (1 parameter per topic)
            or by the eta (1 parameter per unique term in the vocabulary).
        """
        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'eta'")

        is_auto = False

        if isinstance(prior, six.string_types):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0 / self.num_topics)
                init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)], dtype=self.dtype)
            elif prior == 'asymmetric':
                init_prior = \
                    np.asarray([1.0 / (i + np.sqrt(prior_shape)) for i in xrange(prior_shape)], dtype=self.dtype)
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)], dtype=self.dtype)
                if name == 'alpha':
                    logger.info("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, np.number) or isinstance(prior, numbers.Real):
            init_prior = np.asarray([prior] * prior_shape, dtype=self.dtype)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto

    def __str__(self):
        """Get a string representation of the current object.

        Returns
        -------
        str
            Human readable representation of the most important model parameters.

        """
        return "LdaModel(num_terms=%s, num_topics=%s, decay=%s, chunksize=%s)" % (
            self.num_terms, self.num_topics, self.decay, self.chunksize
        )

    def sync_state(self):
        """Propagate the states topic probabilities to the inner object's attribute. """
        self.expElogbeta = np.exp(self.state.get_Elogbeta())
        assert self.expElogbeta.dtype == self.dtype

    def clear(self):
        """Clear the model's state to free some memory. Used in the distributed implementation."""
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        """Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
        for each document in the chunk.

        This function does not modify the model The whole input chunk of document is assumed to fit in RAM;
        chunking of a large corpus must be done earlier in the pipeline. Avoids computing the `phi` variational
        parameter directly using the optimization presented in
        `Lee, Seung: Algorithms for non-negative matrix factorization"
        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.

        Parameters
        ----------
        chunk : {list of list of (int, float), scipy.sparse.csc}
            The corpus chunk on which the inference step will be performed.
        collect_sstats : bool
            If set to True, also collect (and return) sufficient statistics needed to update the model's topic-word
            distributions.

        Returns
        -------
        tuple of (np.ndarray, {np.ndarray of float, None})
            The first element is always returned and it corresponds to the states gamma matrix. The second element is
            only returned if `collect_sstats` == True and corresponds to the sufficient statistics for the M step.

        """
        try:
            len(chunk)
        except TypeError:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = self.random_state.gamma(100., 1. / 100., (len(chunk), self.num_topics)).astype(self.dtype, copy=False)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype

        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            if len(doc) > 0 and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
                # make sure the term IDs are ints, otherwise np will get upset
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.array([cnt for _, cnt in doc], dtype=self.dtype)
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            # TODO treat zeros explicitly, instead of adding epsilon?
            eps = DTYPE_TO_EPS[self.dtype]
            phinorm = np.dot(expElogthetad, expElogbetad) + eps

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + eps
                # If gamma hasn't changed much, we're done.
                meanchange = mean_absolute_difference(gammad, lastgamma)
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break
            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations", converged, len(chunk), self.iterations)

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
            assert sstats.dtype == self.dtype

        assert gamma.dtype == self.dtype
        return gamma, sstats

    def do_estep(self, chunk, state=None):
        """Perform inference on a chunk of documents, and accumulate the collected sufficient statistics.

        Parameters
        ----------
        chunk : {list of list of (int, float), scipy.sparse.csc}
            The corpus chunk on which the inference step will be performed.
        state : :class:`~gensim.models.ldamodel.LdaState`, optional
            The state to be updated with the newly accumulated sufficient statistics. If none, the models
            `self.state` is updated.

        Returns
        -------
        np.ndarray of shape (`len(chunk)`, `self.num_topics`)
            Gamma parameters controlling the topic weights.

        """
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        state.numdocs += gamma.shape[0]  # avoids calling len(chunk) on a generator
        assert gamma.dtype == self.dtype
        return gamma

    def update_alpha(self, gammat, rho):
        """Update parameters for the Dirichlet prior on the per-document topic weights.

        Parameters
        ----------
        gammat : numpy.ndarray
            Previous topic weight parameters.
        rho : float
            Learning rate.

        Returns
        -------
        list of float
            Updated alpha parameters.

        """
        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N
        assert logphat.dtype == self.dtype

        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)
        logger.info("optimized alpha %s", list(self.alpha))

        assert self.alpha.dtype == self.dtype
        return self.alpha

    def update_eta(self, lambdat, rho):
        """Update parameters for the Dirichlet prior on the per-topic word weights.

        Parameters
        ----------
        lambdat : np.ndarray
            Previous lambda parameters.
        rho : float
            Learning rate.

        Returns
        -------
        list of float
            The updated eta parameters.

        """
        N = float(lambdat.shape[0])
        logphat = (sum(dirichlet_expectation(lambda_) for lambda_ in lambdat) / N).reshape((self.num_terms,))
        assert logphat.dtype == self.dtype

        self.eta = update_dir_prior(self.eta, N, logphat, rho)

        assert self.eta.dtype == self.dtype
        return self.eta

    def log_perplexity(self, chunk, total_docs=None):
        """Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.

        Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.

        Parameters
        ----------
        chunk : {list of list of (int, float), scipy.sparse.csc}
            The corpus chunk on which the inference step will be performed.
        total_docs : int
            Number of docs used for evaluation of the perplexity.

        Returns
        -------
        list of float
            The variational bound score calculated for each word.

        """
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words
        )
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until
        the maximum number of allowed iterations is reached. `corpus` must be an iterable.

        In distributed mode, the E step is distributed over a cluster of machines.

        Notes
        -----
        This update also supports updating an already trained model with new documents; the two models are then merged
        in proportion to the number of old vs. new documents. This feature is still experimental for non-stationary
        input streams. For stationary input (no topic drift in new documents), on the other hand, this equals the
        online update of `Matthew D. Hoffman, David M. Blei, Francis Bach:
        "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        and is guaranteed to converge for any `decay` in (0.5, 1.0). Additionally, for smaller corpus sizes, an
        increasing `offset` may be beneficial (see Table 1 in the same paper).




        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`) used to update the
            model.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to Kappa from
            `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to Tau_0 from `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
            chunks_as_numpy (bool): Whether each chunk passed to `.inference` should be a np
                array of not. np can in some settings turn the term IDs
                into floats, these will be converted back into integers in
                inference, which incurs a performance hit. For distributed
                computing it may be desirable to keep the chunks as np
                arrays.
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        chunks_as_numpy : bool
            Whether each chunk passed to the inference step should be a np.ndarray or not. Numpy can in some settings
            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a
            performance hit. For distributed computing it may be desirable to keep the chunks as np.ndarrays.

        """
        # use parameters given in constructor, unless user explicitly overrode them
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        try:
            lencorpus = len(corpus)
        except Exception:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return

        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)

        self.state.numdocs += lencorpus

        if update_every:
            updatetype = "online"
            if passes == 1:
                updatetype += " (single-pass)"
            else:
                updatetype += " (multi-pass)"
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s LDA training, %s topics, %i passes over "
            "the supplied corpus of %i documents, updating model once "
            "every %i documents, evaluating perplexity every %i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, passes, lencorpus,
            updateafter, evalafter, iterations,
            gamma_threshold
        )

        if updates_per_pass * passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(offset + pass_ + (self.num_updates / chunksize), -decay)

        if self.callbacks:
            # pass the list of input callbacks to Callback class
            callback = Callback(self.callbacks)
            callback.set_model(self)
            # initialize metrics list to store metric values after every epoch
            self.metrics = defaultdict(list)

        for pass_ in xrange(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
            dirty = False

            reallen = 0
            for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize, as_numpy=chunks_as_numpy)):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    self.log_perplexity(chunk, total_docs=lencorpus)

                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info(
                        "PROGRESS: pass %i, dispatching documents up to #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info(
                        "PROGRESS: pass %i, at document #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    gammat = self.do_estep(chunk, other)

                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())

                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        # distributed mode: wait for all workers to finish
                        logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other  # frees up memory

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
                    dirty = False
            # endfor single corpus iteration

            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

            # append current epoch's metric values
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for metric, value in current_metrics.items():
                    self.metrics[metric].append(value)

            if dirty:
                # finish any remaining updates
                if self.dispatcher:
                    # distributed mode: wait for all workers to finish
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other
                dirty = False

        # endfor entire corpus update

    def do_mstep(self, rho, other, extra_pass=False):
        """Maximization step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        Parameters
        ----------
        rho : float
            Learning rate.
        other : :class:`~gensim.models.ldamodel.LdaModel`
            The model whose sufficient statistics will be used to update the topics.
        extra_pass : bool
            Whether this step required an additional pass over the corpus.

        """
        logger.debug("updating topics")
        # update self with the new blend; also keep track of how much did
        # the topics change through this update, to assess convergence
        diff = np.log(self.expElogbeta)
        self.state.blend(rho, other)
        diff -= self.state.get_Elogbeta()
        self.sync_state()

        # print out some debug info at the end of each EM iteration
        self.print_topics(5)
        logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            # only update if this isn't an additional pass
            self.num_updates += other.numdocs

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        """Estimate the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)].

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`) used to estimate the
            variational bounds.
        gamma : np.ndarray
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        subsample_ratio : float
            Percentage of the whole corpus represented by the passed `corpus` argument (in case this was a sample).
            Set to 1.0 if the whole corpus was passed.This is used as a multiplicative factor to scale the likelihood
            appropriately.

        Returns
        -------
        list of float
            The variational bound score calculated for each document.

        """
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)

        for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype

            # E[log p(doc | theta, beta)]
            score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always rougly on the same scale.
        score *= subsample_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))

        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)

        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        return score

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Get a representation for some topics.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to be returned. Unlike LSA, there is no natural ordering between the topics in LDA.
            The returned topics subset of all topics is therefore arbitrary and may change between two LDA
            training runs.
        num_words : int, optional
            Number of words to be presented for each topic. These will be the most relevant words (assigned the highest
            probability for each topic).
        log : bool, optional
            Whether the output is also logged, besides being returned.
        formatted : bool, optional
            Whether the topic representations should be formatted as strings. If False, they are returned as
            2 tuples of (word, probability).

        Returns
        -------
        list of {str, tuple of (str, float)}
            a list of topics, each represented either as a string (when `formatted` == True) or word-probability
            pairs.

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * self.random_state.rand(len(self.alpha))
            # random_state.rand returns float64, but converting back to dtype won't speed up anything

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

        shown = []

        topic = self.state.get_lambda()
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, num_words, reverse=True)
            topic_ = [(self.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join(['%.3f*"%s"' % (v, k) for k, v in topic_])

            shown.append((i, topic_))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic_)

        return shown

    def show_topic(self, topicid, topn=10):
        """Get the representation for a single topic. Words here are the actual strings, in constrast to
        :meth:`~gensim.models.ldamodel.LdaModel.get_topic_terms` that represents words by their vocabulary ID.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (str, float)
            Word - probability pairs for the most relevant words generated by the topic.

        """
        return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]

    def get_topics(self):
        """Get the term-topic matrix learned during inference.


        Returns
        -------
        np.ndarray of float of shape (`num_topics`, `vocabulary_size`)
            The probability for each word in each topic.

        """
        topics = self.state.get_lambda()
        return topics / topics.sum(axis=1)[:, None]

    def get_topic_terms(self, topicid, topn=10):
        """Get the representation for a single topic. Words the integer IDs, in constrast to
        :meth:`~gensim.models.ldamodel.LdaModel.show_topic` that represents words by the actual strings.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (int, float)
            Word ID - probability pairs for the most relevant words generated by the topic.

        """
        topic = self.get_topics()[topicid]
        topic = topic / topic.sum()  # normalize to probability distribution
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def top_topics(self, corpus=None, texts=None, dictionary=None, window_size=None,
                   coherence='u_mass', topn=20, processes=-1):
        """Get the topics with the highest coherence score the coherence for each topic.

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Corpus in BoW format.
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
            probability estimator .
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping of id word to create corpus.
            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
        window_size : int, optional
            Is the size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        processes : int, optional
            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
            num_cpus - 1.

        Returns
        -------
        list of (list of (int, str), float)
            Each element in the list is a pair of a topic representation and its coherence score. Topic representations
            are distributions of words, represented as a list of pairs of word IDs and their probabilities.

        """
        cm = CoherenceModel(
            model=self, corpus=corpus, texts=texts, dictionary=dictionary,
            window_size=window_size, coherence=coherence, topn=topn,
            processes=processes
        )
        coherence_scores = cm.get_coherence_per_topic()

        str_topics = []
        for topic in self.get_topics():  # topic = array of vocab_size floats, one per term
            bestn = matutils.argsort(topic, topn=topn, reverse=True)  # top terms for topic
            beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]  # membership, token
            str_topics.append(beststr)  # list of topn (float membership, token) tuples

        scored_topics = zip(str_topics, coherence_scores)
        return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """Get the topic distribution for the given document.

        Parameters
        ----------
        bow : corpus : list of (int, float)
            The document in BOW format.
        minimum_probability : float
            Topics with an assigned probability lower than this threshold will be discarded.
        minimum_phi_value : float
            f `per_word_topics` is True, this represents a lower bound on the term probabilities that are included.
             If set to None, a value of 1e-8 is used to prevent 0s.
        per_word_topics : bool
            If True, this function will also return two extra lists as explained in the "Returns" section.

        Returns
        -------
        list of (int, float)
            Topic distribution for the whole document. Each element in the list is a pair of a topic's id, and
            the probability that was assigned to it.
        list of (int, list of (int, float), optional
            Most probable topics per word. Each element in the list is a pair of a word's id, and a list of
            topics sorted by their relevance to this word. Only returned if `per_word_topics` was set to True.
        list of (int, list of float), optional
            Phi relevance values, multipled by the feature length, for each word-topic combination.
            Each element in the list is a pair of a word's id and a list of the phi values between this word and
            each topic. Only returned if `per_word_topics` was set to True.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        minimum_phi_value = max(minimum_phi_value, 1e-8)  # never allow zero values in sparse output

        # if the input vector is a corpus, return a transformed corpus
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            kwargs = dict(
                per_word_topics=per_word_topics,
                minimum_probability=minimum_probability,
                minimum_phi_value=minimum_phi_value
            )
            return self._apply(corpus, **kwargs)

        gamma, phis = self.inference([bow], collect_sstats=per_word_topics)
        topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution

        document_topics = [
            (topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
            if topicvalue >= minimum_probability
        ]

        if not per_word_topics:
            return document_topics

        word_topic = []  # contains word and corresponding topic
        word_phi = []  # contains word and phi values
        for word_type, weight in bow:
            phi_values = []  # contains (phi_value, topic) pairing to later be sorted
            phi_topic = []  # contains topic and corresponding phi value to be returned 'raw' to user
            for topic_id in range(0, self.num_topics):
                if phis[topic_id][word_type] >= minimum_phi_value:
                    # appends phi values for each topic for that word
                    # these phi values are scaled by feature length
                    phi_values.append((phis[topic_id][word_type], topic_id))
                    phi_topic.append((topic_id, phis[topic_id][word_type]))

            # list with ({word_id => [(topic_0, phi_value), (topic_1, phi_value) ...]).
            word_phi.append((word_type, phi_topic))
            # sorts the topics based on most likely topic
            # returns a list like ({word_id => [topic_id_most_probable, topic_id_second_most_probable, ...]).
            sorted_phi_values = sorted(phi_values, reverse=True)
            topics_sorted = [x[1] for x in sorted_phi_values]
            word_topic.append((word_type, topics_sorted))

        return document_topics, word_topic, word_phi  # returns 2-tuple

    def get_term_topics(self, word_id, minimum_probability=None):
        """Get the most relevant topics to the given word.

        Parameters
        ----------
        word_id : int
            The word for which the topic distribution will be computed.
        minimum_probability : float
            Topics with an assigned probability below this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            The relevant topics represented as pairs of their ID and their assigned probability, sorted
            by relevance to the given word.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]

        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[topic_id][word_id] >= minimum_probability:
                values.append((topic_id, self.expElogbeta[topic_id][word_id]))

        return values

    def diff(self, other, distance="kullback_leibler", num_words=100,
             n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        """Calculate the difference in topic distributions extracted between two trained models (one of them is `self`).


        Notes
        -----
        The difference calculation does not take into account the underlying `dtype`s used in the models.

        Parameters
        ----------
        other : :class:`~gensim.models.ldamodel.LdaModel`
            The model which will be compared against the current object.
        distance : {'kullback_leibler', 'hellinger', 'jaccard', 'jensen_shannon'}
            The distance metric to calculate the difference with.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.

        Returns
        -------
        np.ndarray of shape (`self.num_topics`, `other.num_topics`)
            A difference matrix . Each element corresponds to the difference between the two topics.
        np.ndarray of shape (`self.num_topics`, `other_model.num_topics`, 2), optional
            Annotation matrix where for each pair we include the word from the intersection of the two topics,
            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.

        Examples
        --------

        #. Get the differences between each pair of topics inferred by two models.
        >>> from gensim.models.ldamulticore import LdaMulticore
        >>> from gensim.test.utils import datapath
        >>>
        >>> m1, m2 = LdaMulticore.load(datapath("lda_3_0_1_model")), LdaMulticore.load(datapath("ldamodel_python_3_5"))
        >>> mdiff, annotation = m1.diff(m2)
        >>> topic_diff = mdiff # get matrix with difference for each topic pair from `m1` and `m2`

        """

        distances = {
            "kullback_leibler": kullback_leibler,
            "hellinger": hellinger,
            "jaccard": jaccard_distance,
            "jensen_shannon": jensen_shannon
        }

        if distance not in distances:
            valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
            raise ValueError("Incorrect distance, valid only {}".format(valid_keys))

        if not isinstance(other, self.__class__):
            raise ValueError("The parameter `other` must be of type `{}`".format(self.__name__))

        distance_func = distances[distance]
        d1, d2 = self.get_topics(), other.get_topics()
        t1_size, t2_size = d1.shape[0], d2.shape[0]
        annotation_terms = None

        fst_topics = [{w for (w, _) in self.show_topic(topic, topn=num_words)} for topic in xrange(t1_size)]
        snd_topics = [{w for (w, _) in other.show_topic(topic, topn=num_words)} for topic in xrange(t2_size)]

        if distance == "jaccard":
            d1, d2 = fst_topics, snd_topics

        if diagonal:
            assert t1_size == t2_size, \
                "Both input models should have same no. of topics, " \
                "as the diagonal will only be valid in a square matrix"
            # initialize z and annotation array
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            # initialize z and annotation matrix
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)

        # iterate over each cell in the initialized z and annotation
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]

            z[topic] = distance_func(d1[topic1], d2[topic2])
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])

                pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]

                annotation_terms[topic] = [pos_tokens, neg_tokens]

        if normed:
            if np.abs(np.max(z)) > 1e-8:
                z /= np.max(z)

        return z, annotation_terms

    def __getitem__(self, bow, eps=None):
        """Get the topic distribution for the given document.

        Wraps :meth:`~gensim.models.ldamodel.LdaModel.get_document_topics` to support an operator style call.
        Uses the model's current state (set using constructor arguments) for the additional arguments of the
        wrapper method.

        Parameters
        ---------
        bow : corpus : list of (int, float)
            The document in BOW format.
        eps : float
            Topics with an assigned probability lower than this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            Topic distribution for the given document. Each topic is represented as a pair of its ID and the probability
            assigned to it.

        """
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs):
        """Save the model to file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        Notes
        -----
        Do not save as a compressed file if you intend to load the file back with `mmap`.

        If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all np arrays using np, only
             those ones that exceed `sep_limit` set in `gensim.utils.SaveLoad.save`. The main
             concern here is the `alpha` array if for instance using `alpha='auto'`.

        Please refer to the wiki recipes section (goo.gl/qoje24) for an example on how to work around these issues.

        See Also
        --------
        :meth:`~gensim.models.ldamodel.LdaModel.load`

        Parameters
        ----------
        fname : str
            Path to the system file where the model will be persisted.
        ignore : tuple of str, optional
            The named attributes in the tuple will be left out of the pickled model. The reason why
            the internal `state` is ignored by default is that it uses its own serialisation rather than the one
            provided by this method.
        separately : {list of str, None}, optional
            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This avoids pickle memory errors and allows `mmap`'ing large arrays
            back on load efficiently. If list of str - this attributes will be stored in separate files,
            the automatic check is not performed in this case.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.

        """
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)
        # Save the dictionary separately if not in 'ignore'.
        if 'id2word' not in ignore:
            utils.pickle(self.id2word, utils.smart_extension(fname, '.id2word'))

        # make sure 'state', 'id2word' and 'dispatcher' are ignored from the pickled object, even if
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, six.string_types):
                ignore = [ignore]
            ignore = [e for e in ignore if e]  # make sure None and '' are not in the list
            ignore = list({'state', 'dispatcher', 'id2word'} | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']

        # make sure 'expElogbeta' and 'sstats' are ignored from the pickled object, even if
        # someone sets the separately list themselves.
        separately_explicit = ['expElogbeta', 'sstats']
        # Also add 'alpha' and 'eta' to separately list if they are set 'auto' or some
        # array manually.
        if (isinstance(self.alpha, six.string_types) and self.alpha == 'auto') or \
                (isinstance(self.alpha, np.ndarray) and len(self.alpha.shape) != 1):
            separately_explicit.append('alpha')
        if (isinstance(self.eta, six.string_types) and self.eta == 'auto') or \
                (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        # Merge separately_explicit with separately.
        if separately:
            if isinstance(separately, six.string_types):
                separately = [separately]
            separately = [e for e in separately if e]  # make sure None and '' are not in the list
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(LdaModel, self).save(fname, ignore=ignore, separately=separately, *args, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load a previously saved object from file.

        See Also
        --------
        :meth:`~gensim.models.ldamodel.LdaModel.save`

        Parameters
        ----------
        fname : str
            Path to the file where the model is stored.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.

        Examples
        --------
        #. Large arrays can be memmap'ed back as read-only (shared memory) by setting `mmap='r'`:
        >>> from gensim.test.utils import datapath
        >>>
        >>> fname = datapath("lda_3_0_1_model")
        >>> lda = LdaModel.load(fname, mmap='r')

        """
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LdaModel, cls).load(fname, *args, **kwargs)

        # check if `random_state` attribute has been set after main pickle load
        # if set -> the model to be loaded was saved using a >= 0.13.2 version of Gensim
        # if not set -> the model to be loaded was saved using a < 0.13.2 version of Gensim,
        # so set `random_state` as the default value
        if not hasattr(result, 'random_state'):
            result.random_state = utils.get_random_state(None)  # using default value `get_random_state(None)`
            logging.warning("random_state not set so using default value")

        # dtype could be absent in old models
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64  # float64 was implicitly used before (cause it's default in numpy)
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)

        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = LdaState.load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)

        id2word_fname = utils.smart_extension(fname, '.id2word')
        # check if `id2word_fname` file is present on disk
        # if present -> the model to be loaded was saved using a >= 0.13.2 version of Gensim,
        # so set `result.id2word` using the `id2word_fname` file
        # if not present -> the model to be loaded was saved using a < 0.13.2 version of Gensim,
        # so `result.id2word` already set after the main pickle load
        if os.path.isfile(id2word_fname):
            try:
                result.id2word = utils.unpickle(id2word_fname)
            except Exception as e:
                logging.warning("failed to load id2word dictionary from %s: %s", id2word_fname, e)
        return result
