#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Jonathan Esterhazy <jonathan.esterhazy at gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# HDP inference code is adapted from the onlinehdp.py script by
# Chong Wang (chongw at cs.princeton.edu).
# http://www.cs.princeton.edu/~chongw/software/onlinehdp.tar.gz
#


"""This module encapsulates functionality for the online Hierarchical Dirichlet Process algorithm.

It allows both model estimation from a training corpus and inference of topic
distribution on new, unseen documents.

The core estimation code is directly adapted from the `onlinelhdp.py` script
by C. Wang see
**Wang, Paisley, Blei: Online Variational Inference for the Hierarchical Dirichlet
Process, JMLR (2011).**

http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf

The algorithm:

  * is **streamed**: training documents come in sequentially, no random access,
  * runs in **constant memory** w.r.t. the number of documents: size of the
    training corpus does not affect memory footprint

How to use :class:`~gensim.models.hdpmodel.HdpModel`
----------------------------------------------------------------


#. Run :class:`~gensim.models.hdpmodel.HdpModel` ::

    >>> from gensim.test.utils import common_corpus,common_dictionary
    >>> from gensim.models import hdpmodel
    >>>
    >>> hdp = HdpModel(common_corpus, common_dictionary)

#. You can then infer topic distributions on new, unseen documents, with

    >>> doc_hdp = hdp[doc_bow]

#. To print 20 topics with top 10 most probable words.

    >>> hdp.print_topics(num_topics=20, num_words=10)

#. The model can be updated (trained) with new documents via

    >>> hdp.update(other_corpus)

"""

from __future__ import with_statement

import logging
import time
import warnings

import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from six.moves import xrange

from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation
from gensim.models import basemodel, ldamodel

logger = logging.getLogger(__name__)

meanchangethresh = 0.00001
rhot_bound = 0.0


def expect_log_sticks(sticks):
    """For stick-breaking hdp, return the E[log(sticks)].

    Parameters
    ----------
    sticks : numpy.ndarray
        Array of values for stick.

    Returns
    -------
    numpy.ndarray
        Computed Elogsticks value.

    """
    dig_sum = psi(np.sum(sticks, 0))
    ElogW = psi(sticks[0]) - dig_sum
    Elog1_W = psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0: n - 1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks


def lda_e_step(doc_word_ids, doc_word_counts, alpha, beta, max_iter=100):
    """Performs EM-iteration on a single document for calculation of likelihood for a maximum iteration of `max_iter`.

    Parameters
    ----------
    doc_word_ids : int
        Id of corresponding words in a document.
    doc_word_counts : int
        Count of words in a single document.
    alpha : numpy.ndarray
        Lda equivalent value of alpha.
    beta : numpy.ndarray
        Lda equivalent value of beta.
    max_iter : int, optional
        Maximum number of times the expectation will be maximised.

    Returns
    -------
    tuple of numpy.ndarrays
        Returns a tuple of (likelihood,gamma).

    """
    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    betad = beta[:, doc_word_ids]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(doc_word_counts)
    for _ in xrange(max_iter):
        lastgamma = gamma

        gamma = alpha + expElogtheta * np.dot(counts / phinorm, betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))

    return (likelihood, gamma)


class SuffStats(object):
    """Stores suff stats for document(s)."""
    def __init__(self, T, Wt, Dt):
        """Initialises the suff stats for document(s) in the corpus.

        Parameters
        ----------
        T : int
            Top level truncation level.
        Wt : int
            Length of words in the documents.
        Dt : int
            chunk size.

        """
        self.m_chunksize = Dt
        self.m_var_sticks_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros((T, Wt))

    def set_zero(self):
        """Fill the sticks and beta array with 0 scalar value."""
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)


class HdpModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """The constructor estimates Hierachical Dirichlet Process model parameters based on a training corpus.

    Attributes
    ----------
    lda_alpha : numpy.ndarray
        Lda equivalent value of alpha.
    lda_beta : numpy.ndarray
        Lda equivalent value of beta.
    m_D : int
        Number of documents in the corpus.
    m_Elogbeta : numpy.ndarray:
        Stores value of dirchlet excpectation, i.e., Computed
        :math:`E[log \\theta]` for a vector :math:`\\theta \sim Dir(\\alpha)`.
    m_lambda : numpy.ndarray or scalar
        Drawn samples from the parameterized gamma distribution.
    m_lambda_sum : numpy.ndarray or scalar
        An array with the same shape as m_lambda, with the specified axis (1) removed.
    m_num_docs_processed : int
        Number of documents finished processing.This is incremented in size of chunks.
    m_r : list
        Acts as normaliser in lazy updation of lambda attribute.
    m_rhot : float
        Assigns weight to the information obtained from the mini-chunk and its value it between 0 and 1.
    m_status_up_to_date : bool
        Flag to indicate whether lambda and Elogbeta have been updated(T) or not(F).
    m_timestamp : numpy.ndarray
        Helps to keep track and perform lazy updates on lambda.
    m_updatect : int
        Keeps track of current time and is incremented everytime
        :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda()` is called.
    m_var_sticks : numpy.ndarray
        Array of values for stick.
    m_varphi_ss : numpy.ndarray
        Used to Update top level sticks.
    m_W : int
        Length of dictionary for the input corpus.

    """

    def __init__(self, corpus, id2word, max_chunks=None, max_time=None,
                 chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1,
                 gamma=1, eta=0.01, scale=1.0, var_converge=0.0001,
                 outputdir=None, random_state=None):
        """Fully initialises the hdp model.

        Parameters
        ----------
        corpus : list of list of tuple of ints; [ [ (int,int) ]]
            Corpus of input dataset on which the model will be trained.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`
            Dictionary for the input corpus.
        max_chunks : None, optional
            Upper bound on how many chunks to process.It wraps around corpus beginning in another corpus pass,
            if there are not enough chunks in the corpus
        max_time : None, optional
            Upper bound on time(in seconds) for which model will be trained.
        chunksize : int, optional
            Tells the number of documents to process at a time.
        kappa : float, optional
            Learning rate
        tau : float, optional
            Slow down parameter
        K : int, optional
            Second level truncation level
        T : int, optional
            Top level truncation level
        alpha : int, optional
            Second level concentration
        gamma : int, optional
            First level concentration
        eta : float, optional
            The topic Dirichlet
        scale : float, optional
            Weights information from the mini-chunk of corpus to calculate rhot.
        var_converge : float, optional
            Lower bound on the right side of convergence. Used when updating variational parameters for a
            single document.
        outputdir : str, optional
            Stores topic and options information in the specified directory.
        random_state : :class:`~np.random.RandomState`, optional
            Adds a little random jitter to randomize results around same alpha when trying to fetch a closest
            corrsponding lda model from :meth:`~gensim.models.hdpmodel.HdpModel.suggested_lda_model()`

        """
        self.corpus = corpus
        self.id2word = id2word
        self.chunksize = chunksize
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.outputdir = outputdir

        self.random_state = utils.get_random_state(random_state)

        self.lda_alpha = None
        self.lda_beta = None

        self.m_W = len(id2word)
        self.m_D = 0
        if corpus:
            self.m_D = len(corpus)

        self.m_T = T
        self.m_K = K
        self.m_alpha = alpha
        self.m_gamma = gamma

        self.m_var_sticks = np.zeros((2, T - 1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = range(T - 1, 0, -1)
        self.m_varphi_ss = np.zeros(T)

        self.m_lambda = self.random_state.gamma(1.0, 1.0, (T, self.m_W)) * self.m_D * 100 / (T * self.m_W) - eta
        self.m_eta = eta
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0
        self.m_status_up_to_date = True
        self.m_num_docs_processed = 0

        self.m_timestamp = np.zeros(self.m_W, dtype=int)
        self.m_r = [0]
        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)

        self.m_var_converge = var_converge

        if self.outputdir:
            self.save_options()

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.update(corpus)

    def inference(self, chunk):
        """Infers the gamma value on a trained corpus.

        Parameters
        ----------
        chunk : list of tuple of ints; [ [ (int,int) ]]
            Bag of words representation for a corpus.

        Returns
        -------
        numpy.ndarray
            gamma value.

        Raises
        ------
        RuntimeError
            Need to train model first to do inference.

        """
        if self.lda_alpha is None or self.lda_beta is None:
            raise RuntimeError("model must be trained to perform inference")
        chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        gamma = np.zeros((len(chunk), self.lda_beta.shape[0]))
        for d, doc in enumerate(chunk):
            if not doc:  # leave gamma at zero for empty documents
                continue
            ids, counts = zip(*doc)
            _, gammad = lda_e_step(ids, counts, self.lda_alpha, self.lda_beta)
            gamma[d, :] = gammad
        return gamma

    def __getitem__(self, bow, eps=0.01):
        """Accessor method for generating topic distribution of given document.

        Parameters
        ----------
        bow : sequence of list of tuple of ints; [ (int,int) ]
            Bag-of-words representation of the document to get topics for.
        eps : float, optional
            Ignore topics with probability below `eps`.

        Returns
        -------
            topic distribution for the given document `bow`, as a list of `(topic_id, topic_probability)` 2-tuples.

        """
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(corpus)

        gamma = self.inference([bow])[0]
        topic_dist = gamma / sum(gamma) if sum(gamma) != 0 else []
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist) if topicvalue >= eps]

    def update(self, corpus):
        """Train the model with new documents, by EM-iterating over `corpus` until
        any of the conditions is satisfied(time limit expired,chunk limit reached or whole corpus processed ).

        Parameters
        ----------
        corpus : list of list of tuple of ints; [ [ (int,int) ]]
            The corpus on which Hdp model will be updated.

        """
        save_freq = max(1, int(10000 / self.chunksize))  # save every 10k docs, roughly
        chunks_processed = 0
        start_time = time.clock()

        while True:
            for chunk in utils.grouper(corpus, self.chunksize):
                self.update_chunk(chunk)
                self.m_num_docs_processed += len(chunk)
                chunks_processed += 1

                if self.update_finished(start_time, chunks_processed, self.m_num_docs_processed):
                    self.update_expectations()
                    alpha, beta = self.hdp_to_lda()
                    self.lda_alpha = alpha
                    self.lda_beta = beta
                    self.print_topics(20)
                    if self.outputdir:
                        self.save_topics()
                    return

                elif chunks_processed % save_freq == 0:
                    self.update_expectations()
                    # self.save_topics(self.m_num_docs_processed)
                    self.print_topics(20)
                    logger.info('PROGRESS: finished document %i of %i', self.m_num_docs_processed, self.m_D)

    def update_finished(self, start_time, chunks_processed, docs_processed):
        """Flag to determine whether the Hdp model has been updated with the new corpus or not.

        Parameters
        ----------
        start_time : float
            Indicates the current processor time as a floating point number expressed in seconds. The resolution is
            typically better on Windows than on Unix by one microsecond due to differing implementation of underlying
            function calls.
        chunks_processed : int
            Indicates progress of the update in terms of the number of chunks processed.
        docs_processed : int
            Indicates number of documents finished processing.This is incremented in size of chunks.

        Returns
        -------
        bool
            True if Hdp model is updated, False otherwise.

        """
        return (
            # chunk limit reached
            (self.max_chunks and chunks_processed == self.max_chunks) or

            # time limit reached
            (self.max_time and time.clock() - start_time > self.max_time) or

            # no limits and whole corpus has been processed once
            (not self.max_chunks and not self.max_time and docs_processed >= self.m_D))

    def update_chunk(self, chunk, update=True, opt_o=True):
        """Performs lazy update on necessary columns of lambda and variational inference for documents in the chunk.

        Parameters
        ----------
        chunk : list of list of tuple of ints; [ [ (int,int) ]]
            The chunk of corpus on which Hdp model will be updated.
        update : bool, optional
            Flag to determine whether to update lambda(T) or not (F).
        opt_o : bool, optional
            Passed as argument to :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda()` to determine whether
            the topics need to be ordered(T) or not(F).

        Returns
        -------
        tuple of (float,int)
            A tuple of likelihood and sum of all the word counts from each document in the corpus.

        """
        # Find the unique words in this chunk...
        unique_words = dict()
        word_list = []
        for doc in chunk:
            for word_id, _ in doc:
                if word_id not in unique_words:
                    unique_words[word_id] = len(unique_words)
                    word_list.append(word_id)

        wt = len(word_list)  # length of words in these documents

        # ...and do the lazy updates on the necessary columns of lambda
        rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
        self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
        self.m_Elogbeta[:, word_list] = \
            psi(self.m_eta + self.m_lambda[:, word_list]) - \
            psi(self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])

        ss = SuffStats(self.m_T, wt, len(chunk))

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks)  # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        for doc in chunk:
            if len(doc) > 0:
                doc_word_ids, doc_word_counts = zip(*doc)
                doc_score = self.doc_e_step(
                    ss, Elogsticks_1st,
                    unique_words, doc_word_ids,
                    doc_word_counts, self.m_var_converge
                )
                count += sum(doc_word_counts)
                score += doc_score

        if update:
            self.update_lambda(ss, word_list, opt_o)

        return score, count

    def doc_e_step(self, ss, Elogsticks_1st, unique_words, doc_word_ids, doc_word_counts, var_converge):
        """Performs e step for a single doc.

        Parameters
        ----------
        ss : :class:`~gensim.models.hdpmodel.SuffStats`
            Suffstats for all document(s) in the chunk.
        Elogsticks_1st : numpy.ndarray
            Computed Elogsticks value by stick-breaking process.
        unique_words : int
            Number of unique words in the chunk.
        doc_word_ids : tuple of int
            Word ids of for a single document.
        doc_word_counts : tuple of int
            Word counts of all words in a single document.
        var_converge : float, optional
            Lower bound on the right side of convergence. Used when updating variational parameters for a
            single document.

        Returns
        -------
        float
            Computed value of likelihood for a single document.

        """
        chunkids = [unique_words[id] for id in doc_word_ids]

        Elogbeta_doc = self.m_Elogbeta[:, doc_word_ids]
        # very similar to the hdp equations
        v = np.zeros((2, self.m_K - 1))
        v[0] = 1.0
        v[1] = self.m_alpha

        # back to the uniform
        phi = np.ones((len(doc_word_ids), self.m_K)) * 1.0 / self.m_K

        likelihood = 0.0
        old_likelihood = -1e200
        converge = 1.0

        iter = 0
        max_iter = 100
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            # update variational parameters

            # var_phi
            if iter < 3:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc_word_counts).T)
                (log_var_phi, log_norm) = matutils.ret_log_normalize_vec(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc_word_counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = matutils.ret_log_normalize_vec(var_phi)
                var_phi = np.exp(log_var_phi)

            # phi
            if iter < 3:
                phi = np.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = matutils.ret_log_normalize_vec(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd  # noqa:F821
                (log_phi, log_norm) = matutils.ret_log_normalize_vec(phi)
                phi = np.exp(log_phi)

            # v
            phi_all = phi * np.array(doc_word_counts)[:, np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:, :self.m_K - 1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:, 1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K - 1) * log_alpha
            dig_sum = psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:, np.newaxis] - v) * (psi(v) - dig_sum))
            likelihood -= np.sum(gammaln(np.sum(v, 0))) - np.sum(gammaln(v))

            # Z part
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc_word_counts))

            converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                logger.warning('likelihood is decreasing!')

            iter += 1

        # update the suff_stat ss
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += np.sum(var_phi, 0)
        ss.m_var_beta_ss[:, chunkids] += np.dot(var_phi.T, phi.T * doc_word_counts)

        return likelihood

    def update_lambda(self, sstats, word_list, opt_o):
        """Updates appropriate columns of lambda and top level sticks based on documents.

        Parameters
        ----------
        sstats : :class:`~gensim.models.hdpmodel.SuffStats`
            Suffstats for all document(s) in the chunk.
        word_list : list of int
            Contains word id of all the unique words in the chunk of documents on which update is being performed.
        opt_o : bool, optional
            Flag to determine whether to invoke a call to :meth:`~gensim.models.hdpmodel.HdpModel.optimal_ordering()`.
            This decides whether the topics need to be ordered(T) or not(F).

        """
        self.m_status_up_to_date = False
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-chunk.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound:
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, word_list] = \
            self.m_lambda[:, word_list] * (1 - rhot) + rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_chunksize
        self.m_lambda_sum = (1 - rhot) * self.m_lambda_sum + \
            rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_chunksize

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect
        self.m_r.append(self.m_r[-1] + np.log(1 - rhot))

        self.m_varphi_ss = \
            (1.0 - rhot) * self.m_varphi_ss + rhot * sstats.m_var_sticks_ss * self.m_D / sstats.m_chunksize

        if opt_o:
            self.optimal_ordering()

        # update top level sticks
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T - 1] + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

    def optimal_ordering(self):
        """Performs ordering on the topics."""
        idx = matutils.argsort(self.m_lambda_sum, reverse=True)
        self.m_varphi_ss = self.m_varphi_ss[idx]
        self.m_lambda = self.m_lambda[idx, :]
        self.m_lambda_sum = self.m_lambda_sum[idx]
        self.m_Elogbeta = self.m_Elogbeta[idx, :]

    def update_expectations(self):
        """Since we're doing lazy updates on lambda, at any given moment the current state of lambda may not be
        accurate. This function updates all of the elements of lambda and Elogbeta so that if (for example) we want to
        print out the topics we've learned we'll get the correct behavior.

        """
        for w in xrange(self.m_W):
            self.m_lambda[:, w] *= np.exp(self.m_r[-1] - self.m_r[self.m_timestamp[w]])
        self.m_Elogbeta = \
            psi(self.m_eta + self.m_lambda) - psi(self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])

        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def show_topic(self, topic_id, topn=20, log=False, formatted=False, num_words=None):
        """Print the `num_words` most probable words for topic `topic_id`.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        log : bool, optional
            Logs a message with level INFO on the logger object.
        formatted : bool, optional
            Flag to determine whether to return the topics as a list of strings(T), or as lists of
            (weight, word) pairs(F).
        num_words : int, optional
            Number of most probable words to show from given `topic_id`.

         .. note:: The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Topic terms output displayed whose format depends on `formatted` parameter.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn(
                "The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead."
            )
            topn = num_words

        if not self.m_status_up_to_date:
            self.update_expectations()
        betas = self.m_lambda + self.m_eta
        hdp_formatter = HdpTopicFormatter(self.id2word, betas)
        return hdp_formatter.show_topic(topic_id, topn, log, formatted)

    def get_topics(self):
        """Returns the term topic matrix learned during inference.

        Returns
        -------
        np.ndarray
            `num_topics` x `vocabulary_size` array of floats

        """
        topics = self.m_lambda + self.m_eta
        return topics / topics.sum(axis=1)[:, None]

    def show_topics(self, num_topics=20, num_words=20, log=False, formatted=True):
        """Print the `num_words` most probable words for `num_topics` number of topics.
        Set `num_topics=-1` to print all topics.Set `formatted=True` to return the topics as a list of strings, or
        `False` as lists of (word, weight) pairs.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics for which most probable `num_words` words will be fetched.
        num_words :  int, optional
            Number of most probable words to show from `num_topics` number of topics.
        log : bool, optional
            Logs a message with level INFO on the logger object.
        formatted : bool, optional
            Flag to determine whether to return the topics as a list of strings(T), or as lists of
            (word, weight) pairs(F).

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Output format for topic terms depends on the value of `formatted` parameter.

        """
        if not self.m_status_up_to_date:
            self.update_expectations()
        betas = self.m_lambda + self.m_eta
        hdp_formatter = HdpTopicFormatter(self.id2word, betas)
        return hdp_formatter.show_topics(num_topics, num_words, log, formatted)

    def save_topics(self, doc_count=None):
        """Saves all the topics discovered.

        .. note:: This is a legacy method; use `self.save()` instead.

        Parameters
        ----------
        doc_count : int, optional
            Indicates number of documents finished processing and are to be saved.

        """
        if not self.outputdir:
            logger.error("cannot store topics without having specified an output directory")

        if doc_count is None:
            fname = 'final'
        else:
            fname = 'doc-%i' % doc_count
        fname = '%s/%s.topics' % (self.outputdir, fname)
        logger.info("saving topics to %s", fname)
        betas = self.m_lambda + self.m_eta
        np.savetxt(fname, betas)

    def save_options(self):
        """Writes all the values of the attributes for the current model in options.dat file.

        .. note:: This is a legacy method; use `self.save()` instead.

        """
        if not self.outputdir:
            logger.error("cannot store options without having specified an output directory")
            return
        fname = '%s/options.dat' % self.outputdir
        with utils.smart_open(fname, 'wb') as fout:
            fout.write('tau: %s\n' % str(self.m_tau - 1))
            fout.write('chunksize: %s\n' % str(self.chunksize))
            fout.write('var_converge: %s\n' % str(self.m_var_converge))
            fout.write('D: %s\n' % str(self.m_D))
            fout.write('K: %s\n' % str(self.m_K))
            fout.write('T: %s\n' % str(self.m_T))
            fout.write('W: %s\n' % str(self.m_W))
            fout.write('alpha: %s\n' % str(self.m_alpha))
            fout.write('kappa: %s\n' % str(self.m_kappa))
            fout.write('eta: %s\n' % str(self.m_eta))
            fout.write('gamma: %s\n' % str(self.m_gamma))

    def hdp_to_lda(self):
        """Only returns corresponding alpha, beta values of a LDA almost equivalent to current HDP.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple of numpy arrays of alpha and beta.

        """
        # alpha
        sticks = self.m_var_sticks[0] / (self.m_var_sticks[0] + self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in xrange(0, self.m_T - 1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T - 1] = left
        alpha *= self.m_alpha

        # beta
        beta = (self.m_lambda + self.m_eta) / (self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])

        return alpha, beta

    def suggested_lda_model(self):
        """Returns a trained ldamodel object which is closest to the current hdp model.The num_topics is m_T
        (default is 150) so as to preserve the matrice shapes when we assign alpha and beta.

        Returns
        -------
        :class:`~gensim.models.ldamodel.LdaModel`
            Closest corresponding LdaModel to current HdpModel.

        """
        alpha, beta = self.hdp_to_lda()
        ldam = ldamodel.LdaModel(
            num_topics=self.m_T, alpha=alpha, id2word=self.id2word, random_state=self.random_state, dtype=np.float64
        )
        ldam.expElogbeta[:] = beta
        return ldam

    def evaluate_test_corpus(self, corpus):
        """Evaluates the model on test corpus.

        Parameters
        ----------
        corpus : list of list of tuple of ints; [ [ (int,int) ]]
            The corpus on which Hdp model will be tested.

        Returns
        -------
        float
            The value of total likelihood obtained by evaluating the model for all documents in the test corpus.

        """
        logger.info('TEST: evaluating test corpus')
        if self.lda_alpha is None or self.lda_beta is None:
            self.lda_alpha, self.lda_beta = self.hdp_to_lda()
        score = 0.0
        total_words = 0
        for i, doc in enumerate(corpus):
            if len(doc) > 0:
                doc_word_ids, doc_word_counts = zip(*doc)
                likelihood, gamma = lda_e_step(doc_word_ids, doc_word_counts, self.lda_alpha, self.lda_beta)
                theta = gamma / np.sum(gamma)
                lda_betad = self.lda_beta[:, doc_word_ids]
                log_predicts = np.log(np.dot(theta, lda_betad))
                doc_score = sum(log_predicts) / len(doc)
                logger.info('TEST: %6d    %.5f', i, doc_score)
                score += likelihood
                total_words += sum(doc_word_counts)
        logger.info(
            "TEST: average score: %.5f, total score: %.5f,  test docs: %d",
            score / total_words, score, len(corpus)
        )
        return score


class HdpTopicFormatter(object):
    """Helper class to format the output of topics and most probable words for display."""

    (STYLE_GENSIM, STYLE_PRETTY) = (1, 2)

    def __init__(self, dictionary=None, topic_data=None, topic_file=None, style=None):
        """Initialises the :class:`gensim.models.hdpmodel.HdpTopicFormatter` and stores topic data in sorted order.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`,optional
            Dictionary for the input corpus.
        topic_data : numpy.ndarray, optional
            The term topic matrix.
        topic_file : file, str, or pathlib.Path
            File, filename, or generator to read. If the filename extension is .gz or .bz2, the file is first
            decompressed. Note that generators should return byte strings for Python 3k.
        style : bool, optional
            Flag to determine whether to return the topics as a list of strings(T), or as lists of (word, weight)
            pairs(F).
        data: numpy.ndarray
            Sorted topic data in descending order of sum of probabilities for all words in corresponding topic.

        Raises
        ------
        ValueError
            Either no dictionary or no topic data.

        """
        if dictionary is None:
            raise ValueError('no dictionary!')

        if topic_data is not None:
            topics = topic_data
        elif topic_file is not None:
            topics = np.loadtxt('%s' % topic_file)
        else:
            raise ValueError('no topic data!')

        # sort topics
        topics_sums = np.sum(topics, axis=1)
        idx = matutils.argsort(topics_sums, reverse=True)
        self.data = topics[idx]

        self.dictionary = dictionary

        if style is None:
            style = self.STYLE_GENSIM

        self.style = style

    def print_topics(self, num_topics=10, num_words=10):
        """Gives the most probable `num_words` words from `num_topics` topics.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Output format for `num_words` words from `num_topics` topics depends on the value of `self.style` attribute.

        """
        return self.show_topics(num_topics, num_words, True)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Gives the most probable `num_words` words from `num_topics` topics.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.
        log : bool, optional
            Logs a message with level INFO on the logger object.
        formatted : bool, optional
            Flag to determine whether to return the topics as a list of strings(T), or as lists of
            (word, weight) pairs(F).

        Returns
        -------
        list of tuple of (int ,list of tuple of (unicode,numpy.float64) or list of str)
            Output format for terms from `num_topics` topics depends on the value of `self.style` attribute.

        """
        shown = []
        if num_topics < 0:
            num_topics = len(self.data)

        num_topics = min(num_topics, len(self.data))

        for k in xrange(num_topics):
            lambdak = list(self.data[k, :])
            lambdak = lambdak / sum(lambdak)

            temp = zip(lambdak, xrange(len(lambdak)))
            temp = sorted(temp, key=lambda x: x[0], reverse=True)

            topic_terms = self.show_topic_terms(temp, num_words)

            if formatted:
                topic = self.format_topic(k, topic_terms)

                # assuming we only output formatted topics
                if log:
                    logger.info(topic)
            else:
                topic = (k, topic_terms)
            shown.append(topic)

        return shown

    def print_topic(self, topic_id, topn=None, num_words=None):
        """Prints the `topn` most probable words from topic id `topic_id`.

        Note
        ----
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        num_words : int, optional
            Number of most probable words to show from given `topic_id`.

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Output format for terms from a single topic depends on the value of `formatted` parameter.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn(
                "The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead."
            )
            topn = num_words

        return self.show_topic(topic_id, topn, formatted=True)

    def show_topic(self, topic_id, topn=20, log=False, formatted=False, num_words=None,):
        """Gives the most probable `num_words` words for the id `topic_id`.

        Note
        ----
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        log : bool, optional
            Logs a message with level INFO on the logger object.
        formatted : bool, optional
            Flag to determine whether to return the topics as a list of strings(T), or as lists of
            (word, weight) pairs(F).
        num_words : int, optional
            Number of most probable words to show from given `topic_id`.

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Output format for terms from a single topic depends on the value of `self.style` attribute.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn(
                "The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead."
            )
            topn = num_words

        lambdak = list(self.data[topic_id, :])
        lambdak = lambdak / sum(lambdak)

        temp = zip(lambdak, xrange(len(lambdak)))
        temp = sorted(temp, key=lambda x: x[0], reverse=True)

        topic_terms = self.show_topic_terms(temp, topn)

        if formatted:
            topic = self.format_topic(topic_id, topic_terms)

            # assuming we only output formatted topics
            if log:
                logger.info(topic)
        else:
            topic = (topic_id, topic_terms)

        # we only return the topic_terms
        return topic[1]

    def show_topic_terms(self, topic_data, num_words):
        """Gives the topic terms along with their probabilities for a single topic data.

        Parameters
        ----------
        topic_data : list of tuple of (unicode,numpy.float64)
            Contains probabilities for each word id belonging to a single topic.
        num_words : int
            Number of words for which probabilities are to be extracted from the given single topic data.

        Returns
        -------
        list of tuple of (unicode,numpy.float64)
            A sequence of topic terms and their probabilities.

        """
        return [(self.dictionary[wid], weight) for (weight, wid) in topic_data[:num_words]]

    def format_topic(self, topic_id, topic_terms):
        """Formats the display for a single topic in two different ways.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topic_terms : list of tuple of (unicode,numpy.float64)
            Contains the most probable words from a single topic.

        Returns
        -------
        list of tuple of (unicode,numpy.float64) or list of str
            Output format for topic terms depends on the value of `self.style` attribute.

        """
        if self.STYLE_GENSIM == self.style:
            fmt = ' + '.join(['%.3f*%s' % (weight, word) for (word, weight) in topic_terms])
        else:
            fmt = '\n'.join(['    %20s    %.8f' % (word, weight) for (word, weight) in topic_terms])

        fmt = (topic_id, fmt)
        return fmt
