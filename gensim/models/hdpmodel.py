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


"""Module for `online Hierarchical Dirichlet Processing
<http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.

The core estimation code is directly adapted from the `blei-lab/online-hdp <https://github.com/blei-lab/online-hdp>`_
from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
<http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.

Examples
--------

Train :class:`~gensim.models.hdpmodel.HdpModel`

>>> from gensim.test.utils import common_corpus, common_dictionary
>>> from gensim.models import HdpModel
>>>
>>> hdp = HdpModel(common_corpus, common_dictionary)

You can then infer topic distributions on new, unseen documents, with

>>> unseen_document = [(1, 3.), (2, 4)]
>>> doc_hdp = hdp[unseen_document]

To print 20 topics with top 10 most probable words.

>>> topic_info = hdp.print_topics(num_topics=20, num_words=10)

The model can be updated (trained) with new documents via

>>> hdp.update([[(1, 2)], [(1, 1), (4, 5)]])

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

from gensim.utils import deprecated

logger = logging.getLogger(__name__)

meanchangethresh = 0.00001
rhot_bound = 0.0


def expect_log_sticks(sticks):
    """For stick-breaking hdp, get the :math:`\mathbb{E}[log(sticks)]`.

    Parameters
    ----------
    sticks : numpy.ndarray
        Array of values for stick.

    Returns
    -------
    numpy.ndarray
        Computed :math:`\mathbb{E}[log(sticks)]`.

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
    (numpy.ndarray, numpy.ndarray)
        Computed (:math:`likelihood`, :math:`\\gamma`).

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
        if meanchange < meanchangethresh:
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))

    return likelihood, gamma


class SuffStats(object):
    """Stores sufficient statistics for the current chunk of document(s) whenever Hdp model is updated with new corpus.
    These stats are used when updating lambda and top level sticks. The statistics include number of documents in the
    chunk, length of words in the documents and top level truncation level.

    """
    def __init__(self, T, Wt, Dt):
        """

        Parameters
        ----------
        T : int
            Top level truncation level.
        Wt : int
            Length of words in the documents.
        Dt : int
            Chunk size.

        """
        self.m_chunksize = Dt
        self.m_var_sticks_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros((T, Wt))

    def set_zero(self):
        """Fill the sticks and beta array with 0 scalar value."""
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)


class HdpModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """`Hierarchical Dirichlet Process model <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_

    Topic models promise to help summarize and organize large archives of texts that cannot be easily analyzed by hand.
    Hierarchical Dirichlet process (HDP) is a powerful mixed-membership model for the unsupervised analysis of grouped
    data. Unlike its finite counterpart, latent Dirichlet allocation, the HDP topic model infers the number of topics
    from the data. Here we have used Online HDP, which provides the speed of online variational Bayes with the modeling
    flexibility of the HDP. The idea behind Online variational Bayes in general is to optimize the variational
    objective function  with stochastic optimization.The challenge we face is that the existing coordinate ascent
    variational Bayes algorithms for the HDP require complicated approximation methods or numerical optimization. This
    model utilises stick breaking construction of Hdp which enables it to allow for coordinate-ascent variational Bayes
    without numerical approximation.

    **Stick breaking construction**

    To understand the HDP model we need to understand how it is modelled using the stick breaking construction. A very
    good analogy to understand the stick breaking construction is `chinese restaurant franchise
    <https://www.cs.princeton.edu/courses/archive/fall07/cos597C/scribe/20070921.pdf>`_.


    For this assume that there is a restaurant franchise (`corpus`) which has a large number of restaurants
    (`documents`, `j`) under it. They have a global menu of dishes (`topics`, :math:`\Phi_{k}`) which they serve.
    Also, a single dish (`topic`, :math:`\Phi_{k}`) is only served at a single table `t` for all the customers
    (`words`, :math:`\\theta_{j,i}`) who sit at that table.
    So, when a customer enters the restaurant he/she has the choice to make where he/she wants to sit.
    He/she can choose to sit at a table where some customers are already sitting , or he/she can choose to sit
    at a new table. Here the probability of choosing each option is not same.

    Now, in this the global menu of dishes correspond to the global atoms  :math:`\Phi_{k}`, and each restaurant
    correspond to a single document `j`. So the number of dishes served in a particular restaurant correspond to the
    number of topics in a particular document. And the number of people sitting at each table correspond to the number
    of words belonging to each topic inside the document `j`.

    Now, coming on to the stick breaking construction, the concept understood from the chinese restaurant franchise is
    easily carried over to the stick breaking construction for hdp (`"Figure 1" from "Online Variational Inference
    for the Hierarchical Dirichlet Process" <http://proceedings.mlr.press/v15/wang11a/wang11a.pdf>`_).

    A two level hierarchical dirichlet process is a collection of dirichlet processes :math:`G_{j}` , one for each
    group, which share a base distribution :math:`G_{0}`, which is also a dirichlet process. Also, all :math:`G_{j}`
    share the same set of atoms, :math:`\Phi_{k}`, and only the atom weights :math:`\pi _{jt}` differs.

    There will be multiple document-level atoms :math:`\psi_{jt}` which map to the same corpus-level atom
    :math:`\Phi_{k}`. Here, the :math:`\\beta` signify the weights given to each of the topics globally. Also, each
    factor :math:`\\theta_{j,i}` is distributed according to :math:`G_{j}`, i.e., it takes on the value of
    :math:`\Phi_{k}` with probability :math:`\pi _{jt}`. :math:`C_{j,t}` is an indicator variable whose value `k`
    signifies the index of :math:`\Phi`. This helps to map :math:`\psi_{jt}` to :math:`\Phi_{k}`.

    The top level (`corpus` level) stick proportions correspond the values of :math:`\\beta`,
    bottom level (`document` level) stick proportions correspond to the values of :math:`\pi`.
    The truncation level for the corpus (`K`) and document (`T`) corresponds to the number of :math:`\\beta`
    and :math:`\pi` which are in existence.

    Now, whenever coordinate ascent updates are to be performed, they happen at two level. The document level as well
    as corpus level.

    At document level, we update the following:

    #. The parameters to the document level sticks, i.e, a and b parameters of :math:`\\beta` distribution of the
       variable :math:`\pi _{jt}`.
    #. The parameters to per word topic indicators, :math:`Z_{j,n}`. Here :math:`Z_{j,n}` selects topic parameter
       :math:`\psi_{jt}`.
    #. The parameters to per document topic indices :math:`\Phi_{jtk}`.

    At corpus level, we update the following:

    #. The parameters to the top level sticks, i.e., the parameters of the :math:`\\beta` distribution for the
       corpus level :math:`\\beta`, which signify the topic distribution at corpus level.
    #. The parameters to the topics :math:`\Phi_{k}`.

    Now coming on to the steps involved, procedure for online variational inference for the Hdp model is as follows:

    1. We initialise the corpus level parameters, topic parameters randomly and set current time to 1.
    2. Fetch a random document j from the corpus.
    3. Compute all the parameters required for document level updates.
    4. Compute natural gradients of corpus level parameters.
    5. Initialise the learning rate as a function of kappa, tau and current time. Also, increment current time by 1
       each time it reaches this step.
    6. Update corpus level parameters.

    Repeat 2 to 6 until stopping condition is not met.

    Here the stopping condition corresponds to

    * time limit expired
    * chunk limit reached
    * whole corpus processed

    Attributes
    ----------
    lda_alpha : numpy.ndarray
        Same as :math:`\\alpha` from :class:`gensim.models.ldamodel.LdaModel`.
    lda_beta : numpy.ndarray
        Same as :math:`\\beta` from from :class:`gensim.models.ldamodel.LdaModel`.
    m_D : int
        Number of documents in the corpus.
    m_Elogbeta : numpy.ndarray:
        Stores value of dirichlet expectation, i.e., compute :math:`E[log \\theta]` for a vector
        :math:`\\theta \sim Dir(\\alpha)`.
    m_lambda : {numpy.ndarray, float}
        Drawn samples from the parameterized gamma distribution.
    m_lambda_sum : {numpy.ndarray, float}
        An array with the same shape as `m_lambda`, with the specified axis (1) removed.
    m_num_docs_processed : int
        Number of documents finished processing.This is incremented in size of chunks.
    m_r : list
        Acts as normaliser in lazy updating of `m_lambda` attribute.
    m_rhot : float
        Assigns weight to the information obtained from the mini-chunk and its value it between 0 and 1.
    m_status_up_to_date : bool
        Flag to indicate whether `lambda `and :math:`E[log \\theta]` have been updated if True, otherwise - not.
    m_timestamp : numpy.ndarray
        Helps to keep track and perform lazy updates on lambda.
    m_updatect : int
        Keeps track of current time and is incremented every time :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda`
        is called.
    m_var_sticks : numpy.ndarray
        Array of values for stick.
    m_varphi_ss : numpy.ndarray
        Used to update top level sticks.
    m_W : int
        Length of dictionary for the input corpus.

    """
    def __init__(self, corpus, id2word, max_chunks=None, max_time=None,
                 chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1,
                 gamma=1, eta=0.01, scale=1.0, var_converge=0.0001,
                 outputdir=None, random_state=None):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Corpus in BoW format.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`
            Dictionary for the input corpus.
        max_chunks : int, optional
            Upper bound on how many chunks to process. It wraps around corpus beginning in another corpus pass,
            if there are not enough chunks in the corpus.
        max_time : int, optional
            Upper bound on time (in seconds) for which model will be trained.
        chunksize : int, optional
            Number of documents in one chuck.
        kappa: float,optional
            Learning parameter which acts as exponential decay factor to influence extent of learning from each batch.
        tau: float, optional
            Learning parameter which down-weights early iterations of documents.
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
        random_state : {None, int, array_like, :class:`~np.random.RandomState`, optional}
            Adds a little random jitter to randomize results around same alpha when trying to fetch a closest
            corresponding lda model from :meth:`~gensim.models.hdpmodel.HdpModel.suggested_lda_model`

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
        """Infers the gamma value based for `chunk`.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.

        Returns
        -------
        numpy.ndarray
            First level concentration, i.e., Gamma value.

        Raises
        ------
        RuntimeError
            If model doesn't trained yet.

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
        bow : {iterable of list of (int, float), list of (int, float)
            BoW representation of the document/corpus to get topics for.
        eps : float, optional
            Ignore topics with probability below `eps`.

        Returns
        -------
        list of (int, float) **or** :class:`gensim.interfaces.TransformedCorpus`
            Topic distribution for the given document/corpus `bow`, as a list of `(topic_id, topic_probability)` or
            transformed corpus

        """
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(corpus)

        gamma = self.inference([bow])[0]
        topic_dist = gamma / sum(gamma) if sum(gamma) != 0 else []
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist) if topicvalue >= eps]

    def update(self, corpus):
        """Train the model with new documents, by EM-iterating over `corpus` until any of the conditions is satisfied.

        * time limit expired
        * chunk limit reached
        * whole corpus processed

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Corpus in BoW format.

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
        """Flag to determine whether the model has been updated with the new corpus or not.

        Parameters
        ----------
        start_time : float
            Indicates the current processor time as a floating point number expressed in seconds.
            The resolution is typically better on Windows than on Unix by one microsecond due to differing
            implementation of underlying function calls.
        chunks_processed : int
            Indicates progress of the update in terms of the number of chunks processed.
        docs_processed : int
            Indicates number of documents finished processing.This is incremented in size of chunks.

        Returns
        -------
        bool
            If True - model is updated, False otherwise.

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
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        update : bool, optional
            If True - call :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda`.
        opt_o : bool, optional
            Passed as argument to :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda`.
            If True then the topics will be ordered, False otherwise.

        Returns
        -------
        (float, int)
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
        """Performs E step for a single doc.

        Parameters
        ----------
        ss : :class:`~gensim.models.hdpmodel.SuffStats`
            Stats for all document(s) in the chunk.
        Elogsticks_1st : numpy.ndarray
            Computed Elogsticks value by stick-breaking process.
        unique_words : dict of (int, int)
            Number of unique words in the chunk.
        doc_word_ids : iterable of int
            Word ids of for a single document.
        doc_word_counts : iterable of int
            Word counts of all words in a single document.
        var_converge : float
            Lower bound on the right side of convergence. Used when updating variational parameters for a single
            document.

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
        """Update appropriate columns of lambda and top level sticks based on documents.

        Parameters
        ----------
        sstats : :class:`~gensim.models.hdpmodel.SuffStats`
            Statistic for all document(s) in the chunk.
        word_list : list of int
            Contains word id of all the unique words in the chunk of documents on which update is being performed.
        opt_o : bool, optional
            If True - invokes a call to :meth:`~gensim.models.hdpmodel.HdpModel.optimal_ordering` to order the topics.

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
            If True - logs a message with level INFO on the logger object.
        formatted : bool, optional
            If True - get the topics as a list of strings, otherwise - get the topics as lists of (weight, word) pairs.
        num_words : int, optional
            DEPRECATED, USE `topn` INSTEAD.

        Warnings
        --------
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
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
        """Get the term topic matrix learned during inference.

        Returns
        -------
        np.ndarray
            `num_topics` x `vocabulary_size` array of floats

        """
        topics = self.m_lambda + self.m_eta
        return topics / topics.sum(axis=1)[:, None]

    def show_topics(self, num_topics=20, num_words=20, log=False, formatted=True):
        """Print the `num_words` most probable words for `num_topics` number of topics.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics for which most probable `num_words` words will be fetched, if -1 - print all topics.
        num_words :  int, optional
            Number of most probable words to show from `num_topics` number of topics.
        log : bool, optional
            If True - log a message with level INFO on the logger object.
        formatted : bool, optional
            If True - get the topics as a list of strings, otherwise - get the topics as lists of (weight, word) pairs.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for topic terms depends on the value of `formatted` parameter.

        """
        if not self.m_status_up_to_date:
            self.update_expectations()
        betas = self.m_lambda + self.m_eta
        hdp_formatter = HdpTopicFormatter(self.id2word, betas)
        return hdp_formatter.show_topics(num_topics, num_words, log, formatted)

    @deprecated("This method will be removed in 4.0.0, use `save` instead.")
    def save_topics(self, doc_count=None):
        """Save discovered topics.

        Warnings
        --------
        This method is deprecated, use :meth:`~gensim.models.hdpmodel.HdpModel.save` instead.

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

    @deprecated("This method will be removed in 4.0.0, use `save` instead.")
    def save_options(self):
        """Writes all the values of the attributes for the current model in "options.dat" file.

        Warnings
        --------
        This method is deprecated, use :meth:`~gensim.models.hdpmodel.HdpModel.save` instead.

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
        """Get corresponding alpha and beta values of a LDA almost equivalent to current HDP.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Alpha and Beta arrays.

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
        """Get a trained ldamodel object which is closest to the current hdp model.

        The `num_topics=m_T`, so as to preserve the matrices shapes when we assign alpha and beta.

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
        corpus : iterable of list of (int, float)
            Test corpus in BoW format.

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
    """Helper class for :class:`gensim.models.hdpmodel.HdpModel` to format the output of topics."""
    (STYLE_GENSIM, STYLE_PRETTY) = (1, 2)

    def __init__(self, dictionary=None, topic_data=None, topic_file=None, style=None):
        """Initialise the :class:`gensim.models.hdpmodel.HdpTopicFormatter` and store topic data in sorted order.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`,optional
            Dictionary for the input corpus.
        topic_data : numpy.ndarray, optional
            The term topic matrix.
        topic_file : {file-like object, str, pathlib.Path}
            File, filename, or generator to read. If the filename extension is .gz or .bz2, the file is first
            decompressed. Note that generators should return byte strings for Python 3k.
        style : bool, optional
            If True - get the topics as a list of strings, otherwise - get the topics as lists of (word, weight) pairs.

        Raises
        ------
        ValueError
            Either dictionary is None or both `topic_data` and `topic_file` is None.

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
        """Give the most probable `num_words` words from `num_topics` topics.
        Alias for :meth:`~gensim.models.hdpmodel.HdpTopicFormatter.show_topics`.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for `num_words` words from `num_topics` topics depends on the value of `self.style` attribute.

        """
        return self.show_topics(num_topics, num_words, True)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Give the most probable `num_words` words from `num_topics` topics.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.
        log : bool, optional
            If True - log a message with level INFO on the logger object.
        formatted : bool, optional
            If True - get the topics as a list of strings, otherwise as lists of (word, weight) pairs.

        Returns
        -------
        list of (int, list of (str, numpy.float) **or** list of str)
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
        """Print the `topn` most probable words from topic id `topic_id`.

        Warnings
        --------
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        num_words : int, optional
            DEPRECATED, USE `topn` INSTEAD.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for terms from a single topic depends on the value of `formatted` parameter.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn(
                "The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead."
            )
            topn = num_words

        return self.show_topic(topic_id, topn, formatted=True)

    def show_topic(self, topic_id, topn=20, log=False, formatted=False, num_words=None,):
        """Give the most probable `num_words` words for the id `topic_id`.

        Warnings
        --------
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        log : bool, optional
            If True logs a message with level INFO on the logger object, False otherwise.
        formatted : bool, optional
            If True return the topics as a list of strings, False as lists of
            (word, weight) pairs.
        num_words : int, optional
            DEPRECATED, USE `topn` INSTEAD.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
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
        """Give the topic terms along with their probabilities for a single topic data.

        Parameters
        ----------
        topic_data : list of (str, numpy.float)
            Contains probabilities for each word id belonging to a single topic.
        num_words : int
            Number of words for which probabilities are to be extracted from the given single topic data.

        Returns
        -------
        list of (str, numpy.float)
            A sequence of topic terms and their probabilities.

        """
        return [(self.dictionary[wid], weight) for (weight, wid) in topic_data[:num_words]]

    def format_topic(self, topic_id, topic_terms):
        """Format the display for a single topic in two different ways.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topic_terms : list of (str, numpy.float)
            Contains the most probable words from a single topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for topic terms depends on the value of `self.style` attribute.

        """
        if self.STYLE_GENSIM == self.style:
            fmt = ' + '.join(['%.3f*%s' % (weight, word) for (word, weight) in topic_terms])
        else:
            fmt = '\n'.join(['    %20s    %.8f' % (word, weight) for (word, weight) in topic_terms])

        fmt = (topic_id, fmt)
        return fmt
