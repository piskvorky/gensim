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


"""
This module encapsulates functionality for the online Hierarchical Dirichlet Process algorithm.

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

"""

from __future__ import with_statement

import logging, time
import numpy
from scipy.special import gammaln, psi  # gamma function utils

from gensim import interfaces, utils, matutils
from gensim.models import basemodel, ldamodel
from six.moves import xrange

logger = logging.getLogger(__name__)

meanchangethresh = 0.00001
rhot_bound = 0.0


def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(numpy.sum(alpha))
    else:
        result = psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input


def expect_log_sticks(sticks):
    """
    For stick-breaking hdp, return the E[log(sticks)]
    """
    dig_sum = psi(numpy.sum(sticks, 0))
    ElogW = psi(sticks[0]) - dig_sum
    Elog1_W = psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = numpy.zeros(n)
    Elogsticks[0: n - 1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + numpy.cumsum(Elog1_W)
    return Elogsticks


def lda_e_step(doc_word_ids, doc_word_counts, alpha, beta, max_iter=100):
    gamma = numpy.ones(len(alpha))
    expElogtheta = numpy.exp(dirichlet_expectation(gamma))
    betad = beta[:, doc_word_ids]
    phinorm = numpy.dot(expElogtheta, betad) + 1e-100
    counts = numpy.array(doc_word_counts)
    for _ in xrange(max_iter):
        lastgamma = gamma

        gamma = alpha + expElogtheta * numpy.dot(counts / phinorm, betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)
        phinorm = numpy.dot(expElogtheta, betad) + 1e-100
        meanchange = numpy.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = numpy.sum(counts * numpy.log(phinorm))
    likelihood += numpy.sum((alpha - gamma) * Elogtheta)
    likelihood += numpy.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(numpy.sum(alpha)) - gammaln(numpy.sum(gamma))

    return (likelihood, gamma)


class SuffStats(object):
    def __init__(self, T, Wt, Dt):
        self.m_chunksize = Dt
        self.m_var_sticks_ss = numpy.zeros(T)
        self.m_var_beta_ss = numpy.zeros((T, Wt))

    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)


class HdpModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """
    The constructor estimates Hierachical Dirichlet Process model parameters based
    on a training corpus:

    >>> hdp = HdpModel(corpus, id2word)
    >>> hdp.print_topics(show_topics=20, num_words=10)

    Inference on new documents is based on the approximately LDA-equivalent topics.

    Model persistency is achieved through its `load`/`save` methods.

    """
    def __init__(self, corpus, id2word, max_chunks=None, max_time=None,
                 chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1,
                 gamma=1, eta=0.01, scale=1.0, var_converge=0.0001,
                 outputdir=None):
        """
        `gamma`: first level concentration
        `alpha`: second level concentration
        `eta`: the topic Dirichlet
        `T`: top level truncation level
        `K`: second level truncation level
        `kappa`: learning rate
        `tau`: slow down parameter
        `max_time`: stop training after this many seconds
        `max_chunks`: stop after having processed this many chunks (wrap around
        corpus beginning in another corpus pass, if there are not enough chunks
        in the corpus)
        """
        self.corpus = corpus
        self.id2word = id2word
        self.chunksize = chunksize
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.outputdir = outputdir

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

        self.m_var_sticks = numpy.zeros((2, T - 1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = range(T - 1, 0, -1)
        self.m_varphi_ss = numpy.zeros(T)

        self.m_lambda = numpy.random.gamma(1.0, 1.0, (T, self.m_W)) * self.m_D * 100 / (T * self.m_W) - eta
        self.m_eta = eta
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0
        self.m_status_up_to_date = True
        self.m_num_docs_processed = 0

        self.m_timestamp = numpy.zeros(self.m_W, dtype=int)
        self.m_r = [0]
        self.m_lambda_sum = numpy.sum(self.m_lambda, axis=1)

        self.m_var_converge = var_converge

        if self.outputdir:
            self.save_options()

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.update(corpus)

    def inference(self, chunk):
        if self.lda_alpha is None or self.lda_beta is None:
            raise RuntimeError("model must be trained to perform inference")
        chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents" % len(chunk))

        gamma = numpy.zeros((len(chunk), self.lda_beta.shape[0]))
        for d, doc in enumerate(chunk):
            if not doc:  # leave gamma at zero for empty documents
                continue
            ids, counts = zip(*doc)
            _, gammad = lda_e_step(ids, counts, self.lda_alpha, self.lda_beta)
            gamma[d, :] = gammad
        return gamma

    def __getitem__(self, bow, eps=0.01):
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(corpus)

        gamma = self.inference([bow])[0]
        topic_dist = gamma / sum(gamma) if sum(gamma) != 0 else []
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= eps]

    def update(self, corpus):
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
        return (
            # chunk limit reached
            (self.max_chunks and chunks_processed == self.max_chunks) or

            # time limit reached
            (self.max_time and time.clock() - start_time > self.max_time) or

            # no limits and whole corpus has been processed once
            (not self.max_chunks and not self.max_time and docs_processed >= self.m_D))

    def update_chunk(self, chunk, update=True, opt_o=True):
        # Find the unique words in this chunk...
        unique_words = dict()
        word_list = []
        for doc in chunk:
            for word_id, _ in doc:
                if word_id not in unique_words:
                    unique_words[word_id] = len(unique_words)
                    word_list.append(word_id)

        Wt = len(word_list)  # length of words in these documents

        # ...and do the lazy updates on the necessary columns of lambda
        rw = numpy.array([self.m_r[t] for t in self.m_timestamp[word_list]])
        self.m_lambda[:, word_list] *= numpy.exp(self.m_r[-1] - rw)
        self.m_Elogbeta[:, word_list] = \
            psi(self.m_eta + self.m_lambda[:, word_list]) - \
            psi(self.m_W * self.m_eta + self.m_lambda_sum[:, numpy.newaxis])

        ss = SuffStats(self.m_T, Wt, len(chunk))

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks)  # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        for doc in chunk:
            if len(doc) > 0:
                doc_word_ids, doc_word_counts = zip(*doc)
                doc_score = self.doc_e_step(
                    doc, ss, Elogsticks_1st,
                    word_list, unique_words, doc_word_ids,
                    doc_word_counts, self.m_var_converge)
                count += sum(doc_word_counts)
                score += doc_score

        if update:
            self.update_lambda(ss, word_list, opt_o)

        return (score, count)

    def doc_e_step(self, doc, ss, Elogsticks_1st, word_list,
                   unique_words, doc_word_ids, doc_word_counts, var_converge):
        """
        e step for a single doc
        """
        chunkids = [unique_words[id] for id in doc_word_ids]

        Elogbeta_doc = self.m_Elogbeta[:, doc_word_ids]
        ## very similar to the hdp equations
        v = numpy.zeros((2, self.m_K - 1))
        v[0] = 1.0
        v[1] = self.m_alpha

        # back to the uniform
        phi = numpy.ones((len(doc_word_ids), self.m_K)) * 1.0 / self.m_K

        likelihood = 0.0
        old_likelihood = -1e200
        converge = 1.0
        eps = 1e-100

        iter = 0
        max_iter = 100
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters

            # var_phi
            if iter < 3:
                var_phi = numpy.dot(phi.T,  (Elogbeta_doc * doc_word_counts).T)
                (log_var_phi, log_norm) = matutils.ret_log_normalize_vec(var_phi)
                var_phi = numpy.exp(log_var_phi)
            else:
                var_phi = numpy.dot(phi.T,  (Elogbeta_doc * doc_word_counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = matutils.ret_log_normalize_vec(var_phi)
                var_phi = numpy.exp(log_var_phi)

            # phi
            if iter < 3:
                phi = numpy.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = matutils.ret_log_normalize_vec(phi)
                phi = numpy.exp(log_phi)
            else:
                phi = numpy.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
                (log_phi, log_norm) = matutils.ret_log_normalize_vec(phi)
                phi = numpy.exp(log_phi)

            # v
            phi_all = phi * numpy.array(doc_word_counts)[:, numpy.newaxis]
            v[0] = 1.0 + numpy.sum(phi_all[:, :self.m_K - 1], 0)
            phi_cum = numpy.flipud(numpy.sum(phi_all[:, 1:], 0))
            v[1] = self.m_alpha + numpy.flipud(numpy.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += numpy.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = numpy.log(self.m_alpha)
            likelihood += (self.m_K - 1) * log_alpha
            dig_sum = psi(numpy.sum(v, 0))
            likelihood += numpy.sum((numpy.array([1.0, self.m_alpha])[:, numpy.newaxis] - v) * (psi(v) - dig_sum))
            likelihood -= numpy.sum(gammaln(numpy.sum(v, 0))) - numpy.sum(gammaln(v))

            # Z part
            likelihood += numpy.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += numpy.sum(phi.T * numpy.dot(var_phi, Elogbeta_doc * doc_word_counts))

            converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                logger.warning('likelihood is decreasing!')

            iter += 1

        # update the suff_stat ss
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += numpy.sum(var_phi, 0)
        ss.m_var_beta_ss[:, chunkids] += numpy.dot(var_phi.T, phi.T * doc_word_counts)

        return likelihood

    def update_lambda(self, sstats, word_list, opt_o):
        self.m_status_up_to_date = False
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-chunk.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound:
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, word_list] = self.m_lambda[:, word_list] * (1 - rhot) + \
            rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_chunksize
        self.m_lambda_sum = (1 - rhot) * self.m_lambda_sum + \
            rhot * self.m_D * numpy.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_chunksize

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect
        self.m_r.append(self.m_r[-1] + numpy.log(1 - rhot))

        self.m_varphi_ss = (1.0 - rhot) * self.m_varphi_ss + rhot * \
            sstats.m_var_sticks_ss * self.m_D / sstats.m_chunksize

        if opt_o:
            self.optimal_ordering()

        ## update top level sticks
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T - 1] + 1.0
        var_phi_sum = numpy.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = numpy.flipud(numpy.cumsum(var_phi_sum)) + self.m_gamma

    def optimal_ordering(self):
        """
        ordering the topics
        """
        idx = matutils.argsort(self.m_lambda_sum, reverse=True)
        self.m_varphi_ss = self.m_varphi_ss[idx]
        self.m_lambda = self.m_lambda[idx, :]
        self.m_lambda_sum = self.m_lambda_sum[idx]
        self.m_Elogbeta = self.m_Elogbeta[idx, :]

    def update_expectations(self):
        """
        Since we're doing lazy updates on lambda, at any given moment
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta
        so that if (for example) we want to print out the
        topics we've learned we'll get the correct behavior.
        """
        for w in xrange(self.m_W):
            self.m_lambda[:, w] *= numpy.exp(self.m_r[-1] -
                                          self.m_r[self.m_timestamp[w]])
        self.m_Elogbeta = psi(self.m_eta + self.m_lambda) - \
            psi(self.m_W * self.m_eta + self.m_lambda_sum[:, numpy.newaxis])

        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def show_topics(self, num_topics=20, num_words=20, log=False, formatted=True):
        """
        Print the `num_words` most probable words for `topics` number of topics.
        Set `topics=-1` to print all topics.

        Set `formatted=True` to return the topics as a list of strings, or
        `False` as lists of (weight, word) pairs.

        """
        if not self.m_status_up_to_date:
            self.update_expectations()
        betas = self.m_lambda + self.m_eta
        hdp_formatter = HdpTopicFormatter(self.id2word, betas)
        return hdp_formatter.show_topics(num_topics, num_words, log, formatted)

    def save_topics(self, doc_count=None):
        """legacy method; use `self.save()` instead"""
        if not self.outputdir:
            logger.error("cannot store topics without having specified an output directory")

        if doc_count is None:
            fname = 'final'
        else:
            fname = 'doc-%i' % doc_count
        fname = '%s/%s.topics' % (self.outputdir, fname)
        logger.info("saving topics to %s" % fname)
        betas = self.m_lambda + self.m_eta
        numpy.savetxt(fname, betas)

    def save_options(self):
        """legacy method; use `self.save()` instead"""
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
        """
        Compute the LDA almost equivalent HDP.
        """
        # alpha
        sticks = self.m_var_sticks[0] / (self.m_var_sticks[0] + self.m_var_sticks[1])
        alpha = numpy.zeros(self.m_T)
        left = 1.0
        for i in xrange(0, self.m_T - 1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T - 1] = left
        alpha = alpha * self.m_alpha

        # beta
        beta = (self.m_lambda + self.m_eta) / (self.m_W * self.m_eta + \
                self.m_lambda_sum[:, numpy.newaxis])

        return (alpha, beta)


    def suggested_lda_model(self):
        """
        Returns closest corresponding ldamodel object corresponding to current hdp model.
        """
        alpha, beta = self.hdp_to_lda()
        ldam = ldamodel.LdaModel(num_topics=150, alpha=alpha, id2word=self.id2word)
        ldam.expElogbeta[:] = beta
        return ldam

        
    def evaluate_test_corpus(self, corpus):
        logger.info('TEST: evaluating test corpus')
        if self.lda_alpha is None or self.lda_beta is None:
            self.lda_alpha, self.lda_beta = self.hdp_to_lda()
        score = 0.0
        total_words = 0
        for i, doc in enumerate(corpus):
            if len(doc) > 0:
                doc_word_ids, doc_word_counts = zip(*doc)
                likelihood, gamma = lda_e_step(doc_word_ids, doc_word_counts, self.lda_alpha, self.lda_beta)
                theta = gamma / numpy.sum(gamma)
                lda_betad = self.lda_beta[:, doc_word_ids]
                log_predicts = numpy.log(numpy.dot(theta, lda_betad))
                doc_score = sum(log_predicts) / len(doc)
                logger.info('TEST: %6d    %.5f' % (i, doc_score))
                score += likelihood
                total_words += sum(doc_word_counts)
        logger.info('TEST: average score: %.5f, total score: %.5f,  test docs: %d' % (score / total_words, score, len(corpus)))
        return score
#endclass HdpModel


class HdpTopicFormatter(object):
    (STYLE_GENSIM, STYLE_PRETTY) = (1, 2)

    def __init__(self, dictionary=None, topic_data=None, topic_file=None, style=None):
        if dictionary is None:
            raise ValueError('no dictionary!')

        if topic_data is not None:
            topics = topic_data
        elif topic_file is not None:
            topics = numpy.loadtxt('%s' % topic_file)
        else:
            raise ValueError('no topic data!')

        # sort topics
        topics_sums = numpy.sum(topics, axis=1)
        idx = matutils.argsort(topics_sums, reverse=True)
        self.data = topics[idx]

        self.dictionary = dictionary

        if style is None:
            style = self.STYLE_GENSIM

        self.style = style

    def print_topics(self, num_topics=10, num_words=10):
        return self.show_topics(num_topics, num_words, True)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
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

    def show_topic_terms(self, topic_data, num_words):
        return [(self.dictionary[wid], weight) for (weight, wid) in topic_data[:num_words]]

    def format_topic(self, topic_id, topic_terms):
        if self.STYLE_GENSIM == self.style:
            fmt = ' + '.join(['%.3f*%s' % (weight, word) for (word, weight) in topic_terms])
        else:
            fmt = '\n'.join(['    %20s    %.8f' % (word, weight) for (word, weight) in topic_terms])

        fmt = (topic_id, fmt)
        return fmt
# endclass HdpTopicFormatter
