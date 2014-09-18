#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Artyom Topchyan <artyom.topchyan@live.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>


"""
Python wrapper for Dynamic Topic Models (DTM) and the Document Influence Model (DIM)  [1].

This module allows for DTM and DIM model estimation from a training corpus.

Example:

>>> model = gensim.models.DtmModel('dtm-win64.exe', my_corpus, my_timeslices, num_topics=20, id2word=dictionary)

.. [1] https://code.google.com/p/princeton-statistical-learning/downloads/detail?name=dtm_release-0.8.tgz

"""


import logging
import random
import tempfile
import os
from subprocess import Popen, PIPE

import numpy as np

from gensim import utils, corpora

logger = logging.getLogger(__name__)


class DtmModel(utils.SaveLoad):
    """
    Class for DTM training using DTM binary. Communication between DTM and Python
    takes place by passing around data files on disk and executing the DTM binary as a subprocess.

    """

    def __init__(
        self, dtm_path, corpus=None, time_slices=None, num_topics=100, id2word=None, prefix=None,
            lda_sequence_min_iter=6, lda_sequence_max_iter=20, lda_max_em_iter=10, alpha=0.01, top_chain_var=0.005, rng_seed=0, initialize_lda=False):
        """
        `dtm_path` is path to the dtm executable, e.g. `C:/dtm/dtm-win64.exe`.

        `corpus` is a gensim corpus, aka a stream of sparse document vectors.

        `id2word` is a mapping between tokens ids and token.

        `lda_sequence_min_iter` min iteration of LDA.

        `lda_sequence_max_iter` max iteration of LDA.

        `lda_max_em_iter` max em optiimzatiion iterations in LDA.

        `alpha` is a hyperparameter that affects sparsity of the document-topics for the LDA models in each timeslice.

        `top_chain_var` is a hyperparameter that affects.

        `rng_seed` is the random seed.

        `initialize_lda` initialize DTM with LDA.

        """
        self.dtm_path = dtm_path
        self.id2word = id2word
        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("cannot compute DTM over an empty collection (no terms)")
        self.num_topics = num_topics

        try:
            lencorpus = len(corpus)
        except:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            raise ValueError("cannot compute DTM over an empty corpus")
        if lencorpus != sum(time_slices):
            raise ValueError("mismatched timeslices %{slices} for corpus of len {clen}".format(
                slices=sum(time_slices), clen=lencorpus))
        self.lencorpus = lencorpus
        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)

        self.prefix = prefix
        self.time_slices = time_slices
        self.lda_sequence_min_iter = int(lda_sequence_min_iter)
        self.lda_sequence_max_iter = int(lda_sequence_max_iter)
        self.lda_max_em_iter = int(lda_max_em_iter)
        self.alpha = alpha
        self.top_chain_var = top_chain_var
        self.rng_seed = rng_seed
        self.initialize_lda = str(initialize_lda).lower()

        self.lambda_ = None
        self.obs_ = None
        self.lhood_ = None
        self.gamma_ = None
        self.init_alpha = None
        self.init_beta = None
        self.init_ss = None
        self.em_steps = []
        self.influences_time = []

        if corpus is not None:
            self.train(corpus, time_slices)

    def fout_liklihoods(self):
        return self.prefix + 'train_out/lda-seq/' + 'lhoods.dat'

    def fout_gamma(self):
        return self.prefix + 'train_out/lda-seq/' + 'gam.dat'

    def fout_prob(self):
        return self.prefix + 'train_out/lda-seq/' + 'topic-{i}-var-e-log-prob.dat'

    def fout_observations(self):
        return self.prefix + 'train_out/lda-seq/' + 'topic-{i}-var-obs.dat'

    def fout_influence(self):
        return self.prefix + 'train_out/lda-seq/' + 'influence_time-{i}'

    def foutname(self):
        return self.prefix + 'train_out'

    def fem_steps(self):
        return self.prefix + 'train_out/' + 'em_log.dat'

    def finit_alpha(self):
        return self.prefix + 'train_out/' + 'initial-lda.alpha'

    def finit_beta(self):
        return self.prefix + 'train_out/' + 'initial-lda.beta'

    def flda_ss(self):
        return self.prefix + 'train_out/' + 'initial-lda-ss.dat'

    def fcorpustxt(self):
        return self.prefix + 'train-mult.dat'

    def fcorpus(self):
        return self.prefix + 'train'

    def ftimeslices(self):
        return self.prefix + 'train-seq.dat'

    def convert_input(self, corpus, time_slices):
        """
        Serialize documents in LDA-C format to a temporary text file,.

        """
        logger.info("serializing temporary corpus to %s" % self.fcorpustxt())
        # write out the corpus in a file format that DTM understands:
        corpora.BleiCorpus.save_corpus(self.fcorpustxt(), corpus)

        with utils.smart_open(self.ftimeslices(), 'wb') as fout:
            fout.write(str(len(self.time_slices)) + "\n")
            for sl in time_slices:
                fout.write(str(sl) + "\n")

    def train(self, corpus, time_slices, mode='fit', model='fixed'):
        """
        Train DTM model using specified corpus and time slices.

        'mode' controls the mode of the mode: 'fit' is for training, 'time' for
        analyzing documents through time according to a DTM, basically a held out set.

        'model' controls the coice of model. 'fixed' is for DIM and 'dtm' for DTM.

        """
        self.convert_input(corpus, time_slices)

        arguments = "--ntopics={p0} --model={mofrl}  --mode={p1} --initialize_lda={p2} --corpus_prefix={p3} --outname={p4} --alpha={p5}".format(
            p0=self.num_topics, mofrl=model, p1=mode, p2=self.initialize_lda, p3=self.fcorpus(), p4=self.foutname(), p5=self.alpha)

        params = "--lda_max_em_iter={p0} --lda_sequence_min_iter={p1}  --lda_sequence_max_iter={p2} --top_chain_var={p3} --rng_seed={p4} ".format(
            p0=self.lda_max_em_iter, p1=self.lda_sequence_min_iter, p2=self.lda_sequence_max_iter, p3=self.top_chain_var, p4=self.rng_seed)

        arguments = arguments + " " + params
        logger.info("training DTM with args %s" % arguments)
        try:
            p = Popen([self.dtm_path] + arguments.split(), stdout=PIPE, stderr=PIPE)
            p.communicate()
        except KeyboardInterrupt:
            p.terminate()
        self.em_steps = np.loadtxt(self.fem_steps())
        self.init_alpha = np.loadtxt(self.finit_alpha())
        self.init_beta = np.loadtxt(self.finit_beta())
        self.init_ss = np.loadtxt(self.flda_ss())

        self.lhood_ = np.loadtxt(self.fout_liklihoods())

        # document-topic proportions
        self.gamma_ = np.loadtxt(self.fout_gamma())
        # cast to correct shape, gamme[5,10] is the proprtion of the 10th topic
        # in doc 5
        self.gamma_.shape = (self.lencorpus, self.num_topics)
        # normalize proportions
        self.gamma_ /= self.gamma_.sum(axis=1)[:, np.newaxis]

        self.lambda_ = np.zeros((self.num_topics, self.num_terms * len(self.time_slices)))
        self.obs_ = np.zeros((self.num_topics, self.num_terms * len(self.time_slices)))

        for t in range(self.num_topics):
                topic = "%03d" % t
                self.lambda_[t, :] = np.loadtxt(self.fout_prob().format(i=topic))
                self.obs_[t, :] = np.loadtxt(self.fout_observations().format(i=topic))
        # cast to correct shape, lambda[5,10,0] is the proportion of the 10th
        # topic in doc 5 at time 0
        self.lambda_.shape = (self.num_topics, self.num_terms, len(self.time_slices))
        self.obs_.shape = (self.num_topics, self.num_terms, len(self.time_slices))
        # extract document influence on topics for each time slice
        # influences_time[0] , influences at time 0
        if model == 'fixed':
            for k, t in enumerate(self.time_slices):
                stamp = "%03d" % k
                influence = np.loadtxt(self.fout_influence().format(i=stamp))
                influence.shape = (t, self.num_topics)
                # influence[2,5] influence of document 2 on topic 5
                self.influences_time.append(influence)

    def print_topics(self, topics=10, times=5, topn=10):
        return self.show_topics(topics, times, topn, log=True)

    def show_topics(self, topics=10, times=5, topn=10, log=False, formatted=True):
        """
        Print the `topn` most probable words for `topics` number of topics at 'times' time slices.
        Set `topics=-1` to print all topics.

        Set `formatted=True` to return the topics as a list of strings, or `False` as lists of (weight, word) pairs.

        """
        if topics < 0 or topics >= self.num_topics:
            topics = self.num_topics
            chosen_topics = range(topics)
        else:
            topics = min(topics, self.num_topics)
            chosen_topics = range(topics)
             # add a little random jitter, to randomize results around the same
            # alpha
            # sort_alpha = self.alpha + 0.0001 * \
            #     numpy.random.rand(len(self.alpha))
            # sorted_topics = list(numpy.argsort(sort_alpha))
            # chosen_topics = sorted_topics[: topics / 2] + \
            #     sorted_topics[-topics / 2:]

        if times < 0 or times >= self.time_slices:
            times = self.time_slices
            chosen_times = range(times)
        else:
            times = min(times, self.time_slices)
            chosen_times = range(times)

        shown = []
        for time in chosen_times:
            for i in chosen_topics:
                if formatted:
                    topic = self.print_topic(i, time, topn=topn)
                else:
                    topic = self.show_topic(i, time, topn=topn)
                shown.append(topic)
                # if log:
                # logger.info("topic #%i (%.3f): %s" % (i, self.alpha[i],
                #     topic))
        return shown

    def show_topic(self, topicid, time, topn=50):
        """
        Return `topn` most probable words for the given `topicid`, as a list of
        `(word_probability, word)` 2-tuples.

        """
        topics = self.lambda_[:, :, time]
        topic = topics[topicid]
        # liklihood to probability
        topic = np.exp(topic)
        # normalize to probability dist
        topic = topic / topic.sum()
        # sort according to prob
        bestn = np.argsort(topic)[::-1][:topn]
        beststr = [(topic[id], self.id2word[id]) for id in bestn]
        return beststr

    def print_topic(self, topicid, time, topn=10):
        """Return the given topic, formatted as a string."""
        return ' + '.join(['%.3f*%s' % v for v in self.show_topic(topicid, time, topn)])
