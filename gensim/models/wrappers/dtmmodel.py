#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Artyom Topchyan <artyom.topchyan@live.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>


"""Python wrapper for `Dynamic Topic Models (DTM) <http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf>`_
and the `Document Influence Model (DIM) <http://www.cs.columbia.edu/~blei/papers/GerrishBlei2010.pdf>`_.

Installation
------------

You have 2 ways, how to make binaries:

#. Use precompiled binaries for your OS version from `/magsilva/dtm/ <https://github.com/magsilva/dtm/tree/master/bin>`_
#. Compile binaries manually from `/blei-lab/dtm <https://github.com/blei-lab/dtm.git>`_
   (original instruction available in https://github.com/blei-lab/dtm/blob/master/README.md), or use this ::

    git clone https://github.com/blei-lab/dtm.git
    sudo apt-get install libgsl0-dev
    cd dtm/dtm
    make

Examples
--------
>>> from gensim.test.utils import common_corpus, common_dictionary
>>> from gensim.models.wrappers import DtmModel
>>>
>>> path_to_dtm_binary = "/path/to/dtm/binary"
>>> model = DtmModel(
...    path_to_dtm_binary, corpus=common_corpus, id2word=common_dictionary,
...    time_slices=[1] * len(common_corpus)
... )

"""


import logging
import random
import warnings
import tempfile
import os
from subprocess import PIPE
import numpy as np

from gensim import utils, corpora, matutils
from gensim.utils import check_output

logger = logging.getLogger(__name__)


class DtmModel(utils.SaveLoad):
    """DTM wrapper using DTM binary.

    Communication between DTM and Python takes place by passing around data files on disk and executing
    the DTM binary as a subprocess.

    """

    def __init__(self, dtm_path, corpus=None, time_slices=None, mode='fit', model='dtm', num_topics=100,
                 id2word=None, prefix=None, lda_sequence_min_iter=6, lda_sequence_max_iter=20, lda_max_em_iter=10,
                 alpha=0.01, top_chain_var=0.005, rng_seed=0, initialize_lda=True):
        """

        Parameters
        ----------
        dtm_path : str
            Path to the dtm binary, e.g. `/home/username/dtm/dtm/main`.
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        time_slices : list of int
            Sequence of timestamps.
        mode : {'fit', 'time'}, optional
            Controls the mode of the mode: 'fit' is for training, 'time' for analyzing documents through time
            according to a DTM, basically a held out set.
        model : {'fixed', 'dtm'}, optional
            Control model that will be runned: 'fixed' is for DIM and 'dtm' for DTM.
        num_topics : int, optional
            Number of topics.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`.
        prefix : str, optional
            Prefix for produced temporary files.
        lda_sequence_min_iter : int, optional
             Min iteration of LDA.
        lda_sequence_max_iter : int, optional
            Max iteration of LDA.
        lda_max_em_iter : int, optional
             Max em optimization iterations in LDA.
        alpha : int, optional
            Hyperparameter that affects sparsity of the document-topics for the LDA models in each timeslice.
        top_chain_var : int, optional
            Hyperparameter that affects.
        rng_seed : int, optional
             Random seed.
        initialize_lda : bool, optional
             If True - initialize DTM with LDA.

        """
        if not os.path.isfile(dtm_path):
            raise ValueError("dtm_path must point to the binary file, not to a folder")

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
        except TypeError:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            raise ValueError("cannot compute DTM over an empty corpus")
        if model == "fixed" and any(not text for text in corpus):
            raise ValueError("""There is a text without words in the input corpus.
                    This breaks method='fixed' (The DIM model).""")
        if lencorpus != sum(time_slices):
            raise ValueError(
                "mismatched timeslices %{slices} for corpus of len {clen}"
                .format(slices=sum(time_slices), clen=lencorpus)
            )
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
            self.train(corpus, time_slices, mode, model)

    def fout_liklihoods(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/lda-seq/' + 'lhoods.dat'

    def fout_gamma(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/lda-seq/' + 'gam.dat'

    def fout_prob(self):
        """Get template of path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/lda-seq/' + 'topic-{i}-var-e-log-prob.dat'

    def fout_observations(self):
        """Get template of path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/lda-seq/' + 'topic-{i}-var-obs.dat'

    def fout_influence(self):
        """Get template of path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/lda-seq/' + 'influence_time-{i}'

    def foutname(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out'

    def fem_steps(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/' + 'em_log.dat'

    def finit_alpha(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/' + 'initial-lda.alpha'

    def finit_beta(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/' + 'initial-lda.beta'

    def flda_ss(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train_out/' + 'initial-lda-ss.dat'

    def fcorpustxt(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train-mult.dat'

    def fcorpus(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train'

    def ftimeslices(self):
        """Get path to temporary file.

        Returns
        -------
        str
            Path to file

        """
        return self.prefix + 'train-seq.dat'

    def convert_input(self, corpus, time_slices):
        """Convert corpus into LDA-C format and save to temporary text file.

        Parameters
        ----------
        corpus : iterable of iterable of (int, float)
            Corpus in BoW format.
        time_slices : list of int
            Sequence of timestamps.

        """
        logger.info("serializing temporary corpus to %s", self.fcorpustxt())
        # write out the corpus in a file format that DTM understands:
        corpora.BleiCorpus.save_corpus(self.fcorpustxt(), corpus)

        with utils.smart_open(self.ftimeslices(), 'wb') as fout:
            fout.write(utils.to_utf8(str(len(self.time_slices)) + "\n"))
            for sl in time_slices:
                fout.write(utils.to_utf8(str(sl) + "\n"))

    def train(self, corpus, time_slices, mode, model):
        """Train DTM model.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        time_slices : list of int
            Sequence of timestamps.
        mode : {'fit', 'time'}, optional
            Controls the mode of the mode: 'fit' is for training, 'time' for analyzing documents through time
            according to a DTM, basically a held out set.
        model : {'fixed', 'dtm'}, optional
            Control model that will be runned: 'fixed' is for DIM and 'dtm' for DTM.

        """
        self.convert_input(corpus, time_slices)

        arguments = \
            "--ntopics={p0} --model={mofrl}  --mode={p1} --initialize_lda={p2} --corpus_prefix={p3} " \
            "--outname={p4} --alpha={p5}".format(
                p0=self.num_topics, mofrl=model, p1=mode, p2=self.initialize_lda,
                p3=self.fcorpus(), p4=self.foutname(), p5=self.alpha
            )

        params = \
            "--lda_max_em_iter={p0} --lda_sequence_min_iter={p1}  --lda_sequence_max_iter={p2} " \
            "--top_chain_var={p3} --rng_seed={p4} ".format(
                p0=self.lda_max_em_iter, p1=self.lda_sequence_min_iter, p2=self.lda_sequence_max_iter,
                p3=self.top_chain_var, p4=self.rng_seed
            )

        arguments = arguments + " " + params
        logger.info("training DTM with args %s", arguments)

        cmd = [self.dtm_path] + arguments.split()
        logger.info("Running command %s", cmd)
        check_output(args=cmd, stderr=PIPE)

        self.em_steps = np.loadtxt(self.fem_steps())
        self.init_ss = np.loadtxt(self.flda_ss())

        if self.initialize_lda:
            self.init_alpha = np.loadtxt(self.finit_alpha())
            self.init_beta = np.loadtxt(self.finit_beta())

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

    def print_topics(self, num_topics=10, times=5, num_words=10):
        return self.show_topics(num_topics, times, num_words, log=True)

    def show_topics(self, num_topics=10, times=5, num_words=10, log=False, formatted=True):
        """Get the `num_words` most probable words for `num_topics` number of topics at 'times' time slices.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to return, set `-1` to get all topics.
        times : int, optional
            Number of times.
        num_words : int, optional
            Number of words.
        log : bool, optional
            THIS PARAMETER WILL BE IGNORED.
        formatted : bool, optional
            If `True` - return the topics as a list of strings, otherwise as lists of (weight, word) pairs.

        Returns
        -------
        list of str
            Topics as a list of strings (if formatted=True) **OR**
        list of (float, str)
            Topics as list of (weight, word) pairs (if formatted=False)

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            chosen_topics = range(num_topics)

        if times < 0 or times >= len(self.time_slices):
            times = len(self.time_slices)
            chosen_times = range(times)
        else:
            times = min(times, len(self.time_slices))
            chosen_times = range(times)

        shown = []
        for time in chosen_times:
            for i in chosen_topics:
                if formatted:
                    topic = self.print_topic(i, time, num_words=num_words)
                else:
                    topic = self.show_topic(i, time, num_words=num_words)
                shown.append(topic)
        return shown

    def show_topic(self, topicid, time, topn=50, num_words=None):
        """Get `num_words` most probable words for the given `topicid`.

        Parameters
        ----------
        topicid : int
            Id of topic.
        time : int
            Timestamp.
        topn : int, optional
            Top number of topics that you'll receive.
        num_words : int, optional
            DEPRECATED PARAMETER, use `topn` instead.

        Returns
        -------
        list of (float, int)
            Sequence of probable words, as a list of `(word_probability, word)`.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn("The parameter `num_words` is deprecated, will be removed in 4.0.0, use `topn` instead.")
            topn = num_words

        topics = self.lambda_[:, :, time]
        topic = topics[topicid]
        # likelihood to probability
        topic = np.exp(topic)
        # normalize to probability dist
        topic = topic / topic.sum()
        # sort according to prob
        bestn = matutils.argsort(topic, topn, reverse=True)
        beststr = [(topic[idx], self.id2word[idx]) for idx in bestn]
        return beststr

    def print_topic(self, topicid, time, topn=10, num_words=None):
        """Get the given topic, formatted as a string.

        Parameters
        ----------
        topicid : int
            Id of topic.
        time : int
            Timestamp.
        topn : int, optional
            Top number of topics that you'll receive.
        num_words : int, optional
            DEPRECATED PARAMETER, use `topn` instead.

        Returns
        -------
        str
            The given topic in string format, like '0.132*someword + 0.412*otherword + ...'.

        """
        if num_words is not None:  # deprecated num_words is used
            warnings.warn("The parameter `num_words` is deprecated, will be removed in 4.0.0, use `topn` instead.")
            topn = num_words

        return ' + '.join(['%.3f*%s' % v for v in self.show_topic(topicid, time, topn)])

    def dtm_vis(self, corpus, time):
        """Get data specified by pyLDAvis format.

        Parameters
        ----------
        corpus : iterable of iterable of (int, float)
            Collection of texts in BoW format.
        time : int
            Sequence of timestamp.

        Notes
        -----
        All of these are needed to visualise topics for DTM for a particular time-slice via pyLDAvis.

        Returns
        -------
        doc_topic : numpy.ndarray
            Document-topic proportions.
        topic_term : numpy.ndarray
            Calculated term of topic suitable for pyLDAvis format.
        doc_lengths : list of int
            Length of each documents in corpus.
        term_frequency : numpy.ndarray
            Frequency of each word from vocab.
        vocab : list of str
            List of words from docpus.

        """
        topic_term = np.exp(self.lambda_[:, :, time]) / np.exp(self.lambda_[:, :, time]).sum()
        topic_term *= self.num_topics

        doc_topic = self.gamma_

        doc_lengths = [len(doc) for doc_no, doc in enumerate(corpus)]

        term_frequency = np.zeros(len(self.id2word))
        for doc_no, doc in enumerate(corpus):
            for pair in doc:
                term_frequency[pair[0]] += pair[1]

        vocab = [self.id2word[i] for i in range(0, len(self.id2word))]
        # returns numpy arrays for doc_topic proportions, topic_term proportions, and document_lengths, term_frequency.
        # these should be passed to the `pyLDAvis.prepare` method to visualise one time-slice of DTM topics.
        return doc_topic, topic_term, doc_lengths, term_frequency, vocab

    def dtm_coherence(self, time, num_words=20):
        """Get all topics of a particular time-slice without probability values for it to be used.
        For either "u_mass" or "c_v" coherence.

        Parameters
        ----------
        num_words : int
            Number of words.
        time : int
            Timestamp

        Returns
        -------
        coherence_topics : list of list of str
            All topics of a particular time-slice without probability values for it to be used.

        Warnings
        --------
        TODO: because of print format right now can only return for 1st time-slice, should we fix the coherence
        printing or make changes to the print statements to mirror DTM python?

        """
        coherence_topics = []
        for topic_no in range(0, self.num_topics):
            topic = self.show_topic(topicid=topic_no, time=time, num_words=num_words)
            coherence_topic = []
            for prob, word in topic:
                coherence_topic.append(word)
            coherence_topics.append(coherence_topic)

        return coherence_topics
