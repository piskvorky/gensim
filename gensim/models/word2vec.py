#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Produce word vectors with deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

NOTE: There are more ways to get word vectors in Gensim than just Word2Vec.
See wrappers for FastText, VarEmbed and WordRank.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews,
visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training**
(70x speedup compared to plain NumPy implementation [3]_).

Initialize a model with e.g.::

    >>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

    >>> model.save(fname)
    >>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The word vectors are stored in a KeyedVectors instance in model.wv.
This separates the read-only word vector lookup operations in KeyedVectors from the training code in Word2Vec::

  >>> model.wv['computer']  # numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

The word vectors can also be instantiated from an existing file on disk in the word2vec C format
as a KeyedVectors instance::

    NOTE: It is impossible to continue training the vectors loaded from the C format because hidden weights,
    vocabulary frequency and the binary tree is missing::

        >>> from gensim.models.keyedvectors import KeyedVectors
        >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
        >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format


You can perform various NLP word tasks with the model. Some of them
are already built-in::

  >>> model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.71382287), ...]


  >>> model.wv.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.wv.similarity('woman', 'man')
  0.73723527

Probability of a text under the model::

  >>> model.score(["The fox jumped over a lazy dog".split()])
  0.2158356

Correlation with human opinion on word similarity::

  >>> model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
  0.51, 0.62, 0.13

And on analogies::

  >>> model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

and so on.

If you're finished training a model (i.e. no more updates, only querying),
then switch to the :mod:`gensim.models.KeyedVectors` instance in wv

  >>> word_vectors = model.wv
  >>> del model

to trim unneeded model memory = use much less RAM.

Note that there is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word. Using phrases, you can learn a word2vec model
where "words" are actually multiword expressions, such as `new_york_times` or `financial_crisis`:

    >>> bigram_transformer = gensim.models.Phrases(sentences)
    >>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""
from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict
import threading
import itertools
import warnings

from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.models.keyedvectors import Word2VecKeyedVectors, Vocab
from gensim.models.base_any2vec import BaseAny2VecModel,\
    BaseVocabBuilder, BaseModelTrainables

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000
    raise RuntimeError("Support for Python/Numpy implementations has been continued.")


class Word2Vec(BaseAny2VecModel):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    If you're finished training a model (=no more updates, only querying)
    then switch to the :mod:`gensim.models.KeyedVectors` instance in wv

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `wv.save_word2vec_format()`
    and `KeyedVectors.load_word2vec_format()`.

    """

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=0`), CBOW is used.
        Otherwise (`sg=1`), skip-gram is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).
        Note that for a fully deterministically-reproducible run, you must also limit the model to
        a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
        3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
        environment variable to control hash randomization.)

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1, hierarchical softmax will be used for model training.
        If set to 0 (default), and `negative` is non-zero, negative sampling will be used.

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.

        `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
        Only applies when cbow is used.

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
        returns either `utils.RULE_DISCARD`, `utils.RULE_KEEP` or `utils.RULE_DEFAULT`.
        Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
        of the model.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

        `batch_words` = target size (in words) for batches of examples passed to worker threads (and
        thus cython routines). Default is 10000. (Larger batches will be passed if individual
        texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        """

        self.callbacks = callbacks
        self.load = call_on_class_only

        if FAST_VERSION == -1:
            logger.warning('Slow version of %s is being used', __name__)
        else:
            logger.debug('Fast version of %s is being used', __name__)

        self.sg = int(sg)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.random = random.RandomState(seed)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.null_word = null_word
        self.compute_loss = compute_loss
        self.running_training_loss = 0

        self.wv = Word2VecKeyedVectors()
        self.vocabulary = Word2VecVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word)
        self.trainables = Word2VecTrainables(
            vector_size=size, seed=seed, alpha=alpha, min_alpha=min_alpha, hs=hs, negative=negative, hashfxn=hashfxn)

        super(Word2Vec, self).__init__(workers=workers, vector_size=size, epochs=iter,
            callbacks=callbacks, batch_words=batch_words)

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(
                sentences, total_examples=self.vocabulary.corpus_count, epochs=self.epochs,
                start_alpha=self.trainables.alpha, end_alpha=self.trainables.min_alpha
            )
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored."
                )

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        """
        Build vocabulary from a dictionary of word frequencies.
        Build model vocabulary from a passed dictionary that contains (word,word count).
        Words must be of type unicode strings.

        Parameters
        ----------
        `word_freq` : dict
            Word,Word_Count dictionary.
        `keep_raw_vocab` : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        `corpus_count`: int
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
        returns either `utils.RULE_DISCARD`, `utils.RULE_KEEP` or `utils.RULE_DEFAULT`.
        `update`: bool
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        Returns
        --------
        None

        Examples
        --------
        >>> from gensim.models.word2vec import Word2Vec
        >>> model= Word2Vec()
        >>> model.build_vocab_from_freq({"Word1": 15, "Word2": 20})
        """
        logger.info("Processing provided word frequencies")
        # Instead of scanning text, this will assign provided word frequencies dictionary(word_freq)
        # to be directly the raw vocab
        raw_vocab = word_freq
        logger.info(
            "collected %i different raw word, with total frequency of %i",
            len(raw_vocab), sum(itervalues(raw_vocab))
        )

        # Since no sentences are provided, this is to control the corpus_count
        self.vocabulary.corpus_count = corpus_count if corpus_count else 0
        self.vocabulary.raw_vocab = raw_vocab

        # trim by min_count & precalculate downsampling
        self.vocabulary.prepare_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        self.trainables.prepare_weights(update=update, vocabulary=self.vocabulary)  # build tables & arrays
        self._set_keyedvectors()

    def reset_from(self, other_model):
        """
        Borrow shareable pre-built structures (like vocab) from the other_model. Useful
        if testing multiple models in parallel on the same corpus.
        """
        self.wv.vocab = other_model.wv.vocab
        self.wv.index2word = other_model.wv.index2word
        self.trainables.cum_table = other_model.trainables.cum_table
        self.vocabulary.corpus_count = other_model.vocabulary.corpus_count
        self.trainables.reset_weights(vocabulary=self.vocabulary)

    def _set_keyedvectors(self):
        """Point `keyedvectors` attributes to corresponding `trainables`"""
        self.wv.vectors = self.trainables.vectors
        self.wv.vector_size = self.trainables.vector_size
        self.wv.vocab = self.vocabulary.vocab
        self.wv.index2word = self.vocabulary.index2word

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, self.compute_loss)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1, self.compute_loss)
        return tally, self._raw_word_count(sentences)

    def _get_job_params(self):
        """Return the paramter required for each batch"""
        if self.alpha > self.min_alpha_yet_reached:
            logger.warning("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = self.alpha
        return self.alpha

    def _update_job_params(self, job_params, progress, cur_epoch):
        start_alpha = self.trainables.alpha
        end_alpha = self.trainables.min_alpha
        progress *= (float(cur_epoch + 1) / float(self.epochs))
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)
        return next_alpha

    def _get_thread_working_mem(self):
        work = matutils.zeros_aligned(self.trainables.vector_size, dtype=REAL)  # per-thread private work memory
        neu1 = matutils.zeros_aligned(self.trainables.vector_size, dtype=REAL)
        return work, neu1

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def _clear_post_train(self):
        """Resets certain properties of the model, post training. eg. `kv.syn0norm`"""
        self.wv.vectors_norm = None

    def _set_train_params(self, **kwargs):
        if 'compute_loss' in kwargs:
            self.compute_loss = kwargs['compute_loss']
        self.running_training_loss = 0

    def _set_params_from_kv(self):
        self.trainables.vectors = self.wv.vectors
        self.trainables.vector_size = self.wv.vector_size
        self.vocabulary.vocab = self.wv.vocab
        self.vocabulary.index2word = self.wv.index2word

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        if self.wv:
            self._set_params_from_kv()
        if self.model_trimmed_post_training:
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

        if not self.vocabulary.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.trainables.vectors):
            raise RuntimeError("you must initialize vectors before training the model")

        if not hasattr(self.vocabulary, 'corpus_count'):
            raise ValueError(
                "The number of examples in the training corpus is missing. "
                "Please make sure this is set inside `build_vocab` function."
                "Call the `build_vocab` function before calling `train`."
            )

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper job parameters updation"
                "and progress calculations. "
                "The usual value is total_examples=model.corpus_count."
            )
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.epochs.")

    # basics copied from the train() function
    def score(self, sentences, total_sentences=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """
        Score the log probability for a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        This does not change the fitted model in any way (see Word2Vec.train() for that).

        We have currently only implemented score for the hierarchical softmax scheme,
        so you need to have run word2vec with hs=1 and negative=0 for this to work.

        Note that you should specify total_sentences; we'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the article by [#taddy]_ and the gensim demo at [#deepir]_ for examples of
        how to use such scores in document classification.

        .. [#taddy] Taddy, Matt.  Document Classification by Inversion of Distributed Language Representations,
                    in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
        .. [#deepir] https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb

        """
        if FAST_VERSION < 0:
            warnings.warn(
                "C extension compilation failed, scoring will be slow. "
                "Install a C compiler and reinstall gensim for fastness."
            )

        logger.info(
            "scoring sentences with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.wv.vocab), self.vector_size, self.sg, self.hs, self.vocabulary.sample, self.negative
        )

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        if not self.hs:
            raise RuntimeError(
                "We have currently only implemented score for the hierarchical softmax scheme, "
                "so you need to have run word2vec with hs=1 and negative=0 for this to work."
            )

        def worker_loop():
            """Compute log probability for each sentence, lifting lists of sentences from the jobs queue."""
            work = zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
            neu1 = matutils.zeros_aligned(self.vector_size, dtype=REAL)
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    if sentence_id >= total_sentences:
                        break
                    if self.sg:
                        score = score_sentence_sg(self, sentence, work)
                    else:
                        score = score_sentence_cbow(self, sentence, work, neu1)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = matutils.zeros_aligned(total_sentences, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(utils.grouper(enumerate(sentences), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_sentences:
                    logger.warning(
                        "terminating after %i sentences (set higher total_sentences if you want more).",
                        total_sentences
                    )
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info("reached end of input; waiting to finish %i outstanding jobs", job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed
                        )
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.clear_sims()
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed
        )
        return sentence_scores[:sentence_count]

    def clear_sims(self):
        """
        Removes all L2-normalized vectors for words from the model.
        You will have to recompute them using init_sims method.
        """

        self.wv.vectors_norm = None

    def intersect_word2vec_format(self, fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict'):
        """
        Merge the input-hidden weight matrix from the original C word2vec-tool format
        given, where it intersects with the current vocabulary. (No words are added to the
        existing vocabulary, but intersecting words adopt the file's weights, and
        non-intersecting words are left alone.)

        `binary` is a boolean indicating whether the data is in binary word2vec format.

        `lockf` is a lock-factor value to be set for any imported word-vectors; the
        default value of 0.0 prevents further updating of the vector during subsequent
        training. Use 1.0 to allow further training updates of merged vectors.
        """
        overlap_count = 0
        logger.info("loading projection weights from %s", fname)
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
            if not vector_size == self.vector_size:
                raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
                # TOCONSIDER: maybe mismatched vectors still useful enough to merge (truncating/padding)?
            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for _ in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    if word in self.wv.vocab:
                        overlap_count += 1
                        self.wv.syn0[self.wv.vocab[word].index] = weights
                        self.syn0_lockf[self.wv.vocab[word].index] = lockf  # lock-factor: 0.0 stops further changes
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    word, weights = parts[0], [REAL(x) for x in parts[1:]]
                    if word in self.wv.vocab:
                        overlap_count += 1
                        self.wv.syn0[self.wv.vocab[word].index] = weights
                        self.syn0_lockf[self.wv.vocab[word].index] = lockf  # lock-factor: 0.0 stops further changes
        logger.info("merged %d vectors into %s matrix from %s", overlap_count, self.wv.syn0.shape, fname)

    @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar() instead")
    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        """
        Deprecated. Use self.wv.most_similar() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.most_similar`
        """
        return self.wv.most_similar(positive, negative, topn, restrict_vocab, indexer)

    @deprecated("Method will be removed in 4.0.0, use self.wv.wmdistance() instead")
    def wmdistance(self, document1, document2):
        """
        Deprecated. Use self.wv.wmdistance() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.wmdistance`
        """
        return self.wv.wmdistance(document1, document2)

    @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead")
    def most_similar_cosmul(self, positive=None, negative=None, topn=10):
        """
        Deprecated. Use self.wv.most_similar_cosmul() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.most_similar_cosmul`
        """
        return self.wv.most_similar_cosmul(positive, negative, topn)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_word() instead")
    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Deprecated. Use self.wv.similar_by_word() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.similar_by_word`
        """
        return self.wv.similar_by_word(word, topn, restrict_vocab)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_vector() instead")
    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Deprecated. Use self.wv.similar_by_vector() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.similar_by_vector`
        """
        return self.wv.similar_by_vector(vector, topn, restrict_vocab)

    @deprecated("Method will be removed in 4.0.0, use self.wv.doesnt_match() instead")
    def doesnt_match(self, words):
        """
        Deprecated. Use self.wv.doesnt_match() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.doesnt_match`
        """
        return self.wv.doesnt_match(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.__getitem__() instead")
    def __getitem__(self, words):
        """
        Deprecated. Use self.wv.__getitem__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__getitem__`
        """
        return self.wv.__getitem__(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.__contains__() instead")
    def __contains__(self, word):
        """
        Deprecated. Use self.wv.__contains__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__contains__`
        """
        return self.wv.__contains__(word)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similarity() instead")
    def similarity(self, w1, w2):
        """
        Deprecated. Use self.wv.similarity() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.similarity`
        """
        return self.wv.similarity(w1, w2)

    @deprecated("Method will be removed in 4.0.0, use self.wv.n_similarity() instead")
    def n_similarity(self, ws1, ws2):
        """
        Deprecated. Use self.wv.n_similarity() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.n_similarity`
        """
        return self.wv.n_similarity(ws1, ws2)

    def predict_output_word(self, context_words_list, topn=10):
        """Report the probability distribution of the center word given the context words
        as input to the trained model."""
        if not self.negative:
            raise RuntimeError(
                "We have currently only implemented predict_output_word for the negative sampling scheme, "
                "so you need to have run word2vec with negative > 0 for this to work."
            )

        if not hasattr(self.wv, 'vectors') or not hasattr(self.trainables, 'syn1neg'):
            raise RuntimeError("Parameters required for predicting the output words not found.")

        word_vocabs = [self.wv.vocab[w] for w in context_words_list if w in self.wv.vocab]
        if not word_vocabs:
            warnings.warn("All the input context words are out-of-vocabulary for the current model.")
            return None

        word2_indices = [word.index for word in word_vocabs]

        l1 = np_sum(self.wv.vectors[word2_indices], axis=0)
        if word2_indices and self.cbow_mean:
            l1 /= len(word2_indices)

        prob_values = exp(dot(l1, self.trainables.syn1neg.T))  # propagate hidden -> output and take softmax to get probabilities
        prob_values /= sum(prob_values)
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
        # returning the most probable output words with their probabilities
        return [(self.wv.index2word[index1], prob_values[index1]) for index1 in top_indices]

    def init_sims(self, replace=False):
        """
        init_sims() resides in KeyedVectors because it deals with syn0 mainly, but because syn1 is not an attribute
        of KeyedVectors, it has to be deleted in this class, and the normalizing of syn0 happens inside of KeyedVectors
        """
        if replace and hasattr(self.trainables, 'syn1'):
            del self.trainables.syn1
        return self.wv.init_sims(replace)

    # def estimate_memory(self, vocab_size=None, report=None):
    #     """Estimate required memory for a model using current settings and provided vocabulary size."""
    #     vocab_size = vocab_size or len(self.vocabulary.vocab)
    #     report = report or {}
    #     report['vocab'] = vocab_size * (700 if self.trainables.hs else 500)
    #     report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
    #     if self.trainables.hs:
    #         report['syn1'] = vocab_size * self.vector_size * dtype(REAL).itemsize
    #     if self.trainables.negative:
    #         report['syn1neg'] = vocab_size * self.vector_size * dtype(REAL).itemsize
    #     report['total'] = sum(report.values())
    #     logger.info(
    #         "estimated required memory for %i words and %i dimensions: %i bytes",
    #         vocab_size, self.vector_size, report['total']
    #     )
    #     return report

    @staticmethod
    def log_accuracy(section):
        return Word2VecKeyedVectors.log_accuracy(section)

    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True):
        most_similar = most_similar or Word2VecKeyedVectors.most_similar
        return self.wv.accuracy(questions, restrict_vocab, most_similar, case_insensitive)

    @staticmethod
    @deprecated("Method will be removed in 4.0.0, use self.wv.log_evaluate_word_pairs() instead")
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        """
        Deprecated. Use self.wv.log_evaluate_word_pairs() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.log_evaluate_word_pairs`
        """
        return Word2VecKeyedVectors.log_evaluate_word_pairs(pearson, spearman, oov, pairs)

    @deprecated("Method will be removed in 4.0.0, use self.wv.evaluate_word_pairs() instead")
    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False):
        """
        Deprecated. Use self.wv.evaluate_word_pairs() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.evaluate_word_pairs`
        """
        return self.wv.evaluate_word_pairs(pairs, delimiter, restrict_vocab, case_insensitive, dummy4unknown)

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.wv.index2word), self.vector_size, self.alpha
        )

    @deprecated(
        "Method will be removed in 4.0.0, keep just_word_vectors = model.wv to retain just the KeyedVectors instance"
    )
    def _minimize_model(self, save_syn1=False, save_syn1neg=False, save_syn0_lockf=False):
        if save_syn1 and save_syn1neg and save_syn0_lockf:
            return
        if hasattr(self.trainables, 'syn1') and not save_syn1:
            del self.trainables.syn1
        if hasattr(self.trainables, 'syn1neg') and not save_syn1neg:
            del self.trainables.syn1neg
        if hasattr(self.trainables, 'syn0_lockf') and not save_syn0_lockf:
            del self.trainables.vectors_lockf
        self.model_trimmed_post_training = True

    def delete_temporary_training_data(self, replace_word_vectors_with_normalized=False):
        """
        Discard parameters that are used in training and score. Use if you're sure you're done training a model.
        If `replace_word_vectors_with_normalized` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        """
        if replace_word_vectors_with_normalized:
            self.init_sims(replace=True)
        self._minimize_model()

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['wv.vectors_norm', 'trainables.cum_table'])
        # FIXME: handle ignoring nested attributes
        super(Word2Vec, self).save(*args, **kwargs)

    save.__doc__ = utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(Word2Vec, cls).load(*args, **kwargs)
        # update older models
        if hasattr(model, 'table'):
            delattr(model, 'table')  # discard in favor of cum_table
        if model.trainables.negative and hasattr(model.wv, 'index2word'):
            model.trainables.make_cum_table(vocabulary=model.vocabulary)  # rebuild cum_table from vocabulary
        if not hasattr(model.vocabulary, 'corpus_count'):
            model.corpus_count = None
        for v in model.wv.vocab.values():
            if hasattr(v, 'sample_int'):
                break  # already 0.12.0+ style int probabilities
            elif hasattr(v, 'sample_probability'):
                v.sample_int = int(round(v.sample_probability * 2**32))
                del v.sample_probability
        if not hasattr(model.trainables, 'vectors_lockf') and hasattr(model.trainables, 'vectors'):
            model.trainables.vectors_lockf = ones(len(model.wv.vectors), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.trainables.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    def _load_specials(self, *args, **kwargs):
        super(Word2Vec, self)._load_specials(*args, **kwargs)
        # loading from a pre-KeyedVectors word2vec model
        if not hasattr(self, 'wv'):
            wv = Word2VecKeyedVectors()
            if hasattr(self, 'trainables'):
                wv.vectors = self.trainables.__dict__.get('vectors', [])
                wv.vectors_norm = self.trainables.__dict__.get('vectors_norm', None)
            if hasattr(self, 'vocabulary'):
                wv.vocab = self.vocabulary.__dict__.get('vocab', {})
                wv.index2word = self.vocabulary.__dict__.get('index2word', [])
            self.wv = wv

    @classmethod
    @deprecated("Method will be removed in 4.0.0, use gensim.models.KeyedVectors.load_word2vec_format instead")
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                         limit=None, datatype=REAL):
        """Deprecated. Use gensim.models.KeyedVectors.load_word2vec_format instead."""
        raise DeprecationWarning("Deprecated. Use gensim.models.KeyedVectors.load_word2vec_format instead.")

    @deprecated("Method will be removed in 4.0.0, use model.wv.save_word2vec_format instead")
    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """Deprecated. Use model.wv.save_word2vec_format instead."""
        raise DeprecationWarning("Deprecated. Use model.wv.save_word2vec_format instead.")

    def get_latest_training_loss(self):
        return self.running_training_loss


class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""

    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.smart_open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = utils.to_unicode(text).split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


class LineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or not clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


class PathLineSentences(object):
    """

    Works like word2vec.LineSentence, but will process all files in a directory in alphabetical order by filename.
    The directory can only contain files that can be read by LineSentence: .bz2, .gz, and text files.
    Any file not ending with .bz2 or .gz is assumed to be a text file. Does not work with subdirectories.

    The format of files (either text, or compressed text files) in the path is one sentence = one line,
    with words already preprocessed and separated by whitespace.

    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` should be a path to a directory (as a string) where all files can be opened by the
        LineSentence class. Each file will be read up to `limit` lines (or not clipped if limit is None, the default).

        Example::

            sentences = PathLineSentences(os.getcwd() + '\\corpus\\')

        The files in the directory should be either text files, .bz2 files, or .gz files.

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with utils.smart_open(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


class Word2VecVocab(BaseVocabBuilder):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0):
        super(Word2VecVocab, self).__init__()
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.null_word = null_word

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warning(
                        "Each 'sentences' item should be a list of words (usually unicode strings). "
                        "First item here is instead plain %s.",
                        type(sentence)
                    )
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                    sentence_no, total_words, len(vocab)
                )
            for word in sentence:
                vocab[word] += 1
            total_words += len(sentence)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(vocab), total_words, sentence_no + 1
        )
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        return total_words

    def sort_vocab(self, vectors):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if len(vectors):
            raise RuntimeError("cannot sort vocabulary after model weights already initialized.")
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i

    def prepare_vocab(
        self, update=False, keep_raw_vocab=False, trim_rule=None, min_count=None, sample=None, dry_run=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            logger.info("Loading a fresh vocabulary")
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                self.index2word = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                self.vocab = {}

            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                        self.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
            logger.info(
                "min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique
            )
            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            logger.info(
                "min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                min_count, retain_total, retain_pct, original_total, drop_total
            )
        else:
            logger.info("Updating model with new vocabulary")
            new_total = pre_exist_total = 0
            new_words = pre_exist_words = []
            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    if word in self.vocab:
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            self.vocab[word].count += v
                    else:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                            self.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info(
                "New added %i unique words (%i%% of original %i) "
                "and increased the count of %i pre-existing words (%i%% of original %i)",
                len(new_words), new_unique_pct, original_unique_total, len(pre_exist_words),
                pre_exist_unique_pct, original_unique_total
            )
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info(
            "downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
            downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total
        )

        # return from each step: words-affected, resulting-corpus-size, extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total)  # 'memory': Word2Vec.estimate_memory(vocab_size=len(retain_words))
        }

        return report_values

    def add_null_word(self):
        word, v = '\0', Vocab(count=1, sample_int=0)
        v.index = len(self.wv.vocab)
        self.index2word.append(word)
        self.vocab[word] = v


class Word2VecTrainables(BaseModelTrainables):
    def __init__(self, vector_size=100, seed=1, alpha=0.025, min_alpha=0.0001, hs=0, negative=5, hashfxn=hash):
        super(Word2VecTrainables, self).__init__(vector_size=vector_size, seed=seed)
        self.cum_table = None  # for negative sampling
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.hashfxn = hashfxn

    def prepare_weights(self, update=False, vocabulary=None):
        """Build tables and model weights based on final vocabulary settings."""
        # if not self.wv.index2word:
        #     self.scale_vocab()
        if vocabulary.sorted_vocab and not update:
            vocabulary.sort_vocab(self.vectors)
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree(vocabulary=vocabulary)
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table(vocabulary=vocabulary)
        if vocabulary.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            vocabulary.add_null_word()
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights(vocabulary=vocabulary)
        else:
            self.update_weights(vocabulary=vocabulary)

    def create_binary_tree(self, vocabulary=None):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words", len(vocabulary.vocab))

        # build the huffman tree
        heap = list(itervalues(vocabulary.vocab))
        heapq.heapify(heap)
        for i in xrange(len(vocabulary.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(
                heap, Vocab(count=min1.count + min2.count, index=i + len(vocabulary.vocab), left=min1, right=min2)
            )

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(vocabulary.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(vocabulary.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)

    def make_cum_table(self, vocabulary=None, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(vocabulary.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += vocabulary.vocab[vocabulary.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += vocabulary.vocab[vocabulary.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def reset_weights(self, vocabulary=None):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.vectors = empty((len(vocabulary.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(vocabulary.vocab)):
            # construct deterministic seed from word AND seed argument
            self.vectors[i] = self.seeded_vector(vocabulary.index2word[i] + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(vocabulary.vocab), self.vector_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(vocabulary.vocab), self.vector_size), dtype=REAL)
        self.vectors_norm = None

        self.vectors_lockf = ones(len(vocabulary.vocab), dtype=REAL)  # zeros suppress learning

    def update_weights(self, vocabulary=None):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info("updating layer weights")
        gained_vocab = len(vocabulary.vocab) - len(self.vectors)
        newsyn0 = empty((gained_vocab, self.vector_size), dtype=REAL)

        # randomize the remaining words
        for i in xrange(len(self.vectors), len(vocabulary.vocab)):
            # construct deterministic seed from word AND seed argument
            newsyn0[i - len(self.vectors)] = self.seeded_vector(vocabulary.index2word[i] + str(self.seed))

        # Raise an error if an online update is run before initial training on a corpus
        if not len(self.vectors):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus before doing an online update."
            )

        self.vectors = vstack([self.vectors, newsyn0])

        if self.hs:
            self.syn1 = vstack([self.syn1, zeros((gained_vocab, self.vector_size), dtype=REAL)])
        if self.negative:
            self.syn1neg = vstack([self.syn1neg, zeros((gained_vocab, self.vector_size), dtype=REAL)])
        self.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = ones(len(vocabulary.vocab), dtype=REAL)  # zeros suppress learning


# Example: ./word2vec.py -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 \
# -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    logger.info("running %s", " ".join(sys.argv))
    logger.info("using optimization %s", FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    from gensim.models.word2vec import Word2Vec  # noqa:F811 avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="Use text data from file TRAIN to train the model", required=True)
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors")
    parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5)
    parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=100)
    parser.add_argument(
        "-sample",
        help="Set threshold for occurrence of words. "
             "Those that appear with higher frequency in the training data will be randomly down-sampled;"
             " default is 1e-3, useful range is (0, 1e-5)",
        type=float, default=1e-3
    )
    parser.add_argument(
        "-hs", help="Use Hierarchical Softmax; default is 0 (not used)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument(
        "-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)",
        type=int, default=5
    )
    parser.add_argument("-threads", help="Use THREADS threads (default 12)", type=int, default=12)
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)
    parser.add_argument(
        "-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5",
        type=int, default=5
    )
    parser.add_argument(
        "-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)",
        type=int, default=1, choices=[0, 1]
    )
    parser.add_argument(
        "-binary", help="Save the resulting vectors in binary mode; default is 0 (off)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")

    args = parser.parse_args()

    if args.cbow == 0:
        skipgram = 1
    else:
        skipgram = 0

    corpus = LineSentence(args.train)

    model = Word2Vec(
        corpus, size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, sg=skipgram, hs=args.hs,
        negative=args.negative, cbow_mean=1, iter=args.iter
    )

    if args.output:
        outfile = args.output
        model.wv.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train
        model.save(outfile + '.model')
    if args.binary == 1:
        model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
    else:
        model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        model.accuracy(args.accuracy)

    logger.info("finished running %s", program)
