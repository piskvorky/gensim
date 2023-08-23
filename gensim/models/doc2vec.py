#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Introduction
============

Learn paragraph and document embeddings via the distributed memory and distributed bag of words models from
`Quoc Le and Tomas Mikolov: "Distributed Representations of Sentences and Documents"
<http://arxiv.org/pdf/1405.4053v2.pdf>`_.

The algorithms use either hierarchical softmax or negative sampling; see
`Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean: "Efficient Estimation of Word Representations in
Vector Space, in Proceedings of Workshop at ICLR, 2013" <https://arxiv.org/pdf/1301.3781.pdf>`_ and
`Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean: "Distributed Representations of Words
and Phrases and their Compositionality. In Proceedings of NIPS, 2013"
<https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_.

For a usage example, see the `Doc2vec tutorial
<https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py>`_.

**Make sure you have a C compiler before installing Gensim, to use the optimized doc2vec routines** (70x speedup
compared to plain NumPy implementation, https://rare-technologies.com/parallelizing-word2vec-in-python/).


Usage examples
==============

Initialize & train a model:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    >>>
    >>> documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    >>> model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

Persist a model to disk:

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>>
    >>> fname = get_tmpfile("my_doc2vec_model")
    >>>
    >>> model.save(fname)
    >>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

Infer vector for a new document:

.. sourcecode:: pycon

    >>> vector = model.infer_vector(["system", "response"])

"""

import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer

from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
except ImportError:
    raise utils.NO_CYTHON

try:
    from gensim.models.doc2vec_corpusfile import (
        d2v_train_epoch_dbow,
        d2v_train_epoch_dm_concat,
        d2v_train_epoch_dm,
        CORPUSFILE_VERSION
    )
except ImportError:
    # corpusfile doc2vec is not supported
    CORPUSFILE_VERSION = -1

    def d2v_train_epoch_dbow(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples,
                             _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None,
                             train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                             doctag_vectors=None, doctag_locks=None):
        raise NotImplementedError("Training with corpus_file argument is not supported.")

    def d2v_train_epoch_dm_concat(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch,
                                  _expected_examples, _expected_words, work, _neu1, docvecs_count, word_vectors=None,
                                  word_locks=None, learn_doctags=True, learn_words=True, learn_hidden=True,
                                  doctag_vectors=None, doctag_locks=None):
        raise NotImplementedError("Training with corpus_file argument is not supported.")

    def d2v_train_epoch_dm(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples,
                           _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None,
                           learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None,
                           doctag_locks=None):
        raise NotImplementedError("Training with corpus_file argument is not supported.")


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """Represents a document along with a tag, input document format for :class:`~gensim.models.doc2vec.Doc2Vec`.

    A single document, made up of `words` (a list of unicode string tokens) and `tags` (a list of tokens).
    Tags may be one or more unicode string tokens, but typical practice (which will also be the most memory-efficient)
    is for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from :class:`gensim.models.word2vec.Word2Vec`.

    """
    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).

        """
        return '%s<%s, %s>' % (self.__class__.__name__, self.words, self.tags)


@dataclass
class Doctag:
    """A dataclass shape-compatible with keyedvectors.SimpleVocab, extended to record
    details of string document tags discovered during the initial vocabulary scan.

    Will not be used if all presented document tags are ints. No longer used in a
    completed model: just used during initial scan, and for backward compatibility.
    """
    __slots__ = ('doc_count', 'index', 'word_count')
    doc_count: int  # number of docs where tag appeared
    index: int  # position in underlying array
    word_count: int  # number of words in associated docs

    @property
    def count(self):
        return self.doc_count

    @count.setter
    def count(self, new_val):
        self.doc_count = new_val


class Doc2Vec(Word2Vec):
    def __init__(
            self, documents=None, corpus_file=None, vector_size=100, dm_mean=None, dm=1, dbow_words=0, dm_concat=0,
            dm_tag_count=1, dv=None, dv_mapfile=None, comment=None, trim_rule=None, callbacks=(),
            window=5, epochs=10, shrink_windows=True, **kwargs,
        ):
        """Class for training, using and evaluating neural networks described in
        `Distributed Representations of Sentences and Documents <http://arxiv.org/abs/1405.4053v2>`_.

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Input corpus, can be simply a list of elements, but for larger corpora,consider an iterable that streams
            the documents directly from disk/network. If you don't supply `documents` (or `corpus_file`), the model is
            left uninitialized -- use if you plan to initialize it in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).
            Documents' tags are assigned automatically and are equal to line number, as in
            :class:`~gensim.models.doc2vec.TaggedLineDocument`.
        dm : {1,0}, optional
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        vector_size : int, optional
            Dimensionality of the feature vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling.
            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`
            environment variable to control hash randomization.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        epochs : int, optional
            Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        dm_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm_concat : {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using
            dm_concat mode.
        dbow_words : {1,0}, optional
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.
        shrink_windows : bool, optional
            New in 4.1. Experimental.
            If True, the effective window size is uniformly sampled from  [1, `window`]
            for each target word during training, to match the original word2vec algorithm's
            approximate weighting of context words by distance. Otherwise, the effective
            window size is always fixed to `window` words to either side.

        Some important internal attributes are the following:

        Attributes
        ----------
        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object essentially contains the mapping between words and embeddings. After training, it can be used
            directly to query those embeddings in various ways. See the module level docstring for examples.

        dv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object contains the paragraph vectors learned from the training data. There will be one such vector
            for each unique document tag supplied during training. They may be individually accessed using the tag
            as an indexed-access key. For example, if one of the training documents used a tag of 'doc003':

            .. sourcecode:: pycon

                >>> model.dv['doc003']

        """
        corpus_iterable = documents

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = int(dbow_words)
        self.dm_concat = int(dm_concat)
        self.dm_tag_count = int(dm_tag_count)
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + (2 * window)) * vector_size
            logger.info("using concatenative %d-dimensional layer1", self.layer1_size)

        self.vector_size = vector_size
        self.dv = dv or KeyedVectors(self.vector_size, mapfile_path=dv_mapfile)
        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as desired after any vocab growth
        self.dv.vectors_lockf = np.ones(1, dtype=REAL)  # 0.0 values suppress word-backprop-updates; 1.0 allows

        super(Doc2Vec, self).__init__(
            sentences=corpus_iterable,
            corpus_file=corpus_file,
            vector_size=self.vector_size,
            sg=(1 + dm) % 2,
            null_word=self.dm_concat,
            callbacks=callbacks,
            window=window,
            epochs=epochs,
            shrink_windows=shrink_windows,
            **kwargs,
        )

    @property
    def dm(self):
        """Indicates whether 'distributed memory' (PV-DM) will be used, else 'distributed bag of words'
        (PV-DBOW) is used.

        """
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        """Indicates whether 'distributed bag of words' (PV-DBOW) will be used, else 'distributed memory'
        (PV-DM) is used.

        """
        return self.sg  # same as SG

    @property
    @deprecated("The `docvecs` property has been renamed `dv`.")
    def docvecs(self):
        return self.dv

    @docvecs.setter
    @deprecated("The `docvecs` property has been renamed `dv`.")
    def docvecs(self, value):
        self.dv = value

    def _clear_post_train(self):
        """Resets the current word vectors. """
        self.wv.norms = None
        self.dv.norms = None

    def init_weights(self):
        super(Doc2Vec, self).init_weights()
        # to not use an identical rnd stream as words, deterministically change seed (w/ 1000th prime)
        self.dv.resize_vectors(seed=self.seed + 7919)

    def reset_from(self, other_model):
        """Copy shareable data structures from another (possibly pre-trained) model.

        This specifically causes some structures to be shared, so is limited to
        structures (like those rleated to the known word/tag vocabularies) that
        won't change during training or thereafter. Beware vocabulary edits/updates
        to either model afterwards: the partial sharing and out-of-band modification
        may leave the other model in a broken state.

        Parameters
        ----------
        other_model : :class:`~gensim.models.doc2vec.Doc2Vec`
            Other model whose internal data structures will be copied over to the current object.

        """
        self.wv.key_to_index = other_model.wv.key_to_index
        self.wv.index_to_key = other_model.wv.index_to_key
        self.wv.expandos = other_model.wv.expandos
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.dv.key_to_index = other_model.dv.key_to_index
        self.dv.index_to_key = other_model.dv.index_to_key
        self.dv.expandos = other_model.dv.expandos
        self.init_weights()

    def _do_train_epoch(
        self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
        total_examples=None, total_words=None, offsets=None, start_doctags=None, **kwargs
    ):
        work, neu1 = thread_private_mem
        doctag_vectors = self.dv.vectors
        doctags_lockf = self.dv.vectors_lockf

        offset = offsets[thread_id]
        start_doctag = start_doctags[thread_id]

        if self.sg:
            examples, tally, raw_tally = d2v_train_epoch_dbow(
                self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch,
                total_examples, total_words, work, neu1, len(self.dv),
                doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf, train_words=self.dbow_words)
        elif self.dm_concat:
            examples, tally, raw_tally = d2v_train_epoch_dm_concat(
                self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch,
                total_examples, total_words, work, neu1, len(self.dv),
                doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        else:
            examples, tally, raw_tally = d2v_train_epoch_dm(
                self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch,
                total_examples, total_words, work, neu1, len(self.dv),
                doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)

        return examples, tally, raw_tally

    def _do_train_job(self, job, alpha, inits):
        """Train model using `job` data.

        Parameters
        ----------
        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            The corpus chunk to be used for training this batch.
        alpha : float
            Learning rate to be used for training this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        work, neu1 = inits
        tally = 0
        for doc in job:
            doctag_indexes = [self.dv.get_index(tag) for tag in doc.tags if tag in self.dv]
            doctag_vectors = self.dv.vectors
            doctags_lockf = self.dv.vectors_lockf
            if self.sg:
                tally += train_document_dbow(
                    self, doc.words, doctag_indexes, alpha, work, train_words=self.dbow_words,
                    doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
            elif self.dm_concat:
                tally += train_document_dm_concat(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
            else:
                tally += train_document_dm(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
        return tally, self._raw_word_count(job)

    def train(
        self, corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None,
        epochs=None, start_alpha=None, end_alpha=None,
        word_count=0, queue_factor=2, report_delay=1.0, callbacks=(),
        **kwargs,
    ):
        """Update the model's neural weights.

        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of documents) or `total_words` (count of
        raw words in documents) **MUST** be provided. If `documents` is the same corpus
        that was provided to :meth:`~gensim.models.word2vec.Word2Vec.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.word2vec.Word2Vec.train` is only called once,
        you can set `epochs=self.iter`.

        Parameters
        ----------
        corpus_iterable : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Can be simply a list of elements, but for larger corpora,consider an iterable that streams
            the documents directly from disk/network. If you don't supply `documents` (or `corpus_file`), the model is
            left uninitialized -- use if you plan to initialize it in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (not both of them). Documents' tags are assigned automatically
            and are equal to line number, as in :class:`~gensim.models.doc2vec.TaggedLineDocument`.
        total_examples : int, optional
            Count of documents.
        total_words : int, optional
            Count of raw words in documents.
        epochs : int, optional
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to `train`.
            Use only if making multiple calls to `train`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to
            :meth:`~gensim.models.doc2vec.Doc2Vec.train`.
            Use only if making multiple calls to :meth:`~gensim.models.doc2vec.Doc2Vec.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        word_count : int, optional
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in documents.
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.

        """
        if corpus_file is None and corpus_iterable is None:
            raise TypeError("Either one of corpus_file or corpus_iterable value must be provided")

        if corpus_file is not None and corpus_iterable is not None:
            raise TypeError("Both corpus_file and corpus_iterable must not be provided at the same time")

        if corpus_iterable is None and not os.path.isfile(corpus_file):
            raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)

        if corpus_iterable is not None and not isinstance(corpus_iterable, Iterable):
            raise TypeError("corpus_iterable must be an iterable of TaggedDocument, got %r instead" % corpus_iterable)

        if corpus_file is not None:
            # Calculate offsets for each worker along with initial doctags (doctag ~ document/line number in a file)
            offsets, start_doctags = self._get_offsets_and_start_doctags_for_corpusfile(corpus_file, self.workers)
            kwargs['offsets'] = offsets
            kwargs['start_doctags'] = start_doctags

        super(Doc2Vec, self).train(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file,
            total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks, **kwargs)

    @classmethod
    def _get_offsets_and_start_doctags_for_corpusfile(cls, corpus_file, workers):
        """Get offset and initial document tag in a corpus_file for each worker.

        Firstly, approximate offsets are calculated based on number of workers and corpus_file size.
        Secondly, for each approximate offset we find the maximum offset which points to the beginning of line and
        less than approximate offset.

        Parameters
        ----------
        corpus_file : str
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
        workers : int
            Number of workers.

        Returns
        -------
        list of int, list of int
            Lists with offsets and document tags with length = number of workers.
        """
        corpus_file_size = os.path.getsize(corpus_file)
        approx_offsets = [int(corpus_file_size // workers * i) for i in range(workers)]
        offsets = []
        start_doctags = []

        with utils.open(corpus_file, mode='rb') as fin:
            curr_offset_idx = 0
            prev_filepos = 0

            for line_no, line in enumerate(fin):
                if curr_offset_idx == len(approx_offsets):
                    break

                curr_filepos = prev_filepos + len(line)
                while curr_offset_idx != len(approx_offsets) and approx_offsets[curr_offset_idx] < curr_filepos:
                    offsets.append(prev_filepos)
                    start_doctags.append(line_no)

                    curr_offset_idx += 1

                prev_filepos = curr_filepos

        return offsets, start_doctags

    def _raw_word_count(self, job):
        """Get the number of words in a given job.

        Parameters
        ----------
        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            Corpus chunk.

        Returns
        -------
        int
            Number of raw words in the corpus chunk.

        """
        return sum(len(sentence.words) for sentence in job)

    def estimated_lookup_memory(self):
        """Get estimated memory for tag lookup, 0 if using pure int tags.

        Returns
        -------
        int
            The estimated RAM required to look up a tag in bytes.

        """
        return 60 * len(self.dv) + 140 * len(self.dv)

    def infer_vector(self, doc_words, alpha=None, min_alpha=None, epochs=None):
        """Infer a vector for given post-bulk training document.

        Notes
        -----
        Subsequent calls to this function may infer different representations for the same document.
        For a more stable representation, increase the number of epochs to assert a stricter convergence.

        Parameters
        ----------
        doc_words : list of str
            A document for which the vector representation will be inferred.
        alpha : float, optional
            The initial learning rate. If unspecified, value from model initialization will be reused.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` over all inference epochs. If unspecified,
            value from model initialization will be reused.
        epochs : int, optional
            Number of times to train the new document. Larger values take more time, but may improve
            quality and run-to-run stability of inferred vectors. If unspecified, the `epochs` value
            from model initialization will be reused.

        Returns
        -------
        np.ndarray
            The inferred paragraph vector for the new document.

        """
        if isinstance(doc_words, str):  # a common mistake; fail with a nicer error
            raise TypeError("Parameter doc_words of infer_vector() must be a list of strings (not a single string).")

        alpha = alpha or self.alpha
        min_alpha = min_alpha or self.min_alpha
        epochs = epochs or self.epochs

        doctag_vectors = pseudorandom_weak_vector(self.dv.vector_size, seed_string=' '.join(doc_words))
        doctag_vectors = doctag_vectors.reshape(1, self.dv.vector_size)

        doctags_lockf = np.ones(1, dtype=REAL)
        doctag_indexes = [0]
        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

        alpha_delta = (alpha - min_alpha) / max(epochs - 1, 1)

        for i in range(epochs):
            if self.sg:
                train_document_dbow(
                    self, doc_words, doctag_indexes, alpha, work,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
            elif self.dm_concat:
                train_document_dm_concat(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
            else:
                train_document_dm(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf
                )
            alpha -= alpha_delta

        return doctag_vectors[0]

    def __getitem__(self, tag):
        """Get the vector representation of (possibly multi-term) tag.

        Parameters
        ----------
        tag : {str, int, list of str, list of int}
            The tag (or tags) to be looked up in the model.

        Returns
        -------
        np.ndarray
            The vector representations of each tag as a matrix (will be 1D if `tag` was a single tag)

        """
        if isinstance(tag, (str, int, integer,)):
            if tag not in self.wv:
                return self.dv[tag]
            return self.wv[tag]
        return vstack([self[i] for i in tag])

    def __str__(self):
        """Abbreviated name reflecting major configuration parameters.

        Returns
        -------
        str
            Human readable representation of the models internal state.

        """
        segments = []
        if self.comment:
            segments.append('"%s"' % self.comment)
        if self.sg:
            if self.dbow_words:
                segments.append('dbow+w')  # also training words
            else:
                segments.append('dbow')  # PV-DBOW (skip-gram-style)

        else:  # PV-DM...
            if self.dm_concat:
                segments.append('dm/c')  # ...with concatenative context layer
            else:
                if self.cbow_mean:
                    segments.append('dm/m')
                else:
                    segments.append('dm/s')
        segments.append('d%d' % self.dv.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.min_count > 1:
            segments.append('mc%d' % self.min_count)
        if self.sample > 0:
            segments.append('s%g' % self.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s<%s>' % (self.__class__.__name__, ','.join(segments))

    def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
        """Store the input-hidden weight matrix in the same format used by the original C word2vec-tool.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        doctag_vec : bool, optional
            Indicates whether to store document vectors.
        word_vec : bool, optional
            Indicates whether to store word vectors.
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision in case of repeated string in doctag
            and word vocab.
        fvocab : str, optional
            Optional file path used to save the vocabulary.
        binary : bool, optional
            If True, the data will be saved in binary word2vec format, otherwise - will be saved in plain text.

        """
        total_vec = None
        # save word vectors
        if word_vec:
            if doctag_vec:
                total_vec = len(self.wv) + len(self.dv)
            self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
        # save document vectors
        if doctag_vec:
            write_header = True
            append = False
            if word_vec:
                # simply appending to existing file
                write_header = False
                append = True
            self.dv.save_word2vec_format(
                fname, prefix=prefix, fvocab=fvocab, binary=binary,
                write_header=write_header, append=append,
                sort_attr='doc_count')

    @deprecated(
        "Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. "
        "init_sims() is now obsoleted and will be completely removed in future versions. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
    )
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors. Obsoleted.

        If you need a single unit-normalized vector for some key, call
        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:
        ``doc2vec_model.dv.get_vector(key, norm=True)``.

        To refresh norms after you performed some atypical out-of-band vector tampering,
        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.

        Parameters
        ----------
        replace : bool
            If True, forget the original trained vectors and only keep the normalized ones.
            You lose information if you do this.

        """
        self.dv.init_sims(replace=replace)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.doc2vec.Doc2Vec` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.
        *args : object
            Additional arguments, see `~gensim.models.word2vec.Word2Vec.load`.
        **kwargs : object
            Additional arguments, see `~gensim.models.word2vec.Word2Vec.load`.

        See Also
        --------
        :meth:`~gensim.models.doc2vec.Doc2Vec.save`
            Save :class:`~gensim.models.doc2vec.Doc2Vec` model.

        Returns
        -------
        :class:`~gensim.models.doc2vec.Doc2Vec`
            Loaded model.

        """
        try:
            return super(Doc2Vec, cls).load(*args, rethrow=True, **kwargs)
        except AttributeError as ae:
            logger.error(
                "Model load error. Was model saved using code from an older Gensim version? "
                "Try loading older model using gensim-3.8.3, then re-saving, to restore "
                "compatibility with current code.")
            raise ae

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings.

        Parameters
        ----------
        vocab_size : int, optional
            Number of raw words in the vocabulary.
        report : dict of (str, int), optional
            A dictionary from string representations of the **specific** model's memory consuming members
            to their size in bytes.

        Returns
        -------
        dict of (str, int), optional
            A dictionary from string representations of the model's memory consuming members to their size in bytes.
            Includes members from the base classes as well as weights and tag lookup memory estimation specific to the
            class.

        """
        report = report or {}
        report['doctag_lookup'] = self.estimated_lookup_memory()
        report['doctag_syn0'] = len(self.dv) * self.vector_size * dtype(REAL).itemsize
        return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)

    def build_vocab(
            self, corpus_iterable=None, corpus_file=None, update=False, progress_per=10000,
            keep_raw_vocab=False, trim_rule=None, **kwargs,
        ):
        """Build vocabulary from a sequence of documents (can be a once-only generator stream).

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Can be simply a list of :class:`~gensim.models.doc2vec.TaggedDocument` elements, but for larger corpora,
            consider an iterable that streams the documents directly from disk/network.
            See :class:`~gensim.models.doc2vec.TaggedBrownCorpus` or :class:`~gensim.models.doc2vec.TaggedLineDocument`
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (not both of them). Documents' tags are assigned automatically
            and are equal to a line number, as in :class:`~gensim.models.doc2vec.TaggedLineDocument`.
        update : bool
            If true, the new words in `documents` will be added to model's vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        **kwargs
            Additional key word arguments passed to the internal vocabulary construction.

        """
        total_words, corpus_count = self.scan_vocab(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file,
            progress_per=progress_per, trim_rule=trim_rule,
        )
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        report_values = self.prepare_vocab(update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)

        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        """Build vocabulary from a dictionary of word frequencies.

        Build model vocabulary from a passed dictionary that contains a (word -> word count) mapping.
        Words must be of type unicode strings.

        Parameters
        ----------
        word_freq : dict of (str, int)
            Word <-> count mapping.
        keep_raw_vocab : bool, optional
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        corpus_count : int, optional
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        update : bool, optional
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        """
        logger.info("processing provided word frequencies")
        # Instead of scanning text, this will assign provided word frequencies dictionary(word_freq)
        # to be directly the raw vocab.
        raw_vocab = word_freq
        logger.info(
            "collected %i different raw words, with total frequency of %i",
            len(raw_vocab), sum(raw_vocab.values()),
        )

        # Since no documents are provided, this is to control the corpus_count
        self.corpus_count = corpus_count or 0
        self.raw_vocab = raw_vocab

        # trim by min_count & precalculate downsampling
        report_values = self.prepare_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)

    def _scan_vocab(self, corpus_iterable, progress_per, trim_rule):
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        checked_string_types = 0
        vocab = defaultdict(int)
        max_rawint = -1  # highest raw int tag seen (-1 for none)
        doctags_lookup = {}
        doctags_list = []
        for document_no, document in enumerate(corpus_iterable):
            if not checked_string_types:
                if isinstance(document.words, str):
                    logger.warning(
                        "Each 'words' should be a list of words (usually unicode strings). "
                        "First 'words' here is instead plain %s.",
                        type(document.words),
                    )
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info(
                    "PROGRESS: at example #%i, processed %i words (%i words/s), %i word types, %i tags",
                    document_no, total_words, interval_rate, len(vocab), len(doctags_list)
                )
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                # Note a document tag during initial corpus scan, for structure sizing.
                if isinstance(tag, (int, integer,)):
                    max_rawint = max(max_rawint, tag)
                else:
                    if tag in doctags_lookup:
                        doctags_lookup[tag].doc_count += 1
                        doctags_lookup[tag].word_count += document_length
                    else:
                        doctags_lookup[tag] = Doctag(index=len(doctags_list), word_count=document_length, doc_count=1)
                        doctags_list.append(tag)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        corpus_count = document_no + 1
        if len(doctags_list) > corpus_count:
            logger.warning("More unique tags (%i) than documents (%i).", len(doctags_list), corpus_count)
        if max_rawint > corpus_count:
            logger.warning(
                "Highest int doctag (%i) larger than count of documents (%i). This means "
                "at least %i excess, unused slots (%i bytes) will be allocated for vectors.",
                max_rawint, corpus_count, max_rawint - corpus_count,
                (max_rawint - corpus_count) * self.vector_size * dtype(REAL).itemsize,
            )
        if max_rawint > -1:
            # adjust indexes/list to account for range of pure-int keyed doctags
            for key in doctags_list:
                doctags_lookup[key].index = doctags_lookup[key].index + max_rawint + 1
            doctags_list = list(range(0, max_rawint + 1)) + doctags_list

        self.dv.index_to_key = doctags_list
        for t, dt in doctags_lookup.items():
            self.dv.key_to_index[t] = dt.index
            self.dv.set_vecattr(t, 'word_count', dt.word_count)
            self.dv.set_vecattr(t, 'doc_count', dt.doc_count)
        self.raw_vocab = vocab
        return total_words, corpus_count

    def scan_vocab(self, corpus_iterable=None, corpus_file=None, progress_per=100000, trim_rule=None):
        """Create the model's vocabulary: a mapping from unique words in the corpus to their frequency count.

        Parameters
        ----------
        documents : iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            The tagged documents used to create the vocabulary. Their tags can be either str tokens or ints (faster).
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (not both of them).
        progress_per : int
            Progress will be logged every `progress_per` documents.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        Returns
        -------
        (int, int)
            Tuple of `(total words in the corpus, number of documents)`.

        """
        logger.info("collecting all words and their counts")
        if corpus_file is not None:
            corpus_iterable = TaggedLineDocument(corpus_file)

        total_words, corpus_count = self._scan_vocab(corpus_iterable, progress_per, trim_rule)

        logger.info(
            "collected %i word types and %i unique tags from a corpus of %i examples and %i words",
            len(self.raw_vocab), len(self.dv), corpus_count, total_words,
        )

        return total_words, corpus_count

    def similarity_unseen_docs(self, doc_words1, doc_words2, alpha=None, min_alpha=None, epochs=None):
        """Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : list of str
            Input document.
        doc_words2 : list of str
            Input document.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        epochs : int, optional
            Number of epoch to train the new document.

        Returns
        -------
        float
            The cosine similarity between `doc_words1` and `doc_words2`.

        """
        d1 = self.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
        d2 = self.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
        return np.dot(matutils.unitvec(d1), matutils.unitvec(d2))


class Doc2VecVocab(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture"""


class Doc2VecTrainables(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture"""


class TaggedBrownCorpus:
    def __init__(self, dirname):
        """Reader for the `Brown corpus (part of NLTK data) <http://www.nltk.org/book/ch02.html#tab-brown-sources>`_.

        Parameters
        ----------
        dirname : str
            Path to folder with Brown corpus.

        """
        self.dirname = dirname

    def __iter__(self):
        """Iterate through the corpus.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source`.

        """
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            with utils.open(fname, 'rb') as fin:
                for item_no, line in enumerate(fin):
                    line = utils.to_unicode(line)
                    # each file line is a single document in the Brown corpus
                    # each token is WORD/POS_TAG
                    token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                    # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                    words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                    if not words:  # don't bother sending out empty documents
                        continue
                    yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])


class TaggedLineDocument:
    def __init__(self, source):
        """Iterate over a file that contains documents:
        one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.

        Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
        automatically from the document line number (each document gets a unique integer tag).

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.doc2vec import TaggedLineDocument
            >>>
            >>> for document in TaggedLineDocument(datapath("head500.noblanks.cor")):
            ...     pass

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source` specified in the constructor.

        """
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.open(self.source, 'rb') as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
