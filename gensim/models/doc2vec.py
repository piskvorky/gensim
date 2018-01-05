#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_. See [#tutorial]_

**Make sure you have a C compiler before installing gensim, to use optimized (compiled)
doc2vec training** (70x speedup [blog]_).

Initialize a model with e.g.::

>>> model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

If you're finished training a model (=no more updates, only querying), you can do

  >>> model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True):

to trim unneeded model memory = use (much) less RAM.



.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents.
       http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
.. [blog] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

.. [#tutorial] Doc2vec in gensim tutorial,
               https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb



"""

import logging
import os

try:
    from queue import Queue
except ImportError:
    from Queue import Queue  # noqa:F401

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, array, float32 as REAL, empty, ones, \
    memmap as np_memmap, sqrt, newaxis, ndarray, dot, vstack, \
    divide as np_divide, integer


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2VecKeyedVectors, Word2VecVocab, Word2VecTrainables
from six.moves import xrange, zip
from six import string_types, integer_types
from gensim.models.base_any2vec import BaseWordEmbedddingsModel, BaseKeyedVectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from types import GeneratorType
from numpy.linalg import norm

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec

except ImportError:
    raise RuntimeError("Support for Python/Numpy implementations has been continued.")


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """
    A single document, made up of `words` (a list of unicode string tokens)
    and `tags` (a list of tokens). Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from Word2Vec.

    """

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.tags)


class Doctag(namedtuple('Doctag', 'offset, word_count, doc_count')):
    """A string document tag discovered during the initial vocabulary
    scan. (The document-vector equivalent of a Vocab object.)

    Will not be used if all presented document tags are ints.

    The offset is only the true index into the doctags_syn0/doctags_syn0_lockf
    if-and-only-if no raw-int tags were used. If any raw-int tags were used,
    string Doctag vectors begin at index (max_rawint + 1), so the true index is
    (rawint_index + 1 + offset). See also _index_to_doctag().
    """
    __slots__ = ()

    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)


class Doc2Vec(BaseWordEmbedddingsModel):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""

    def __init__(self, documents=None, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, callbacks=(), **kwargs):
        """Initialize the model from an iterable of `documents`. Each document is a
        TaggedDocument object that will be used for training.

        Parameters
        ----------
        documents : iterable of iterables
            The `documents` iterable can be simply a list of TaggedDocument elements, but for larger corpora,
            consider an iterable that streams the documents directly from disk/network.
            If you don't supply `documents`, the model is left uninitialized -- use if
            you plan to initialize it in some other way.

        dm : int {1,0}
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.

        size : int
            Dimensionality of the feature vectors.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        iter : int
            Number of iterations (epochs) over the corpus.
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        dm_mean : int {1,0}
            If 0 , use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm_concat : int {1,0}
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int
            Expected constant number of document tags per document, when using
            dm_concat mode; default is 1.
        dbow_words : int {1,0}
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        callbacks : :obj: `list` of :obj: `~gensim.callbacks.Callback`
            List of callbacks that need to be executed/run at specific stages during training.

        """

        if 'sentences' in kwargs:
            raise DeprecationWarning(
                "Parameter 'sentences' was renamed to 'documents', and will be removed in 4.0.0, "
                "use 'documents' instead."
            )

        if 'iter' in kwargs:
            kwargs['epochs'] = kwargs['iter']

        super(Doc2Vec, self).__init__(
            sg=(1 + dm) % 2,
            null_word=dm_concat,
            callbacks=callbacks,
            **kwargs)

        self.load = call_on_class_only

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count

        kwargs['null_word'] = dm_concat
        vocabulary_keys = ['max_vocab_size', 'min_count', 'sample', 'sorted_vocab', 'null_word']
        vocabulary_kwargs = dict((k, kwargs[k]) for k in vocabulary_keys if k in kwargs)
        self.vocabulary = Doc2VecVocab(**vocabulary_kwargs)

        trainables_keys = ['seed', 'hs', 'negative', 'hashfxn', 'window']
        trainables_kwargs = dict((k, kwargs[k]) for k in trainables_keys if k in kwargs)
        self.trainables = Doc2VecTrainables(
            mapfile_path=docvecs_mapfile, dm=dm, dm_concat=dm_concat, dm_tag_count=dm_tag_count,
            vector_size=self.vector_size, **trainables_kwargs)

        self.wv = Word2VecKeyedVectors()
        self.docvecs = docvecs or Doc2VecKeyedVectors()

        self.comment = comment
        if documents is not None:
            if isinstance(documents, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(
                documents, total_examples=self.corpus_count, epochs=self.epochs,
                start_alpha=self.alpha, end_alpha=self.min_alpha, callbacks=callbacks)

    @property
    def dm(self):
        """int {1,0} : `dm=1` indicates 'distributed memory' (PV-DM) else
        `distributed bag of words` (PV-DBOW) is used."""
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        """int {1,0} : `dbow=1` indicates `distributed bag of words` (PV-DBOW) else
        'distributed memory' (PV-DM) is used."""
        return self.sg  # same as SG

    def _set_train_params(self, **kwargs):
        self.trainables.hs = self.hs
        self.trainables.negative = self.negative

    def _set_keyedvectors(self):
        super(Doc2Vec, self)._set_keyedvectors()
        self.docvecs.vectors_docs = self.trainables.__dict__.get('vectors_docs', [])
        self.docvecs.doctags = self.vocabulary.__dict__.get('doctags', {})
        self.docvecs.max_rawint = self.vocabulary.__dict__.get('max_rawint', -1)
        self.docvecs.offset2doctag = self.vocabulary.__dict__.get('offset2doctag', [])
        self.docvecs.count = self.vocabulary.__dict__.get('count', 0)
        self.docvecs.mapfile_path = self.vocabulary.__dict__.get('mapfile_path', None)

    def _set_params_from_kv(self):
        super(Doc2Vec, self)._set_params_from_kv()
        self.trainables.vectors_docs = self.docvecs.__dict__.get('vectors_docs', [])
        self.vocabulary.doctags = self.docvecs.__dict__.get('doctags', {})
        self.vocabulary.max_rawint = self.docvecs.__dict__.get('max_rawint', -1)
        self.vocabulary.offset2doctag = self.docvecs.__dict__.get('offset2doctag', [])
        self.vocabulary.count = self.docvecs.__dict__.get('count', 0)
        self.vocabulary.mapfile_path = self.docvecs.__dict__.get('mapfile_path', None)

    def _clear_post_train(self):
        """Resets certain properties of the model, post training. eg. `kv.syn0norm`"""
        self.clear_sims()

    def clear_sims(self):
        self.wv.vectors_norm = None
        self.trainables.vectors_docs_norm = None

    def reset_from(self, other_model):
        """Reuse shareable structures from other_model."""
        self.vocabulary.vocab = other_model.vocabulary.vocab
        self.vocabulary.index2word = other_model.vocabulary.index2word
        self.trainables.cum_table = other_model.trainables.cum_table
        self.corpus_count = other_model.corpus_count
        self.vocabulary.count = other_model.vocabulary.count
        self.vocabulary.doctags = other_model.vocabulary.doctags
        self.vocabulary.offset2doctag = other_model.vocabulary.offset2doctag
        self.trainables.reset_weights(vocabulary=self.vocabulary)
        self._set_keyedvectors()

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            doctag_indexes = self.vocabulary.indexed_doctags(doc.tags)
            doctag_vectors = self.trainables.vectors_docs
            doctag_locks = self.trainables.vectors_docs_lockf
            if self.sg:
                tally += train_document_dbow(
                    self, doc.words, doctag_indexes, alpha, work, train_words=self.dbow_words,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            elif self.dm_concat:
                tally += train_document_dm_concat(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            else:
                tally += train_document_dm(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
        return tally, self._raw_word_count(job)

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0, callbacks=()):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        The `documents` iterable can be simply a list of TaggedDocument elements.

        To support linear learning-rate decay from (initial) alpha to min_alpha, and accurate
        progress-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) **MUST** be provided (if the corpus is the same as was provided to
        :meth:`~gensim.models.word2vec.Word2Vec.build_vocab()`, the count of examples in that corpus
        will be available in the model's :attr:`corpus_count` property).

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case,
        where :meth:`~gensim.models.word2vec.Word2Vec.train()` is only called once,
        the model's cached `iter` value should be supplied as `epochs` value.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float
            Initial learning rate.
        end_alpha : float
            Final learning rate. Drops linearly from `start_alpha`.
        word_count : int
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float
            Seconds to wait before reporting progress.
        callbacks : :obj: `list` of :obj: `~gensim.callbacks.Callback`
            List of callbacks that need to be executed/run at specific stages during training.
        """
        super(Doc2Vec, self).train(
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)
        self._set_keyedvectors()

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence.words) for sentence in job)

    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.vocabulary.offset2doctag) + 140 * len(self.vocabulary.doctags)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.

        Parameters
        ----------
        doc_words : :obj: `list` of :obj: `str`
            Document should be a list of (word) tokens.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int
            Number of times to train the new document.

        Returns
        -------
        :obj: `numpy.ndarray`
            Returns the inferred vector for the new document.

        """
        doctag_vectors, doctag_locks = self.trainables._get_doctag_trainables(doc_words)
        doctag_indexes = [0]
        work = zeros(self.trainables.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)

        for i in range(steps):
            if self.sg:
                train_document_dbow(
                    self, doc_words, doctag_indexes, alpha, work,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            elif self.dm_concat:
                train_document_dm_concat(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            else:
                train_document_dm(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha
        self._set_keyedvectors()

        return doctag_vectors[0]

    def __getitem__(self, tag):
        if tag not in self.wv.vocab:
            return self.docvecs[tag]
        return self.wv[tag]

    def __str__(self):
        """Abbreviated name reflecting major configuration paramaters."""
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
        segments.append('d%d' % self.trainables.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.vocabulary.min_count > 1:
            segments.append('mc%d' % self.vocabulary.min_count)
        if self.vocabulary.sample > 0:
            segments.append('s%g' % self.vocabulary.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))

    def delete_temporary_training_data(self, keep_doctags_vectors=True, keep_inference=True):
        """Discard parameters that are used in training and score. Use if you're sure you're done training a model.

        Parameters
        ----------
        keep_doctags_vectors : bool
            Set `keep_doctags_vectors` to False if you don't want to save doctags vectors,
            in this case you can't to use docvecs's most_similar, similarity etc. methods.
        keep_inference : bool
            Set `keep_inference` to False if you don't want to store parameters that is used for infer_vector method

        Returns
        -------
        None

        """
        if not keep_inference:
            if hasattr(self.trainables, 'syn1'):
                del self.trainables.syn1
            if hasattr(self.trainables, 'syn1neg'):
                del self.trainables.syn1neg
            if hasattr(self.trainables, 'vectors_lockf'):
                del self.trainables.vectors_lockf
        self.model_trimmed_post_training = True
        if self.docvecs and hasattr(self.trainables, 'vectors_docs') and not keep_doctags_vectors:
            del self.trainables.vectors_docs
        if self.docvecs and hasattr(self.trainables, 'vectors_docs_lockf'):
            del self.trainables.vectors_docs_lockf
        self._set_keyedvectors()

    def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        doctag_vec : bool
            Indicates whether to store document vectors.
        word_vec : bool
            Indicates whether to store word vectors.
        prefix : str
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.

        Returns
        -------
        None

        """
        total_vec = len(self.wv.vocab) + len(self.docvecs)
        write_first_line = False
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(self.wv.vocab)
            self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
        # save document vectors
        if doctag_vec:
            if not word_vec:
                total_vec = len(self.docvecs)
                write_first_line = True
            self.docvecs.save_word2vec_format(
                fname, prefix=prefix, fvocab=fvocab, total_vec=total_vec,
                binary=binary, write_first_line=write_first_line)


class Doc2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0):
        super(Doc2VecVocab, self).__init__(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word)
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.count = 0

    def scan_vocab(self, documents, progress_per=10000, trim_rule=None, **kwargs):
        logger.info("collecting all words and their counts")
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        checked_string_types = 0
        vocab = defaultdict(int)
        for document_no, document in enumerate(documents):
            if not checked_string_types:
                if isinstance(document.words, string_types):
                    logger.warning(
                        "Each 'words' should be a list of words (usually unicode strings). "
                        "First 'words' here is instead plain %s.",
                        type(document.words)
                    )
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info(
                    "PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags",
                    document_no, total_words, interval_rate, len(vocab), self.count
                )
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                self.note_doctag(tag, document_no, document_length)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        logger.info(
            "collected %i word types and %i unique tags from a corpus of %i examples and %i words",
            len(vocab), self.count, document_no + 1, total_words
        )
        corpus_count = document_no + 1
        self.raw_vocab = vocab
        return total_words, corpus_count

    def note_doctag(self, key, document_no, document_length):
        """Note a document tag during initial corpus scan, for structure sizing."""
        if isinstance(key, integer_types + (integer,)):
            self.max_rawint = max(self.max_rawint, key)
        else:
            if key in self.doctags:
                self.doctags[key] = self.doctags[key].repeat(document_length)
            else:
                self.doctags[key] = Doctag(len(self.offset2doctag), document_length, 1)
                self.offset2doctag.append(key)
        self.count = self.max_rawint + 1 + len(self.offset2doctag)

    def indexed_doctags(self, doctag_tokens):
        """Return indexes and backing-arrays used in training examples."""
        return [_int_index(index, self.doctags, self.max_rawint) for index in doctag_tokens if self._tag_seen(index)]

    def _tag_seen(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags


class Doc2VecTrainables(Word2VecTrainables):
    def __init__(self, mapfile_path=None, dm=1, dm_concat=0, dm_tag_count=1, vector_size=100, seed=1, hs=0, negative=5,
                 hashfxn=hash, window=5):
        super(Doc2VecTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hashfxn=hashfxn)
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + (2 * window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1", self.layer1_size)
        self.vectors_docs = []
        self.mapfile_path = mapfile_path

    def reset_weights(self, hs, negative, vocabulary=None):
        super(Doc2VecTrainables, self).reset_weights(hs, negative, vocabulary=vocabulary)
        self.reset_doc_weights(vocabulary=vocabulary)

    def reset_doc_weights(self, vocabulary=None):
        length = max(len(vocabulary.doctags), vocabulary.count)
        if self.mapfile_path:
            self.vectors_docs = np_memmap(
                self.mapfile_path + '.vectors_docs', dtype=REAL, mode='w+', shape=(length, self.vector_size)
            )
            self.vectors_docs_lockf = np_memmap(
                self.mapfile_path + '.vectors_docs_lockf', dtype=REAL, mode='w+', shape=(length,)
            )
            self.vectors_docs_lockf.fill(1.0)
        else:
            self.vectors_docs = empty((length, self.vector_size), dtype=REAL)
            self.vectors_docs_lockf = ones((length,), dtype=REAL)  # zeros suppress learning

        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (self.seed, _index_to_doctag(i, vocabulary.offset2doctag, vocabulary.max_rawint))
            self.vectors_docs[i] = self.seeded_vector(seed)

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'vectors_docs_norm'])
        super(Word2VecTrainables, self).save(*args, **kwargs)

    def _get_doctag_trainables(self, doc_words):
        doctag_vectors = zeros((1, self.vector_size), dtype=REAL)
        doctag_vectors[0] = self.seeded_vector(' '.join(doc_words))
        doctag_locks = ones(1, dtype=REAL)
        return doctag_vectors, doctag_locks


class Doc2VecKeyedVectors(BaseKeyedVectors):
    def __init__(self):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.vectors_docs = []
        self.mapfile_path = None

    @property
    def index2entity(self):
        return self.offset2doctag

    @index2entity.setter
    def index2entity(self, value):
        self.offset2doctag = value

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if index in self:
            if isinstance(index, string_types + integer_types + (integer,)):
                return self.vectors_docs[_int_index(index, self.doctags, self.max_rawint)]
            return vstack([self[i] for i in index])
        raise KeyError("tag '%s' not seen in training corpus/invalid" % index)

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags

    def __len__(self):
        return self.count

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.doc2vec.Doc2VecKeyedVectors.load` which supports
        operations on trained document vectors like `most_similar`.

        Parameters
        ----------
        fname : str
            Path to the file.

        Returns
        -------
        None

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_docs_norm'])
        super(Doc2VecKeyedVectors, self).save(*args, **kwargs)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training or inference** after doing a replace.
        The model becomes effectively read-only = you can call `most_similar`, `similarity`
        etc., but not `train` or `infer_vector`.

        """
        if getattr(self, 'vectors_docs_norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.vectors_docs.shape[0]):
                    self.vectors_docs[i, :] /= sqrt((self.vectors_docs[i, :] ** 2).sum(-1))
                self.vectors_docs_norm = self.vectors_docs
            else:
                if self.mapfile_path:
                    self.vectors_docs_norm = np_memmap(
                        self.mapfile_path + '.vectors_docs_norm', dtype=REAL,
                        mode='w+', shape=self.vectors_docs.shape)
                else:
                    self.vectors_docs_norm = empty(self.vectors_docs.shape, dtype=REAL)
                np_divide(
                    self.vectors_docs, sqrt((self.vectors_docs ** 2).sum(-1))[..., newaxis], self.vectors_docs_norm)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying `vectors_docs_norm` vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)

        Parameters
        ----------
        positive : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute positively.
        negative : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute negatively.
        topn : int
            Number of top-N similar docvecs to return.
        clip_start : int
            Start clipping index.
        clip_end : int
            End clipping index.

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (doc, similarity)

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()
        clip_end = clip_end or len(self.vectors_docs_norm)

        if isinstance(positive, string_types + integer_types + (integer,)) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.vectors_docs_norm[_int_index(doc, self.doctags, self.max_rawint)])
                all_docs.add(_int_index(doc, self.doctags, self.max_rawint))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors_docs_norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [
            (_index_to_doctag(sim + clip_start, self.offset2doctag, self.max_rawint), float(dists[sim]))
            for sim in best
            if (sim + clip_start) not in all_docs
        ]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        docs : :obj: `list` of (str or int)
            List of seen documents specified by their corresponding string tags or integer indices.

        Returns
        -------
        str or int
            The document further away from the mean of all the documents.

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s", docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(self.vectors_docs_norm[_int_index(doc, self.doctags, self.max_rawint)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        d1 : int or str
            Indicate the first document by it's string tag or integer index.
        d2 : int or str
            Indicate the second document by it's string tag or integer index.

        Returns
        -------
        float
            The cosine similarity between the vectors of the two documents.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        ds1 : :obj: `list` of (str or int)
            Specify the first set of documents as a list of their integer indices or string tags.
        ds2 : :obj: `list` of (str or int)
            Specify the second set of documents as a list of their integer indices or string tags.

        Returns
        -------
        float
            The cosine similarity between the means of the documents in each of the two sets.

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    # required by base keyed vectors class
    def distances(self, d1, other_docs=()):
        """Compute distances from given document (string tag or int index) to all documents in `other_docs`.
        If `other_docs` is empty, return distance between `d1` and all documents seen during training.
        """
        input_vector = self[d1]
        if not other_docs:
            other_vectors = self.vectors_docs
        else:
            other_vectors = self[other_vectors]
        return 1 - WordEmbeddingsKeyedVectors.cosine_similarities(input_vector, other_vectors)

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :obj: `~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : :obj: `list` of :obj: `str`
            The first document. Document should be a list of (word) tokens.
        doc_words2 : :obj: `list` of :obj: `str`
            The second document. Document should be a list of (word) tokens.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int
            Number of times to train the new document.

        Returns
        -------
        float
            The cosine similarity between the unseen documents.

        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))

    def save_word2vec_format(self, fname, prefix='*dt_', fvocab=None,
                             total_vec=None, binary=False, write_first_line=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        prefix : str
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)
        write_first_line : bool
            Whether to print the first line in the file. Useful when saving doc-vectors after word-vectors.

        Returns
        -------
        None

        """
        total_vec = total_vec or len(self)
        with utils.smart_open(fname, 'ab') as fout:
            if write_first_line:
                logger.info("storing %sx%s projection weights into %s", total_vec, self.vectors_docs.shape[1], fname)
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, self.vectors_docs.shape[1])))
            # store as in input order
            for i in range(len(self)):
                doctag = u"%s%s" % (prefix, _index_to_doctag(i, self.offset2doctag, self.max_rawint))
                row = self.vectors_docs[i]
                if binary:
                    fout.write(utils.to_utf8(doctag) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (doctag, ' '.join("%f" % val for val in row))))


class TaggedBrownCorpus(object):
    """Iterate over documents from the Brown corpus (part of NLTK data), yielding
    each document out as a TaggedDocument object."""

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for item_no, line in enumerate(utils.smart_open(fname)):
                line = utils.to_unicode(line)
                # each file line is a single document in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty documents
                    continue
                yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])


class TaggedLineDocument(object):
    """Simple format: one document = one line = one TaggedDocument object.

    Words are expected to be already preprocessed and separated by whitespace,
    tags are constructed automatically from the document line number."""

    def __init__(self, source):
        """
        `source` can be either a string (filename) or a file object.

        Example::

            documents = TaggedLineDocument('myfile.txt')

        Or for compressed files::

            documents = TaggedLineDocument('compressed_text.txt.bz2')
            documents = TaggedLineDocument('compressed_text.txt.gz')

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])


def _int_index(index, doctags, max_rawint):
    """Return int index for either string or int index"""
    if isinstance(index, integer_types + (integer,)):
        return index
    else:
        return max_rawint + 1 + doctags[index].offset


def _index_to_doctag(i_index, offset2doctag, max_rawint):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - max_rawint - 1
        if 0 <= candidate_offset < len(offset2doctag):
            return offset2doctag[candidate_offset]
        else:
            return i_index
