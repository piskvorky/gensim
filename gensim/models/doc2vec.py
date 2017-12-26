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

from numpy import zeros, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide, integer


from gensim.utils import call_on_class_only, deprecated
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2VecKeyedVectors, Word2VecVocab, Word2VecTrainables
from gensim.models.keyedvectors import KeyedVectors
from six.moves import xrange, zip
from six import string_types, integer_types
from gensim.models.base_any2vec import BaseWordEmbedddingsModel
from types import GeneratorType

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
    logger.info("Using FAST_VERSION - %s", FAST_VERSION)
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
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
    (rawint_index + 1 + offset). See also DocvecsArray.index_to_doctag().
    """
    __slots__ = ()

    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)


class Doc2Vec(BaseWordEmbedddingsModel):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""

    def __init__(self, documents=None, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, **kwargs):
        """
        Initialize the model from an iterable of `documents`. Each document is a
        TaggedDocument object that will be used for training.

        The `documents` iterable can be simply a list of TaggedDocument elements, but for larger corpora,
        consider an iterable that streams the documents directly from disk/network.

        If you don't supply `documents`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `dm` defines the training algorithm. By default (`dm=1`), 'distributed memory' (PV-DM) is used.
        Otherwise, `distributed bag of words` (PV-DBOW) is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the predicted word and context words used for prediction
        within a document.

        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

        `seed` = for the random number generator.
        Note that for a fully deterministically-reproducible run, you must also limit the model to
        a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
        3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
        environment variable to control hash randomization.)

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 1e-3, values of 1e-5 (or lower) may also be useful, set to 0.0 to disable downsampling.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `iter` = number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5,
        but values of 10 or 20 are common in published 'Paragraph Vector' experiments.

        `hs` = if 1, hierarchical softmax will be used for model training.
        If set to 0 (default), and `negative` is non-zero, negative sampling will be used.

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.

        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when dm is used in non-concatenative mode.

        `dm_concat` = if 1, use concatenation of context vectors rather than sum/average;
        default is 0 (off). Note concatenation results in a much-larger model, as the input
        is no longer the size of one (sampled or arithmetically combined) word vector, but the
        size of the tag(s) and all words in the context strung together.

        `dm_tag_count` = expected constant number of document tags per document, when using
        dm_concat mode; default is 1.

        `dbow_words` if set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
        doc-vector training; default is 0 (faster training of doc-vectors only).

        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
        returns either util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT.
        Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
        of the model.
        """
        # from IPython.core.debugger import set_trace
        # set_trace()

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
            **kwargs)

        self.load = call_on_class_only

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count

        vocabulary_keys = ['max_vocab_size', 'min_count', 'sample', 'sorted_vocab', 'null_word']
        vocabulary_kwargs = dict((k, kwargs[k]) for k in vocabulary_keys if k in kwargs)
        self.vocabulary = Doc2VecVocab(**vocabulary_kwargs)

        trainables_keys = ['seed', 'hs', 'negative', 'hashfxn', 'window']
        trainables_kwargs = dict((k, kwargs[k]) for k in trainables_keys if k in kwargs)
        self.trainables = Doc2VecTrainables(
            mapfile_path=docvecs_mapfile, dm=dm, dm_concat=dm_concat, dm_tag_count=dm_tag_count,
            vector_size=self.vector_size, **trainables_kwargs)

        self.wv = Word2VecKeyedVectors()
        # self.docvecs = docvecs or DocvecsArray(docvecs_mapfile)

        self.comment = comment
        if documents is not None:
            if isinstance(documents, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(
                documents, total_examples=self.vocabulary.corpus_count, epochs=self.epochs,
                start_alpha=self.alpha, end_alpha=self.min_alpha)

    @property
    def dm(self):
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        return self.sg  # same as SG

    def _clear_post_train(self):
        """Resets certain properties of the model, post training. eg. `kv.syn0norm`"""
        self.wv.vectors_norm = None
        self.trainables.vectors_docs_norm = None

    def _set_train_params(self, **kwargs):
        self.trainables.hs = self.hs
        self.trainables.negative = self.negative

    # def clear_sims(self):
    #     super(Doc2Vec, self).clear_sims()
    #     self.docvecs.clear_sims()

    # def reset_from(self, other_model):
    #     """Reuse shareable structures from other_model."""
    #     self.docvecs.borrow_from(other_model.docvecs)
    #     super(Doc2Vec, self).reset_from(other_model)

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
            self.vocabulary.trained_item(doctag_indexes)
        return tally, self._raw_word_count(job)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence.words) for sentence in job)


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
        self.corpus_count = document_no + 1
        self.raw_vocab = vocab
        return total_words

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

    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    def indexed_doctags(self, doctag_tokens):
        """Return indexes and backing-arrays used in training examples."""
        return [self._int_index(index) for index in doctag_tokens if self._tag_seen(index)]

    def trained_item(self, indexed_tuple):
        """Persist any changes made to the given indexes (matching tuple previously
        returned by indexed_doctags()); a no-op for this implementation"""
        pass

    def _int_index(self, index):
        """Return int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return self.max_rawint + 1 + self.doctags[index].offset

    def _tag_seen(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags


class Doc2VecTrainables(Word2VecTrainables):
    def __init__(self, mapfile_path=None, dm=1, dm_concat=0, dm_tag_count=1, vector_size=100, seed=1, hs=0, negative=5,
                 hashfxn=hash, window=5):
        super(Doc2VecTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hs=hs, negative=negative, hashfxn=hashfxn)
        self.dm = dm
        self.dm_concat = dm_concat
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + (2 * window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1", self.layer1_size)
        self.vectors_docs = []
        self.mapfile_path = mapfile_path

    def reset_weights(self, vocabulary=None):
        super(Doc2VecTrainables, self).reset_weights(vocabulary=vocabulary)
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
            seed = "%d %s" % (self.seed, vocabulary.index_to_doctag(i))
            self.vectors_docs[i] = self.seeded_vector(seed)


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
