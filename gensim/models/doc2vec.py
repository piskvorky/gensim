#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_. See [tutorial]_

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



.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [blog] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

.. [tutorial] Doc2vec in gensim tutorial, https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb



"""

import logging
import os
import warnings

try:
    from queue import Queue
except ImportError:
    from Queue import Queue  # noqa:F401

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide, integer


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, train_cbow_pair, train_sg_pair, train_batch_sg
from gensim.models.keyedvectors import KeyedVectors
from six.moves import xrange, zip
from six import string_types, integer_types

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
    logger.debug('Fast version of %s is being used', __name__)
except ImportError:
    logger.warning('Slow version of %s is being used', __name__)
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed bag of words model ("PV-DBOW") by training on a single document.

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        If `train_words` is True, simultaneously train word-to-word (not just doc-to-word)
        examples, exactly as per Word2Vec skip-gram training. (Without this option,
        word vectors are neither consulted nor updated during DBOW doc vector training.)

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)

        return len(doc_words)

    def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                          learn_doctags=True, learn_words=True, learn_hidden=True,
                          word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document.

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`. This
        method implements the DM model with a projection (input) layer that is
        either the sum or mean of the context vectors, depending on the model's
        `dm_mean` configuration field.  See `train_document_dm_concat()` for the DM
        model with a concatenated input layer.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]

        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
            l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
            count = len(word2_indexes) + len(doctag_indexes)
            if model.cbow_mean and count > 1:
                l1 /= count
            neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                    learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]

        return len(word_vocabs)

    def train_document_dm_concat(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                                 learn_doctags=True, learn_words=True, learn_hidden=True,
                                 word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document, using a
        concatenation of the context window word vectors (rather than a sum or average).

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        doctag_len = len(doctag_indexes)
        if doctag_len != model.dm_tag_count:
            return 0  # skip doc without expected number of doctag(s) (TODO: warn/pad?)

        null_word = model.wv.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            word_context_indexes = (
                padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
            )
            predict_word = model.wv.vocab[model.wv.index2word[padded_document_indexes[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha,
                                    learn_hidden=learn_hidden, learn_vectors=False)

            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doctag_locks[doctag_indexes], word_locks[word_context_indexes]))
            neu1e_r = (neu1e.reshape(-1, model.vector_size)
                       * np_repeat(e_locks, model.vector_size).reshape(-1, model.vector_size))

            if learn_doctags:
                np_add.at(doctag_vectors, doctag_indexes, neu1e_r[:doctag_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indexes, neu1e_r[doctag_len:])

        return len(padded_document_indexes) - pre_pad_count - post_pad_count


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


# for compatibility
class LabeledSentence(TaggedDocument):
    def __init__(self, *args, **kwargs):
        warnings.warn('LabeledSentence has been replaced by TaggedDocument', DeprecationWarning)


class DocvecsArray(utils.SaveLoad):
    """
    Default storage of doc vectors during/after training, in a numpy array.

    As the 'docvecs' property of a Doc2Vec model, allows access and
    comparison of document vectors.

    >>> docvec = d2v_model.docvecs[99]
    >>> docvec = d2v_model.docvecs['SENT_99']  # if string tag used in training
    >>> sims = d2v_model.docvecs.most_similar(99)
    >>> sims = d2v_model.docvecs.most_similar('SENT_99')
    >>> sims = d2v_model.docvecs.most_similar(docvec)

    If only plain int tags are presented during training, the dict (of
    string tag -> index) and list (of index -> string tag) stay empty,
    saving memory.

    Supplying a mapfile_path (as by initializing a Doc2Vec model with a
    'docvecs_mapfile' value) will use a pair of memory-mapped
    files as the array backing for doctag_syn0/doctag_syn0_lockf values.

    The Doc2Vec model automatically uses this class, but a future alternative
    implementation, based on another persistence mechanism like LMDB, LevelDB,
    or SQLite, should also be possible.
    """

    def __init__(self, mapfile_path=None):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.mapfile_path = mapfile_path

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
        return ([self._int_index(index) for index in doctag_tokens if index in self],
                self.doctag_syn0, self.doctag_syn0_lockf, doctag_tokens)

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

    def _key_index(self, i_index, missing=None):
        """Return string index for given int index, if available"""
        warnings.warn("use DocvecsArray.index_to_doctag", DeprecationWarning)
        return self.index_to_doctag(i_index)

    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(index, string_types + integer_types + (integer,)):
            return self.doctag_syn0[self._int_index(index)]

        return vstack([self[i] for i in index])

    def __len__(self):
        return self.count

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])
        super(DocvecsArray, self).save(*args, **kwargs)

    def borrow_from(self, other_docvecs):
        self.count = other_docvecs.count
        self.doctags = other_docvecs.doctags
        self.offset2doctag = other_docvecs.offset2doctag

    def clear_sims(self):
        self.doctag_syn0norm = None

    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.offset2doctag) + 140 * len(self.doctags)

    def reset_weights(self, model):
        length = max(len(self.doctags), self.count)
        if self.mapfile_path:
            self.doctag_syn0 = np_memmap(self.mapfile_path + '.doctag_syn0', dtype=REAL,
                                         mode='w+', shape=(length, model.vector_size))
            self.doctag_syn0_lockf = np_memmap(self.mapfile_path + '.doctag_syn0_lockf', dtype=REAL,
                                               mode='w+', shape=(length,))
            self.doctag_syn0_lockf.fill(1.0)
        else:
            self.doctag_syn0 = empty((length, model.vector_size), dtype=REAL)
            self.doctag_syn0_lockf = ones((length,), dtype=REAL)  # zeros suppress learning

        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (model.seed, self.index_to_doctag(i))
            self.doctag_syn0[i] = model.seeded_vector(seed)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training or inference** after doing a replace.
        The model becomes effectively read-only = you can call `most_similar`, `similarity`
        etc., but not `train` or `infer_vector`.

        """
        if getattr(self, 'doctag_syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.doctag_syn0.shape[0]):
                    self.doctag_syn0[i, :] /= sqrt((self.doctag_syn0[i, :] ** 2).sum(-1))
                self.doctag_syn0norm = self.doctag_syn0
            else:
                if self.mapfile_path:
                    self.doctag_syn0norm = np_memmap(
                        self.mapfile_path + '.doctag_syn0norm', dtype=REAL,
                        mode='w+', shape=self.doctag_syn0.shape)
                else:
                    self.doctag_syn0norm = empty(self.doctag_syn0.shape, dtype=REAL)
                np_divide(self.doctag_syn0, sqrt((self.doctag_syn0 ** 2).sum(-1))[..., newaxis], self.doctag_syn0norm)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying doctag_syn0norm vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)
        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()
        clip_end = clip_end or len(self.doctag_syn0norm)

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
                mean.append(weight * self.doctag_syn0norm[self._int_index(doc)])
                all_docs.add(self._int_index(doc))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.doctag_syn0norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [(self.index_to_doctag(sim + clip_start), float(dists[sim])) for sim in best if (sim + clip_start) not in all_docs]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s", docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(self.doctag_syn0norm[self._int_index(doc)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Compute cosine similarity between two post-bulk out of training documents.

        Document should be a list of (word) tokens.
        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))


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


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""

    def __init__(self, documents=None, dm_mean=None,
                 dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
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
                default is 1e-3, values of 1e-5 (or lower) may also be useful, value 0. disable downsampling.

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

        if 'sentences' in kwargs:
            raise DeprecationWarning("'sentences' in doc2vec was renamed to 'documents'. Please use documents parameter.")

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
        if self.dm and self.dm_concat:
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size

        self.docvecs = docvecs or DocvecsArray(docvecs_mapfile)
        self.comment = comment
        if documents is not None:
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(documents, total_examples=self.corpus_count, epochs=self.iter)

    @property
    def dm(self):
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        return self.sg  # same as SG

    def clear_sims(self):
        super(Doc2Vec, self).clear_sims()
        self.docvecs.clear_sims()

    def reset_weights(self):
        if self.dm and self.dm_concat:
            # expand l1 size to match concatenated tags+words length
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1", self.layer1_size)
        super(Doc2Vec, self).reset_weights()
        self.docvecs.reset_weights(self)

    def reset_from(self, other_model):
        """Reuse shareable structures from other_model."""
        self.docvecs.borrow_from(other_model.docvecs)
        super(Doc2Vec, self).reset_from(other_model)

    def scan_vocab(self, documents, progress_per=10000, trim_rule=None, update=False):
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
                    logger.warning("Each 'words' should be a list of words (usually unicode strings). First 'words' here is instead plain %s.", type(document.words))
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info("PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags", document_no, total_words, interval_rate, len(vocab), len(self.docvecs))
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                self.docvecs.note_doctag(tag, document_no, document_length)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        logger.info("collected %i word types and %i unique tags from a corpus of %i examples and %i words", len(vocab), len(self.docvecs), document_no + 1, total_words)
        self.corpus_count = document_no + 1
        self.raw_vocab = vocab

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1,
                                           doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            self.docvecs.trained_item(indexed_doctags)
        return tally, self._raw_word_count(job)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence.words) for sentence in job)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.

        Document should be a list of (word) tokens.
        """
        doctag_vectors = empty((1, self.vector_size), dtype=REAL)
        doctag_vectors[0] = self.seeded_vector(' '.join(doc_words))
        doctag_locks = ones(1, dtype=REAL)
        doctag_indexes = [0]

        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

        for i in range(steps):
            if self.sg:
                train_document_dbow(self, doc_words, doctag_indexes, alpha, work,
                                    learn_words=False, learn_hidden=False,
                                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                train_document_dm_concat(self, doc_words, doctag_indexes, alpha, work, neu1,
                                         learn_words=False, learn_hidden=False,
                                         doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                train_document_dm(self, doc_words, doctag_indexes, alpha, work, neu1,
                                  learn_words=False, learn_hidden=False,
                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

        return doctag_vectors[0]

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings."""
        report = report or {}
        report['doctag_lookup'] = self.docvecs.estimated_lookup_memory()
        report['doctag_syn0'] = self.docvecs.count * self.vector_size * dtype(REAL).itemsize
        return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)

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
        segments.append('d%d' % self.vector_size)  # dimensions
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
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))

    def delete_temporary_training_data(self, keep_doctags_vectors=True, keep_inference=True):
        """
        Discard parameters that are used in training and score. Use if you're sure you're done training a model.
        Set `keep_doctags_vectors` to False if you don't want to save doctags vectors,
        in this case you can't to use docvecs's most_similar, similarity etc. methods.
        Set `keep_inference` to False if you don't want to store parameters that is used for infer_vector method
        """
        if not keep_inference:
            self._minimize_model(False, False, False)
        if self.docvecs and hasattr(self.docvecs, 'doctag_syn0') and not keep_doctags_vectors:
            del self.docvecs.doctag_syn0
        if self.docvecs and hasattr(self.docvecs, 'doctag_syn0_lockf'):
            del self.docvecs.doctag_syn0_lockf

    def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix.

         `fname` is the file used to save the vectors in
         `doctag_vec` is an optional boolean indicating whether to store document vectors
         `word_vec` is an optional boolean indicating whether to store word vectors
         (if both doctag_vec and word_vec are True, then both vectors are stored in the same file)
         `prefix` to uniquely identify doctags from word vocab, and avoid collision
         in case of repeated string in doctag and word vocab
         `fvocab` is an optional file used to save the vocabulary
         `binary` is an optional boolean indicating whether the data is to be saved
         in binary word2vec format (default: False)

        """
        total_vec = len(self.wv.vocab) + len(self.docvecs)
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(self.wv.vocab)
            KeyedVectors.save_word2vec_format(self.wv, fname, fvocab, binary, total_vec)
        # save document vectors
        if doctag_vec:
            with utils.smart_open(fname, 'ab') as fout:
                if not word_vec:
                    total_vec = len(self.docvecs)
                    logger.info("storing %sx%s projection weights into %s", total_vec, self.vector_size, fname)
                    fout.write(utils.to_utf8("%s %s\n" % (total_vec, self.vector_size)))
                # store as in input order
                for i in range(len(self.docvecs)):
                    doctag = prefix + str(self.docvecs.index_to_doctag(i))
                    row = self.docvecs.doctag_syn0[i]
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
