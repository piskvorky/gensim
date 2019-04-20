#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""Optimized Cython functions for training a :class:`~gensim.models.fasttext.FastText` model.

The main entry points are :func:`~gensim.models.fasttext_inner.train_batch_sg`
and :func:`~gensim.models.fasttext_inner.train_batch_cbow`.  They may be
called directly from Python code.

Notes
-----
The implementation of the above functions heavily depends on the
FastTextConfig struct defined in :file:`gensim/models/fasttext_inner.pxd`.

The FAST_VERSION constant determines what flavor of BLAS we're currently using:

    0: double
    1: float
    2: no BLAS, use Cython loops instead

See Also
--------
`Basic Linear Algebra Subprograms <http://www.netlib.org/blas/>`_

"""

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

#
# We make use of the following BLAS functions (or their analogs, if BLAS is
# unavailable):
#
# scopy(dimensionality, x, inc_x, y, inc_y):
#   Performs y = x
#
# sscal: y *= alpha
#
# saxpy(dimensionality, alpha, x, inc_x, y, inc_y):
#   Calculates y = y + alpha * x (Single precision A*X Plus Y).
#
# sdot: dot product
#
# The increments (inc_x and inc_y) are usually 1 in our case.
#

#
# FIXME: why are we importing EXP_TABLE and then redefining it?
#
from word2vec_inner cimport bisect_left, random_int32, scopy, saxpy, dsdot, sscal, \
     REAL_t, EXP_TABLE, our_dot, our_saxpy, our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef void fasttext_fast_sentence_sg_neg(FastTextConfig *c, int i, int j) nogil:
    """Perform skipgram training with negative sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of the word at the center of the current window.  This is
        referred to as word2 in some parts of the implementation.
    j : int
        The index of another word inside the window.  This is referred to as
        word in some parts of the implementation.

    Notes
    -----
    Modifies c.next_random as a side-effect.

    """

    #
    # Unpack the struct, extracting only the required parts into separate
    # variables.  This is here for historical reasons.  We could bypass these
    # declarations and use parts of the struct directly, but that would be
    # somewhat more verbose.
    #
    cdef:
        int negative = c.negative
        np.uint32_t *cum_table = c.cum_table
        unsigned long long cum_table_len = c.cum_table_len
        REAL_t *syn0_vocab = c.syn0_vocab
        REAL_t *syn0_ngrams = c.syn0_ngrams
        REAL_t *syn1neg = c.syn1neg
        int size = c.size
        np.uint32_t word_index = c.indexes[j]
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]
        REAL_t alpha = c.alpha
        REAL_t *work = c.work
        REAL_t *l1 = c.neu1
        unsigned long long next_random = c.next_random
        REAL_t *word_locks_vocab = c.word_locks_vocab
        REAL_t *word_locks_ngrams = c.word_locks_ngrams

    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_index[d] * size], &ONE, l1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&size, &norm_factor, l1 , &ONE)

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, l1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1neg[row2], &ONE)
    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)

    c.next_random = next_random


cdef void fasttext_fast_sentence_sg_hs(FastTextConfig *c, int i, int j) nogil:
    """Perform skipgram training with hierarchical sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of the word at the center of the current window.  This is
        referred to as word2 in some parts of the implementation.
    j : int
        The index of another word inside the window.  This is referred to as
        word in some parts of the implementation.

    """
    cdef:
        np.uint32_t *word_point = c.points[j]
        np.uint8_t *word_code = c.codes[j]
        int codelen = c.codelens[j]
        REAL_t *syn0_vocab = c.syn0_vocab
        REAL_t *syn0_ngrams = c.syn0_ngrams
        REAL_t *syn1 = c.syn1
        int size = c.size
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]
        REAL_t alpha = c.alpha
        REAL_t *work = c.work
        REAL_t *l1 = c.neu1
        REAL_t *word_locks_vocab = c.word_locks_vocab
        REAL_t *word_locks_ngrams = c.word_locks_ngrams

    #
    # b : long long
    #   iteration variable
    # row1 : long long
    #   Offset for word2 (!!) into the syn0_vocab array
    # row2 : long long
    #   Another offset into the syn0_vocab array
    # f : REAL_t
    #   ?
    # f_dot : REAL_t
    #   Dot product result
    # g : REAL_t
    #   ?
    #
    cdef long long b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g, f_dot

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            row2 = subwords_index[d] * size
            our_saxpy(&size, &ONEF, &syn0_ngrams[row2], &ONE, l1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&size, &norm_factor, l1 , &ONE)

    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, l1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        row2 = subwords_index[d] * size
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[row2], &ONE)


cdef void fasttext_fast_sentence_cbow_neg(FastTextConfig *c, int i, int j, int k) nogil:
    """Perform CBOW training with negative sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of a word inside the current window.
    j : int
        The start of the current window.
    k : int
        The end of the current window.  Essentially, j <= i < k.

    Notes
    -----
    Modifies c.next_random as a side-effect.

    """

    cdef:
        int negative = c.negative
        np.uint32_t *cum_table = c.cum_table
        unsigned long long cum_table_len = c.cum_table_len
        # int *codelens = c.codelens
        REAL_t *neu1 = c.neu1
        REAL_t *syn0_vocab = c.syn0_vocab
        REAL_t *syn0_ngrams = c.syn0_ngrams
        REAL_t *syn1neg = c.syn1neg
        int size = c.size
        np.uint32_t *indexes = c.indexes
        np.uint32_t **subwords_idx = c.subwords_idx
        int *subwords_idx_len = c.subwords_idx_len
        REAL_t alpha = c.alpha
        REAL_t *work = c.work
        int cbow_mean = c.cbow_mean
        unsigned long long next_random = c.next_random
        REAL_t *word_locks_vocab = c.word_locks_vocab
        REAL_t *word_locks_ngrams = c.word_locks_ngrams

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)

    c.next_random = next_random


cdef void fasttext_fast_sentence_cbow_hs(FastTextConfig *c, int i, int j, int k) nogil:
    """Perform CBOW training with hierarchical sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of a word inside the current window.
    j : int
        The start of the current window.
    k : int
        The end of the current window.  Essentially, j <= i < k.

    """

    cdef:
        np.uint32_t *word_point = c.points[i]
        np.uint8_t *word_code = c.codes[i]
        int *codelens = c.codelens
        REAL_t *neu1 = c.neu1
        REAL_t *syn0_vocab = c.syn0_vocab
        REAL_t *syn0_ngrams = c.syn0_ngrams
        REAL_t *syn1 = c.syn1
        int size = c.size
        np.uint32_t *indexes = c.indexes
        np.uint32_t **subwords_idx = c.subwords_idx
        int *subwords_idx_len = c.subwords_idx_len
        REAL_t alpha = c.alpha
        REAL_t *work = c.work
        int cbow_mean = c.cbow_mean
        REAL_t *word_locks_vocab = c.word_locks_vocab
        REAL_t *word_locks_ngrams = c.word_locks_ngrams

    cdef long long b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)


cdef void init_ft_config(FastTextConfig *c, model, alpha, _work, _neu1):
    """Load model parameters into a FastTextConfig struct.

    The struct itself is defined and documented in fasttext_inner.pxd.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to the struct to initialize.
    model : gensim.models.fasttext.FastText
        The model to load.
    alpha : float
        The initial learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.

    """
    c.hs = model.hs
    c.negative = model.negative
    c.sample = (model.vocabulary.sample != 0)
    c.cbow_mean = model.cbow_mean
    c.window = model.window
    c.workers = model.workers

    c.syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    c.word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_vocab_lockf))
    c.syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))
    c.word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_ngrams_lockf))

    c.alpha = alpha
    c.size = model.wv.vector_size

    if c.hs:
        c.syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if c.negative:
        c.syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        c.cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        c.cum_table_len = len(model.vocabulary.cum_table)
    if c.negative or c.sample:
        c.next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c.work = <REAL_t *>np.PyArray_DATA(_work)
    c.neu1 = <REAL_t *>np.PyArray_DATA(_neu1)


cdef object populate_ft_config(FastTextConfig *c, vocab, buckets_word, sentences):
    """Prepare C structures so we can go "full C" and release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token and store the result up front to save time: we'll be seeing each
    token multiple times because of windowing, so better to do the work once
    here.

    Parameters
    ----------
    c : FastTextConfig*
        A pointer to the struct that will contain the populated indices.
    vocab : dict
        The vocabulary
    buckets_word : dict
        A map containing the buckets each word appears in
    sentences : iterable
        The sentences to read

    Returns
    -------
    effective_words : int
        The number of in-vocabulary tokens.
    effective_sentences : int
        The number of non-empty sentences.

    Notes
    -----
    If sampling is used, each vocab term must have the .sample_int attribute
    initialized.

    See Also
    --------
    :meth:`gensim.models.word2vec.Word2VecVocab.create_binary_tree`

    """
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vocab[token] if token in vocab else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and word.sample_int < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word.index

            c.subwords_idx_len[effective_words] = <int>(len(buckets_word[word.index]))
            c.subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(buckets_word[word.index])

            if c.hs:
                c.codelens[effective_words] = <int>len(word.code)
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

    return effective_words, effective_sentences


cdef void fasttext_train_any(FastTextConfig *c, int num_sentences, int sg) nogil:
    """Performs training on a fully initialized and populated configuration.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to the configuration struct.
    num_sentences : int
        The number of sentences to train.
    sg : int
        1 for skipgram, 0 for CBOW.

    """
    cdef:
        int sent_idx
        int sentence_start
        int sentence_end
        int i
        int window_start
        int window_end
        int j

    for sent_idx in range(num_sentences):
        sentence_start = c.sentence_idx[sent_idx]
        sentence_end = c.sentence_idx[sent_idx + 1]
        for i in range(sentence_start, sentence_end):
            #
            # Determine window boundaries, making sure we don't leak into
            # adjacent sentences.
            #
            window_start = i - c.window + c.reduced_windows[i]
            if window_start < sentence_start:
                window_start = sentence_start
            window_end = i + c.window + 1 - c.reduced_windows[i]
            if window_end > sentence_end:
                window_end = sentence_end

            #
            # TODO: why can't I use min/max here?  I get a segfault.
            #
            # window_start = max(sentence_start, i - c.window + c.reduced_windows[i])
            # window_end = min(sentence_end, i + c.window + 1 - c.reduced_windows[i])
            #
            if sg == 0:
                if c.hs:
                    fasttext_fast_sentence_cbow_hs(c, i, window_start, window_end)
                if c.negative:
                    fasttext_fast_sentence_cbow_neg(c, i, window_start, window_end)
            else:
                for j in range(window_start, window_end):
                    if j == i:
                        #
                        # TODO: why do we ignore the token at the "center" of
                        # the window?
                        #
                        continue
                    if c.hs:
                        fasttext_fast_sentence_sg_hs(c, i, j)
                    if c.negative:
                        fasttext_fast_sentence_sg_neg(c, i, j)


def train_batch_sg(model, sentences, alpha, _work, _l1):
    """Update skip-gram model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        Model to be trained.
    sentences : iterable of list of str
        A single batch: part of the corpus streamed directly from disk/network.
    alpha : float
        Learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _l1 : np.ndarray
        Private working memory for each worker.

    Returns
    -------
    int
        Effective number of words trained.

    """
    cdef:
        FastTextConfig c
        int num_words = 0
        int num_sentences = 0

    init_ft_config(&c, model, alpha, _work, _l1)

    num_words, num_sentences = populate_ft_config(&c, model.wv.vocab, model.wv.buckets_word, sentences)

    # precompute "reduced window" offsets in a single randint() call
    for i, randint in enumerate(model.random.randint(0, c.window, num_words)):
        c.reduced_windows[i] = randint

    with nogil:
        fasttext_train_any(&c, num_sentences, 1)

    return num_words


def train_batch_cbow(model, sentences, alpha, _work, _neu1):
    """Update the CBOW model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        Model to be trained.
    sentences : iterable of list of str
        A single batch: part of the corpus streamed directly from disk/network.
    alpha : float
        Learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.

    Returns
    -------
    int
        Effective number of words trained.

    """
    cdef:
        FastTextConfig c
        int num_words = 0
        int num_sentences = 0

    init_ft_config(&c, model, alpha, _work, _neu1)

    num_words, num_sentences = populate_ft_config(&c, model.wv.vocab, model.wv.buckets_word, sentences)

    # precompute "reduced window" offsets in a single randint() call
    for i, randint in enumerate(model.random.randint(0, c.window, num_words)):
        c.reduced_windows[i] = randint

    # release GIL & train on all sentences in the batch
    with nogil:
        fasttext_train_any(&c, num_sentences, 0)

    return num_words


def init():
    """Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized into table EXP_TABLE.
    Also calculate log(sigmoid(x)) into LOG_TABLE.

    Returns
    -------
    {0, 1, 2}
        Enumeration to signify underlying data type returned by the BLAS dot product calculation.
        0 signifies double, 1 signifies double, and 2 signifies that custom cython loops were used
        instead of BLAS.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if abs(d_res - expected) < 0.0001:
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif abs(p_res[0] - expected) < 0.0001:
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
