#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""Optimized Cython functions for training a :class:`~gensim.models.fasttext.FastText` model.

The main entry point is :func:`~gensim.models.fasttext_inner.train_batch_any`
which may be called directly from Python code.

Notes
-----
The implementation of the above functions heavily depends on the
FastTextConfig struct defined in :file:`gensim/models/fasttext_inner.pxd`.

The gensim.models.word2vec.FAST_VERSION value reports what flavor of BLAS
we're currently using:

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
# The versions are as chosen in word2vec_inner.pyx, and aliased to `our_` functions

from gensim.models.word2vec_inner cimport bisect_left, random_int32, scopy, sscal, \
     REAL_t, our_dot, our_saxpy

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000

DEF EXP_TABLE_SIZE = 512
DEF MAX_EXP = 8

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

    cdef:
        np.uint32_t word_index = c.indexes[j]
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]

    cdef long long row1 = word2_index * c.size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))

    scopy(&c.size, &c.syn0_vocab[row1], &ONE, c.neu1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[subwords_index[d] * c.size], &ONE, c.neu1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&c.size, &norm_factor, c.neu1, &ONE)

    for d in range(c.negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(
                c.cum_table, (c.next_random >> 16) % c.cum_table[c.cum_table_len-1], 0, c.cum_table_len)
            c.next_random = (c.next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * c.alpha
        our_saxpy(&c.size, &g, &c.syn1neg[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
    our_saxpy(&c.size, &c.vocab_lockf[word2_index % c.vocab_lockf_len], c.work, &ONE, &c.syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        our_saxpy(&c.size, &c.ngrams_lockf[subwords_index[d] % c.ngrams_lockf_len],
                  c.work, &ONE, &c.syn0_ngrams[subwords_index[d]*c.size], &ONE)


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
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]

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
    cdef long long row1 = word2_index * c.size, row2
    cdef REAL_t f, g, f_dot

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))

    scopy(&c.size, &c.syn0_vocab[row1], &ONE, c.neu1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            row2 = subwords_index[d] * c.size
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[row2], &ONE, c.neu1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&c.size, &norm_factor, c.neu1, &ONE)

    for b in range(codelen):
        row2 = word_point[b] * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1[row2], &ONE)

    our_saxpy(&c.size, &c.vocab_lockf[word2_index % c.vocab_lockf_len], c.work, &ONE, &c.syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        row2 = subwords_index[d] * c.size
        our_saxpy(
            &c.size, &c.ngrams_lockf[subwords_index[d] % c.ngrams_lockf_len], c.work, &ONE,
            &c.syn0_ngrams[row2], &ONE)


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

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = c.indexes[i]

    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&c.size, &ONEF, &c.syn0_vocab[c.indexes[m] * c.size], &ONE, c.neu1, &ONE)
        for d in range(c.subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[c.subwords_idx[m][d] * c.size], &ONE, c.neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if c.cbow_mean:
        sscal(&c.size, &inv_count, c.neu1, &ONE)

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))

    for d in range(c.negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(c.cum_table, (c.next_random >> 16) % c.cum_table[c.cum_table_len-1], 0, c.cum_table_len)
            c.next_random = (c.next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP:
            f = 0.0
        elif f_dot >= MAX_EXP:
            f = 1.0
        else:
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1neg[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1neg[row2], &ONE)

    if not c.cbow_mean:  # divide error over summed window vectors
        sscal(&c.size, &inv_count, c.work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(
            &c.size, &c.vocab_lockf[c.indexes[m] % c.vocab_lockf_len], c.work, &ONE,
            &c.syn0_vocab[c.indexes[m]*c.size], &ONE)
        for d in range(c.subwords_idx_len[m]):
            our_saxpy(
                &c.size, &c.ngrams_lockf[c.subwords_idx[m][d] % c.ngrams_lockf_len], c.work, &ONE,
                &c.syn0_ngrams[c.subwords_idx[m][d]*c.size], &ONE)


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

    cdef long long b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot
    cdef int m

    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&c.size, &ONEF, &c.syn0_vocab[c.indexes[m] * c.size], &ONE, c.neu1, &ONE)
        for d in range(c.subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[c.subwords_idx[m][d] * c.size], &ONE, c.neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if c.cbow_mean:
        sscal(&c.size, &inv_count, c.neu1, &ONE)

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    for b in range(c.codelens[i]):
        row2 = word_point[b] * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1[row2], &ONE)

    if not c.cbow_mean:  # divide error over summed window vectors
        sscal(&c.size, &inv_count, c.work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(
            &c.size, &c.vocab_lockf[c.indexes[m] % c.vocab_lockf_len], c.work, &ONE,
            &c.syn0_vocab[c.indexes[m]*c.size], &ONE)
        for d in range(c.subwords_idx_len[m]):
            our_saxpy(
                &c.size, &c.ngrams_lockf[c.subwords_idx[m][d] % c.ngrams_lockf_len], c.work, &ONE,
                &c.syn0_ngrams[c.subwords_idx[m][d]*c.size], &ONE)


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
    c.sg = model.sg
    c.hs = model.hs
    c.negative = model.negative
    c.sample = (model.sample != 0)
    c.cbow_mean = model.cbow_mean
    c.window = model.window
    c.workers = model.workers

    c.syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    c.syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))

    # EXPERIMENTAL lockf scaled suppression/enablement of training
    c.vocab_lockf = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab_lockf))
    c.vocab_lockf_len = len(model.wv.vectors_vocab_lockf)
    c.ngrams_lockf = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams_lockf))
    c.ngrams_lockf_len = len(model.wv.vectors_ngrams_lockf)

    c.alpha = alpha
    c.size = model.wv.vector_size

    if c.hs:
        c.syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if c.negative:
        c.syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        c.cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        c.cum_table_len = len(model.cum_table)
    if c.negative or c.sample:
        c.next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c.work = <REAL_t *>np.PyArray_DATA(_work)
    c.neu1 = <REAL_t *>np.PyArray_DATA(_neu1)


cdef object populate_ft_config(FastTextConfig *c, wv, buckets_word, sentences):
    """Prepare C structures so we can go "full C" and release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token and store the result up front to save time: we'll be seeing each
    token multiple times because of windowing, so better to do the work once
    here.

    Parameters
    ----------
    c : FastTextConfig*
        A pointer to the struct that will contain the populated indices.
    wv : FastTextKeyedVectors
        The vocabulary
    buckets_word : list
        A list containing the buckets each word appears in
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
    cdef np.uint32_t *vocab_sample_ints
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0

    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = wv.expandos['code']
        vocab_points = wv.expandos['point']
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word_index = wv.key_to_index.get(token, None)
            if word_index is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word_index

            if wv.bucket:
                c.subwords_idx_len[effective_words] = <int>(len(buckets_word[word_index]))
                c.subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(buckets_word[word_index])
            else:
                c.subwords_idx_len[effective_words] = 0

            if c.hs:
                c.codelens[effective_words] = <int>len(vocab_codes[word_index])
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])

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


cdef void fasttext_train_any(FastTextConfig *c, int num_sentences) nogil:
    """Performs training on a fully initialized and populated configuration.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to the configuration struct.
    num_sentences : int
        The number of sentences to train.

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
            if c.sg == 0:
                if c.hs:
                    fasttext_fast_sentence_cbow_hs(c, i, window_start, window_end)
                if c.negative:
                    fasttext_fast_sentence_cbow_neg(c, i, window_start, window_end)
            else:
                for j in range(window_start, window_end):
                    if j == i:
                        # no reason to train a center word as predicting itself
                        continue
                    if c.hs:
                        fasttext_fast_sentence_sg_hs(c, i, j)
                    if c.negative:
                        fasttext_fast_sentence_sg_neg(c, i, j)


def train_batch_any(model, sentences, alpha, _work, _neu1):
    """Update the model by training on a sequence of sentences.

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

    num_words, num_sentences = populate_ft_config(&c, model.wv, model.wv.buckets_word, sentences)

    # precompute "reduced window" offsets in a single randint() call
    if model.shrink_windows:
        for i, randint in enumerate(model.random.randint(0, c.window, num_words)):
            c.reduced_windows[i] = randint
    else:
        for i in range(num_words):
            c.reduced_windows[i] = 0

    # release GIL & train on all sentences in the batch
    with nogil:
        fasttext_train_any(&c, num_sentences)

    return num_words


cpdef ft_hash_bytes(bytes bytez):
    """Calculate hash based on `bytez`.
    Reproduce `hash method from Facebook fastText implementation
    <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`_.

    Parameters
    ----------
    bytez : bytes
        The string whose hash needs to be calculated, encoded as UTF-8.

    Returns
    -------
    unsigned int
        The hash of the string.

    """
    cdef np.uint32_t h = 2166136261
    cdef char b

    for b in bytez:
        h = h ^ <np.uint32_t>(<np.int8_t>b)
        h = h * 16777619
    return h


cpdef compute_ngrams(word, unsigned int min_n, unsigned int max_n):
    """Get the list of all possible ngrams for a given word.

    Parameters
    ----------
    word : str
        The word whose ngrams need to be computed.
    min_n : unsigned int
        Minimum character length of the ngrams.
    max_n : unsigned int
        Maximum character length of the ngrams.

    Returns
    -------
    list of str
        Sequence of character ngrams.

    """
    cdef unicode extended_word = f'<{word}>'
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams

#
# UTF-8 bytes that begin with 10 are subsequent bytes of a multi-byte sequence,
# as opposed to a new character.
#
cdef unsigned char _MB_MASK = 0xC0
cdef unsigned char _MB_START = 0x80


cpdef compute_ngrams_bytes(word, unsigned int min_n, unsigned int max_n):
    """Computes ngrams for a word.

    Ported from the original FB implementation.

    Parameters
    ----------
    word : str
        A unicode string.
    min_n : unsigned int
        The minimum ngram length.
    max_n : unsigned int
        The maximum ngram length.

    Returns:
    --------
    list of str
        A list of ngrams, where each ngram is a list of **bytes**.

    See Also
    --------
    `Original implementation <https://github.com/facebookresearch/fastText/blob/7842495a4d64c7a3bb4339d45d6e64321d002ed8/src/dictionary.cc#L172>`__

    """
    cdef bytes utf8_word = ('<%s>' % word).encode("utf-8")
    cdef const unsigned char *bytez = utf8_word
    cdef size_t num_bytes = len(utf8_word)
    cdef size_t j, i, n

    ngrams = []
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue

        j, n = i, 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                ngram = bytes(bytez[i:j])
                ngrams.append(ngram)
            n += 1
    return ngrams


def init():
    """Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized into table EXP_TABLE.
    Also calculate log(sigmoid(x)) into LOG_TABLE.

    We recalc, rather than re-use the table from word2vec_inner, because Facebook's FastText
    code uses a 512-slot table rather than the 1000 precedent of word2vec.c.
    """
    cdef int i

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )


init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
