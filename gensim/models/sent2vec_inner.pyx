#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
# distutils : language = c++
# cython : embedsignature = True

import cython
import numpy as np
import sys
from numpy import zeros, float32 as REAL
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX, srand

# scipy <= 0.15
try:
     from scipy.linalg.blas import fblas
except ImportError:
     # in scipy > 0.15, fblas function has been removed
     import scipy.linalg.blas as fblas

from word2vec_inner cimport bisect_left, random_int32, \
     scopy, saxpy, sdot, dsdot, snrm2, sscal, \
     REAL_t, EXP_TABLE, \
     LOG_TABLE, our_dot, our_saxpy, \
     our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

from word2vec import FAST_VERSION

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef int EXP_TABLE_SIZE = 1000
cdef int MAX_EXP = 6

cdef REAL_t negative_sampling(const int target, const REAL_t lr, REAL_t *grad, REAL_t *wo,
                              REAL_t *hidden, const int vector_size, const int neg, int *negatives,
                              int *negpos, const int negatives_len)nogil:
    """
    Get loss using negative sampling.
    Pararmeters
    -----------
    target : const int
        Word id of target word.
    lr : REAL_t
        Current learning rate.
    grad : REAL_t *
        Pointer to gradient vector.
    wo : REAL_t *
        Pointer to output sentence vector.
    hidden : REAL_t *
        Pointer to hidden vector.
    vector_size : const int
        Size of the sentence embeddings.
    neg : const int
        Number of noise words to be drawn for negative sampling.
    negatives : int *
        Pointer to array containing noise words.
    negpos : int *
        Pointer to position of noise word to be drawn.
    negatives_len : const int
        Length of the array containing noise words.
    Returns
    -------
    loss : REAL_t
        Negative sampling loss.
    """
    cdef REAL_t loss = <REAL_t> 0.0
    cdef int label_true, label_false
    label_true = <int> 1
    label_false = <int> 0
    cdef int temp, new_negpos
    for i from 0 <= i <= neg:
        if i == 0:
            loss += binary_logistic(target, label_true, lr, vector_size, wo, grad, hidden)
        else:
            temp = get_negative(target, negatives, negpos, negatives_len)
            loss += binary_logistic(temp, label_false, lr, vector_size, wo, grad, hidden)
    return loss


cdef REAL_t sigmoid(const REAL_t val)nogil:

    """
    Get value of sigmoid function for input.
    Parameters
    ----------
    val : const REAL_t
        Input value for which sigmoid function is to be computed.
    Returns
    -------
    REAL_t
        Value of sigmoid function for input value.
    """
    cdef int temp = <int>((val + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
    if temp < 0:
        return 0.0
    elif temp >= EXP_TABLE_SIZE:
        return 1.0
    else:
        return EXP_TABLE[temp]
    

cdef REAL_t log(const REAL_t val)nogil:

    """
    Compute log value for given input.
    Parameters
    ----------
    val : const REAL_t
        Input value for which log function is to be calculated.
    Returns
    -------
    REAL_t
        Value of the log function for input value.
    """
    if val >= 1.0:
        return 0.0
    else:
        return LOG_TABLE[<int>(val * EXP_TABLE_SIZE)]


cdef REAL_t binary_logistic(const int target, const int label, const REAL_t lr,
                            const int vector_size, REAL_t *wo, REAL_t *grad,
                            REAL_t *hidden)nogil:

    """
    Compute loss for given target, label and learning rate using binary logistic regression.
    Pararmeters
    -----------
    target : const int
        Word id of target word.
    label : const int
        1 if no negative is sampled, 0 otherwise.
    lr : REAL_t
        Current learning rate.
    grad : REAL_t *
        Pointer to gradient vector.
    wo : REAL_t *
        Pointer to output sentence vector.
    hidden : REAL_t *
        Pointer to hidden vector.
    vector_size : const int
        Size of the sentence embeddings.
    Returns
    -------
    loss : REAL_t
        Binary logistic regression loss.
    """
    cdef REAL_t temp = <REAL_t>our_dot(&vector_size, &wo[target], &ONE, hidden, &ONE)
    cdef REAL_t score = sigmoid(temp)
    cdef REAL_t alpha = lr * (<REAL_t>label - score)
    our_saxpy(&vector_size, &alpha, &wo[target], &ONE, grad, &ONE)
    our_saxpy(&vector_size, &alpha, hidden, &ONE, &wo[target], &ONE)
    if label == 1:
        return -log(score)
    else:
        return -log(ONEF - score)
    

cdef int get_negative(const int target, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    """
    Get a negative from the list of negatives for caluculating nagtive sampling loss.
    Pararmeters
    -----------
    target : const int
        Word id of target word.
    negatives : int *
        Pointer to array containing noise words.
    negpos : int *
        Pointer to position of noise word to be drawn.
    negatives_len : const int
        Length of the array containing noise words.
    Returns
    -------
    int
        Word id of negative sample.
    """
    cdef int negative
    while True:
        negative = negatives[negpos[0]]
        negpos[0] = (negpos[0] + ONE) % negatives_len
        if target != negative:
            break
    return negative


cdef REAL_t update(vector[int] &context, int target, REAL_t lr, REAL_t *hidden, REAL_t *grad,
                  int vector_size, int *negpos, int neg, int negatives_len,
                  REAL_t *wi, REAL_t *wo, int *negatives)nogil:

    """
    Update model's neural weights for given context, target word and learning rate.
    Pararmeters
    -----------
    context : vector[int]
        Vector of word ids of context words.
    target : const int
        Word id of target word.
    lr : REAL_t
        Current learning rate.
    grad : REAL_t *
        Pointer to gradient vector.
    wo : REAL_t *
        Pointer to output sentence vector.
    wi : REAL_t *
        Pointer to input sentence vector.
    hidden : REAL_t *
        Pointer to hidden vector.
    vector_size : const int
        Size of the sentence embeddings.
    neg : const int
        Number of noise words to be drawn for negative sampling.
    negatives : int *
        Pointer to array containing noise words.
    negpos : int *
        Pointer to position of noise word to be drawn.
    negatives_len : const int
        Length of the array containing noise words.
    Returns
    -------
    loss : REAL_t
        Model loss.
    """
    if context.size() <= 0:
        return 0

    cdef REAL_t alpha = ONEF / <REAL_t>(context.size())
    cdef REAL_t loss
    cdef int i

    for i from 0 <= i < context.size():
        our_saxpy(&vector_size, &ONEF, &wi[context[i]], &ONE, hidden, &ONE)
    sscal(&vector_size, &alpha, hidden, &ONE)
    loss = negative_sampling(target, lr, grad, wo, hidden, vector_size, neg, negatives, negpos, negatives_len)
    sscal(&vector_size, &alpha, grad, &ONE)
    for i from 0 <= i < context.size():
        our_saxpy(&vector_size, &ONEF, grad, &ONE, &wi[context[i]], &ONE)
    return loss


cdef REAL_t random_uniform()nogil:

    """
    Returns
    -------
    REAT_t
        Generate random real number between 0 and 1.
    """
    return rand() / (RAND_MAX + 1.0)


cdef int random_range(int a, int b)nogil:

    """
    Generate random integer for given input range.
    Parameters
    ----------
    a : int
        Lower limit of input range.
    b: int
        Upper limit of input range.
    Returns
    -------
    int
        Random integer in given input range [a,b].
    """
    return a + <int>(rand() % ((b - a) + 1))


cdef int get_line(vector[int] &wids, vector[int] &words, int max_line_size)nogil:

    """
    Converting sentence to a list of word ids inferred from the dictionary.
    Parameters
    ----------
    wids : vector[int]
        Vector of word ids for a sentence.
    words : vector[int] &
        Reference to vector of valid word ids for a sentence.
    max_line_size : int
        Maximum length of input sentence.
    Returns
    -------
    ntokens : int
        Number of tokens processed in given sentence.
    """
    cdef int ntokens = <int> 0
    cdef int i

    for i from 0 <= i < wids.size():
        if wids[i] < 0:
            continue
        ntokens += 1
        words.push_back(wids[i])
        if ntokens >= max_line_size:
            break
    return ntokens


cdef void add_ngrams_train(vector[int] &line, int n, int k, int bucket, int size)nogil:

    """
    Training word ngrams for a given context and target word.
    Parameters
    ----------
    line : vector[int] &
        Reference to vector of word ids for input sentence.
    n : int
        Number of word ngrams.
    k : int
        Number of word ngrams dropped while training a Sent2Vec model.
    bucket : int
        Number of hash buckets for vocabulary.
    size : int
        Size of sentence embeddings.
    """
    cdef int num_discarded = 0
    cdef vector[int] discard
    cdef int line_size = line.size()
    cdef int token_to_discard
    cdef unsigned int i, j, h

    for i from 0 <= i < line.size():
        discard.push_back(0)

    while num_discarded < k and line_size - num_discarded > 2:
        token_to_discard = random_range(0, line_size-1)
        if discard[token_to_discard] == 0:
            discard[token_to_discard] = 1
            num_discarded += 1

    for i from 0 <= i < line_size:
        if discard[i] == 1:
            continue
        h = line[i]
        for j from i + 1 <= j < line_size:
            if j >= i + n or discard[j] == 1:
                break
            h = h * 116049371 + line[j]
            line.push_back(size + h % bucket)


cdef (int, int, REAL_t) _do_train_job_util(vector[vector[int]] &word_ids, REAL_t *pdiscard, int max_line_size,
                             int word_ngrams, int dropout_k, REAL_t lr, REAL_t *hidden, REAL_t *grad,
                             int vector_size, int *negpos, int neg, int negatives_len,
                             REAL_t *wi, REAL_t *wo, int *negatives, int bucket, int size)nogil:

    """
    Utility cython nogil function to train a batch of input sentences.
    """
    cdef int local_token_count = 0
    cdef int nexamples = 0
    cdef REAL_t loss = <REAL_t> 0.0
    cdef vector[int] words, context
    cdef int i, j, ntokens_temp

    for i from 0 <= i < word_ids.size():
        ntokens_temp = get_line(word_ids[i], words, max_line_size)
        local_token_count += ntokens_temp
        words_size = words.size()
        if words_size > 0:
            for j from 0 <= j < words_size:
                if random_uniform() > pdiscard[words[j]]:
                    continue
                nexamples += 1
                context.assign(words.begin(), words.end())
                context[j] = 0
                add_ngrams_train(context, word_ngrams, dropout_k, bucket, size)
                loss += update(context, words[j], lr, hidden, grad, vector_size,
                               negpos, neg, negatives_len, wi, wo, negatives)
                context.clear()
        words.clear()

    return local_token_count, nexamples, loss


def _do_train_job_fast(model, sentences_, lr_, hidden_, grad_):

    """
    Train a batch of input sentences using a nogil cython utility function.
    Parameters
    ----------
    model : Object
        Sent2Vec python object.
    sentences_ : iterable of iterable of str
        Input sentences.
    lr_ : float
        Learning rate for given batch of input sentences.
    hidden_ : numpy.ndarray
        Hidden vector for neural network computation.
    grad_ : numpy.ndarray
        Gradient vector for neural network computation.
    Returns
    -------
    local_token_count : int
        Number of tokens processed for given training batch.
    nexamples : int
        Number of examples processed in given training batch.
    loss : REAL_t
        Loss for given training batch.
    """
    cdef REAL_t lr = <REAL_t> lr_
    cdef int vector_size = <int> model.vector_size
    cdef int *negatives = <int *> np.PyArray_DATA(model.negatives)
    cdef int negpos = <int> model.negpos
    cdef int neg = <int> model.negative
    cdef REAL_t *wi = <REAL_t *> np.PyArray_DATA(model.wi)
    cdef REAL_t *wo = <REAL_t *> np.PyArray_DATA(model.wo)
    cdef int negatives_len = <int> len(model.negatives)
    cdef REAL_t *hidden = <REAL_t *> np.PyArray_DATA(hidden_)
    cdef REAL_t *grad = <REAL_t *> np.PyArray_DATA(grad_)
    cdef REAL_t *pdiscard = <REAL_t *> np.PyArray_DATA(np.array(model.vocabulary.pdiscard))
    cdef int max_line_size = <int> (model.vocabulary.max_line_size)
    cdef int size = <int> (model.vocabulary.size)
    cdef int bucket = <int> (model.vocabulary.bucket)
    cdef int word_ngrams = <int> (model.word_ngrams)
    cdef int dropout_k = <int> (model.dropout_k)
    srand(model.seed)

    cdef vector[vector[int]] word_ids
    cdef vector[int] ids
    cdef int i, local_token_count, nexamples
    cdef REAL_t loss

    for sentence in sentences_:
        for word in sentence:
            h = model.vocabulary.find(word)
            ids.push_back(<int> model.vocabulary.word2int[h])
        word_ids.push_back(ids)
        ids.clear()

    with nogil:
        local_token_count, nexamples, loss = _do_train_job_util(word_ids, pdiscard,
                                                                max_line_size, word_ngrams,
                                                                dropout_k, lr, hidden,
                                                                grad, vector_size, &negpos,
                                                                neg, negatives_len, wi, wo,
                                                                negatives, bucket, size)
    model.negpos = negpos
    return local_token_count, nexamples, loss
