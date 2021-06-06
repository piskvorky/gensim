#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True

from libc.math cimport pow, log, exp, abs, sqrt
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t


cimport numpy as np


cdef init_sslm_config(StateSpaceLanguageModelConfig * config, model):

    config[0].num_time_slices = model.num_time_slices
    config[0].vocab_len = model.vocab_len

    config[0].chain_variance = model.chain_variance
    config[0].obs_variance = model.obs_variance

    config[0].obs = <REAL_t *> (np.PyArray_DATA(model.obs))
    config[0].mean = <REAL_t *> (np.PyArray_DATA(model.mean))
    config[0].variance = <REAL_t *> (np.PyArray_DATA(model.variance))

    config[0].fwd_mean = <REAL_t *> (np.PyArray_DATA(model.fwd_mean))
    config[0].fwd_variance = <REAL_t *> (np.PyArray_DATA(model.fwd_variance))

    config[0].zeta = <REAL_t *> (np.PyArray_DATA(model.zeta))

    config[0].e_log_prob = <REAL_t *> (np.PyArray_DATA(model.e_log_prob))

    # Default initialization should raise exception if it used without proper initialization
    config[0].deriv = NULL
    config[0].mean_deriv_mtx = NULL
    word = -1


import numpy as np
from scipy import optimize


cdef compute_post_mean(REAL_t *mean, REAL_t *fwd_mean, const REAL_t *fwd_variance, const REAL_t *obs,
                       const int word, const int num_time_slices,
                       const REAL_t obs_variance, const REAL_t chain_variance):
    """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
    <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

    Notes
    -----
    This function essentially computes E[\beta_{t,w}] for t = 1:T.

    .. :math::

        Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
        = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
        (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

    .. :math::

        Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
        = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
        (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

    Parameters
    ----------
    mean: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the mean values to be used for inference for each word for a time slice.
    fwd_mean: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        The forward posterior values for the mean
    fwd_variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        The forward posterior values for the variance
    obs: obs
        A matrix containing the document to topic ratios
    num_time_slices: int 
        Number of time slices in the model.
    word: int
        The word's ID to process.
    chain_variance : REAL_t 
        Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
    obs_variance: REAL_t 
        Observed variance used to approximate the true and forward variance.


    """
    # print("")
    cdef Py_ssize_t T = num_time_slices
    obs = obs + word * num_time_slices
    fwd_variance = fwd_variance + word * (num_time_slices + 1)
    mean = mean + word * (num_time_slices + 1)
    fwd_mean = fwd_mean + word * (num_time_slices + 1)

    cdef Py_ssize_t t
    cdef REAL_t c

    # forward
    fwd_mean[0] = 0

    for t in range(1, T + 1):
        c = obs_variance / (fwd_variance[t - 1] + chain_variance + obs_variance)
        fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]

    # backward pass
    mean[T] = fwd_mean[T]

    for t in range(T - 1, -1, -1):
        if chain_variance == 0.0:
            c = 0.0
        else:
            c = chain_variance / (fwd_variance[t] + chain_variance)
        mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]


cdef compute_post_variance(REAL_t *variance, REAL_t *fwd_variance,
                           const REAL_t obs_variance, const REAL_t chain_variance,
                           const int word, const int num_time_slices):
    r"""Get the variance, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
    <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

    This function accepts the word to compute variance for, along with the associated sslm class object,
    and returns the `variance` and the posterior approximation `fwd_variance`.

    Notes
    -----
    This function essentially computes Var[\beta_{t,w}] for t = 1:T

    .. :math::

        fwd\_variance[t] \equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\ for\ 1:t) =
        (obs\_variance / fwd\_variance[t - 1] + chain\_variance + obs\_variance ) *
        (fwd\_variance[t - 1] + obs\_variance)

    .. :math::

        variance[t] \equiv E((beta_{t,w}-mean\_cap_{t,w})^2 |beta\_cap_{t}\ for\ 1:t) =
        fwd\_variance[t - 1] + (fwd\_variance[t - 1] / fwd\_variance[t - 1] + obs\_variance)^2 *
        (variance[t - 1] - (fwd\_variance[t-1] + obs\_variance))

    Parameters
    ----------
    variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the variance values to be used for inference of word in a time slice
    fwd_variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        The forward posterior values for the variance
    obs_variance: REAL_t
        Observed variance used to approximate the true and forward variance.
    chain_variance : REAL_t
        Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
    word: int
        The word's ID to process.
    num_time_slices: int 
        Number of time slices in the model.
    """
    cdef int INIT_VARIANCE_CONST = 1000

    cdef Py_ssize_t T = num_time_slices
    variance = variance + word * (num_time_slices + 1)
    fwd_variance = fwd_variance + word * (num_time_slices + 1)
    cdef REAL_t c
    cdef Py_ssize_t t

    # forward pass. Set initial variance very high
    fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST

    for t in range(1, T + 1):
        if obs_variance != 0.0:
            c = obs_variance / (fwd_variance[t - 1] + chain_variance + obs_variance)
        else:
            c = 0
        fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)

    # backward pass
    variance[T] = fwd_variance[T]
    for t in range(T - 1, -1, -1):
        if fwd_variance[t] > 0.0:
            c = pow((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
        else:
            c = 0
        variance[t] = c * (variance[t + 1] - chain_variance) + (1 - c) * fwd_variance[t]


cdef compute_mean_deriv(REAL_t *deriv, const REAL_t *variance, const REAL_t obs_variance, const REAL_t chain_variance,
                        const int word, const int time, const int num_time_slices):
    """Helper functions for optimizing a function.

    Compute the derivative of:

    .. :math::

        E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.

    Parameters
    ----------
    deriv : a pointer to C-array of the REAL_t type and size of (num_time_slices)
        Derivative for each time slice.
    variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the variance values to be used for inference of word in a time slice
    obs_variance: REAL_t
        Observed variance used to approximate the true and forward variance.
    chain_variance : REAL_t
        Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
    word : int
        The word's ID.
    time : int
        The time slice.
    num_time_slices
        Number of time slices in the model.
    """

    cdef Py_ssize_t T = num_time_slices

    cdef REAL_t *fwd_variance = variance + word * (num_time_slices + 1)
    cdef Py_ssize_t t
    cdef REAL_t val
    cdef REAL_t w

    deriv[0] = 0

    # forward pass
    for t in range(1, T + 1):
        if obs_variance > 0.0:
            w = obs_variance / (fwd_variance[t - 1] + chain_variance + obs_variance)
        else:
            w = 0.0
        val = w * deriv[t - 1]

        if time == t - 1:
            val += (1 - w)

        deriv[t] = val

    for t in range(T - 1, -1, -1):
        if chain_variance == 0.0:
            w = 0.0
        else:
            w = chain_variance / (fwd_variance[t] + chain_variance)

        deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]


cdef compute_obs_deriv(REAL_t *deriv, const REAL_t *mean, const REAL_t *mean_deriv_mtx, const REAL_t *variance,
                       const REAL_t *zeta, const REAL_t *totals, const REAL_t *word_counts,
                       const REAL_t chain_variance, const int word, const int num_time_slices
                       ):
    """Derivation of obs which is used in derivative function `df_obs` while optimizing.

    Parameters
    ----------
    deriv:
    mean: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the mean values to be used for inference for each word for a time slice.
    mean_deriv_mtx:
        Mean derivative for each time slice.
    variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the variance values to be used for inference of word in a time slice
    zeta: a pointer to C-array of the REAL_t type and size of (num_time_slices)
        An extra variational parameter with a value for each time slice.
    word_counts : a pointer to C-array of the REAL_t type and size of (num_time_slices)
        Total word counts for each time slice.
    totals : a pointer to C-array of the REAL_t type and size of (num_time_slices)
        The totals for each time slice.
    chain_variance : REAL_t
        Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
    word : int
        The word's ID to process.
    num_time_slices: int 
        Number of time slices in the model.
    """

    cdef REAL_t init_mult = 1000

    cdef Py_ssize_t T = num_time_slices

    mean = mean + word * (num_time_slices + 1)
    variance = variance + word * (num_time_slices + 1)

    cdef Py_ssize_t u, t
    cdef REAL_t term1, term2, term3, term4

    cdef REAL_t *temp_vect = <REAL_t *> malloc(T * sizeof(REAL_t))
    if temp_vect == NULL:
        raise

    for u in range(T):
        temp_vect[u] = exp(mean[u + 1] + variance[u + 1] / 2)

    cdef REAL_t *mean_deriv = NULL

    for t in range(T):

        mean_deriv = mean_deriv_mtx + t * (num_time_slices + 1)
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0
        term4 = 0.0

        for u in range(1, T + 1):
            term1 += (mean[u] - mean[u - 1]) * (mean_deriv[u] - mean_deriv[u - 1])
            term2 += (word_counts[u - 1] - (totals[u - 1] * temp_vect[u - 1] / zeta[u - 1])) * mean_deriv[u]

        if chain_variance != 0.0:

            # TODO should not it be term2 here, in not prime version term2
            term1 = - (term1 / chain_variance) - (mean[0] * mean_deriv[0]) / (init_mult * chain_variance)
        else:
            term1 = 0.0

        deriv[t] = term1 + term2 + term3 + term4

    free(temp_vect)

cdef update_zeta(REAL_t * zeta, const REAL_t *mean, const REAL_t *variance,
                 const int num_time_slices, const int vocab_len):
    """Update the Zeta variational parameter.

    Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
    over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.
    Parameters
    ----------
    zeta: a pointer to C-array of the REAL_t type and size of (num_time_slices)
        An extra variational parameter with a value for each time slice.
    mean: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the mean values to be used for inference for each word for a time slice.
    variance: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the variance values to be used for inference of word in a time slice
    num_time_slices: int 
        Number of time slies in the model.
    vocab_len:
        Length of the model's vocabulary 
    """

    cdef Py_ssize_t i, w

    cdef REAL_t temp

    for i in range(num_time_slices):
        temp = 0.0

        for w in range(vocab_len):
            # TODO check if it possible to use BLAS here
            # TODO compare with original code, some info about log
            temp += exp(mean[w * (num_time_slices + 1) + i + 1] + variance[
                w * (num_time_slices + 1) + i + 1] / 2.0)

        zeta[i] = temp


cdef REAL_t compute_bound(StateSpaceLanguageModelConfig * config, REAL_t *sstats, REAL_t *totals):
    """Compute the maximized lower bound achieved for the log probability of the true posterior.

    Uses the formula presented in the appendix of the DTM paper (formula no. 5).

    Parameters
    ----------
    config 
        A pointer to the instance of config structure which stores links to the data.
    sstats : numpy.ndarray
        Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
        time slice, expected shape (`self.vocab_len`, `num_topics`).
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.

    Returns
    -------
    float
        The maximized lower bound.

    """

    cdef int vocab_len = config[0].vocab_len
    cdef Py_ssize_t T = config[0].num_time_slices

    cdef REAL_t term_1 = 0.0
    cdef REAL_t term_2 = 0.0
    cdef REAL_t term_3 = 0.0

    cdef REAL_t val = 0.0
    cdef REAL_t ent = 0.0

    cdef REAL_t chain_variance = config[0].chain_variance

    cdef REAL_t *mean = config[0].mean
    cdef REAL_t *fwd_mean = config[0].fwd_mean
    cdef REAL_t *variance = config[0].variance
    cdef REAL_t *zeta = config[0].zeta

    cdef Py_ssize_t i, t, w

    for i in range(vocab_len):
        config[0].word = i
        compute_post_mean(mean, fwd_mean, config[0].fwd_variance, config[0].obs,
                          i, T, config[0].obs_variance, chain_variance)

    update_zeta(zeta, mean, variance, T, vocab_len)

    val = 0.0

    for i in range(vocab_len):
        val += variance[i * (config[0].num_time_slices + 1)] - variance[i * (config[0].num_time_slices + 1) + T]

    # TODO check if it is correct, not val (2.0 / chain_variance)
    val = val / 2.0 * chain_variance

    cdef REAL_t m, prev_m, v

    for t in range(1, T + 1):

        term_1 = 0.0
        term_2 = 0.0
        ent = 0.0

        for w in range(vocab_len):
            m = mean[w * (config[0].num_time_slices + 1) + t]
            prev_m = mean[w * (config[0].num_time_slices + 1) + t - 1]

            v = variance[w * (config[0].num_time_slices + 1) + t]

            term_1 += \
                (pow(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - log(chain_variance)
            term_2 += sstats[w * config[0].num_time_slices + t - 1] * m

            ent += log(v) / 2  # note the 2pi's cancel with term1 (see doc)

        term_3 = -totals[t - 1] * log(zeta[t - 1])

        val += term_2 + term_3 + ent - term_1

    return val

#
cdef compute_expected_log_prob(REAL_t *e_log_prob, const REAL_t *zeta, const REAL_t *mean,
                               const int vocab_len, const int num_time_slices):
    """Compute the expected log probability given values of m.

    The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
    The below implementation is the result of solving the equation and is implemented as in the original
    Blei DTM code.
    
    Parameters
    ----------
    e_log_prob:
        A matrix containing the topic to word ratios.
    zeta: a pointer to C-array of the REAL_t type and size of (num_time_slices)
        An extra variational parameter with a value for each time slice.
    mean: a pointer to C-array of the REAL_t type and size of (vocab_len, num_time_slices + 1)
        Contains the mean values to be used for inference for each word for a time slice.
    num_time_slices: int 
        Number of time slies in the model.
    vocab_len:
        Length of the model's vocabulary 
  
    """

    cdef Py_ssize_t w, t

    for w in range(vocab_len):
        for t in range(num_time_slices):
            e_log_prob[w * num_time_slices + t] = mean[w * (num_time_slices + 1) + t + 1] - log(
                zeta[t])


cdef update_obs(StateSpaceLanguageModelConfig *config, REAL_t *sstats, REAL_t *totals):
    """Optimize the bound with respect to the observed variables.

    Parameters
    ----------
    config 
        A pointer to the instance of config structure which stores links to the data.
    sstats
        Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
        current time slice, expected shape (vocab_len, num_time_slices).
    
    totals:

    Returns
    -------
    (numpy.ndarray of float, numpy.ndarray of float)
        The updated optimized values for obs and the zeta variational parameter.

    """

    cdef int OBS_NORM_CUTOFF = 2
    cdef REAL_t STEP_SIZE = 0.01
    cdef REAL_t TOL = 0.001

    cdef Py_ssize_t W = config[0].vocab_len
    cdef Py_ssize_t T = config[0].num_time_slices

    cdef int runs = 0

    cdef REAL_t *mean_deriv_mtx = <REAL_t *> malloc(T * (T + 1) * sizeof(REAL_t))
    if mean_deriv_mtx == NULL:
        raise
    cdef Py_ssize_t w, t
    cdef REAL_t counts_norm

    cdef REAL_t * obs
    config[0].totals = totals

    np_norm_cutoff_obs = np.zeros(T, dtype=np.double)
    np_w_counts = np.zeros(T, dtype=np.double)
    np_obs = np.zeros(T, dtype=np.double)

    # This is a work memory for df_obs function
    working_array = np.zeros(T, dtype=np.double)

    cdef REAL_t *norm_cutoff_obs = NULL
    cdef REAL_t *w_counts

    for w in range(W):
        w_counts = sstats + w * config[0].num_time_slices
        config[0].word = w

        counts_norm = 0.0

        # now we find L2 norm of w_counts
        for i in range(config[0].num_time_slices):
            counts_norm += w_counts[i] * w_counts[i]

        counts_norm = sqrt(counts_norm)

        if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not NULL:
            obs = config[0].obs + w * config[0].num_time_slices
            norm_cutoff_obs = <REAL_t *> (np.PyArray_DATA(np_norm_cutoff_obs))
            # norm_cutoff_obs = <REAL_t *> malloc(config[0].num_time_slices * sizeof(REAL_t))
            if norm_cutoff_obs == NULL:
                raise
            memcpy(norm_cutoff_obs, obs, config[0].num_time_slices * sizeof(REAL_t))

        else:
            if counts_norm < OBS_NORM_CUTOFF:
                np_w_counts = np.zeros(config[0].num_time_slices, dtype=np.double)
                w_counts = <REAL_t *> (np.PyArray_DATA(np_w_counts))

            for t in range(T):
                compute_mean_deriv(mean_deriv_mtx + t * (config[0].num_time_slices + 1), config[0].variance,
                                   config[0].obs_variance, config[0].chain_variance, w, t, T)

            np_deriv = np.zeros(T, dtype=np.double)
            deriv = <REAL_t *> (np.PyArray_DATA(np_deriv))
            config[0].deriv = deriv

            obs = <REAL_t *> (np.PyArray_DATA(np_obs))
            memcpy(obs, config[0].obs + w * config[0].num_time_slices, config[0].num_time_slices * sizeof(REAL_t))

            config[0].word_counts = w_counts
            config[0].mean_deriv_mtx = mean_deriv_mtx

            # Passing C config structure as integer in Python code
            args = (<uintptr_t>(config),working_array)

            temp_obs = optimize.fmin_cg(
                f=f_obs, fprime=df_obs, x0=np_obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0
            )

            obs = <REAL_t *> (np.PyArray_DATA(temp_obs))

            runs += 1

            if counts_norm < OBS_NORM_CUTOFF:

                norm_cutoff_obs = <REAL_t *> (np.PyArray_DATA(np_norm_cutoff_obs))
                memcpy(norm_cutoff_obs, obs, config[0].num_time_slices * sizeof(REAL_t))

            memcpy(config[0].obs + w * config[0].num_time_slices, obs, config[0].num_time_slices * sizeof(REAL_t))

    update_zeta(config[0].zeta, config[0].mean, config[0].variance,
                config[0].num_time_slices, config[0].vocab_len)

    free(mean_deriv_mtx)


# the following functions are used in update_obs as the objective function.
def f_obs(_x, uintptr_t c, work_array):
    """Function which we are optimising for minimizing obs.

    Parameters
    ----------
    _x : np.ndarray of float64
        The obs values for this word.
    c: uintptr_t
        An pointer's value or address where config structure is stored in the memory.
    work_array:
        Additional work memory
    Returns
    -------
    REAL_t
        The value of the objective function evaluated at point `x`.

    """
    cdef StateSpaceLanguageModelConfig * config = <StateSpaceLanguageModelConfig *> c
    cdef REAL_t *x = <REAL_t *> (np.PyArray_DATA(_x))

    # flag
    cdef int init_mult = 1000
    cdef Py_ssize_t T = config[0].num_time_slices
    cdef Py_ssize_t t

    cdef REAL_t val = 0.0
    cdef REAL_t term1 = 0.0
    cdef REAL_t term2 = 0.0

    # term 3 and 4 for DIM
    cdef REAL_t term3 = 0.0
    cdef REAL_t term4 = 0.0

    # obs[word] = x
    memcpy(config[0].obs + config[0].word * config[0].num_time_slices, x, config[0].num_time_slices * sizeof(REAL_t))

    compute_post_mean(config[0].mean, config[0].fwd_mean, config[0].fwd_variance, config[0].obs,
                      config[0].word, config[0].num_time_slices, config[0].obs_variance, config[0].chain_variance)

    cdef REAL_t *mean = config[0].mean + config[0].word * (config[0].num_time_slices + 1)
    cdef REAL_t *variance = config[0].variance + config[0].word * (config[0].num_time_slices + 1)

    for t in range(1, T + 1):

        term1 += (mean[t] - mean[t - 1]) * (mean[t] - mean[t - 1])

        term2 += config[0].word_counts[t - 1] * mean[t] - config[0].totals[t - 1] * \
                 exp(mean[t] + variance[t] / 2) / config[0].zeta[t - 1]


    if config[0].chain_variance > 0.0:

        term1 = -(term1 / (2 * config[0].chain_variance)) - \
                mean[0] * mean[0] / (2 * init_mult * config[0].chain_variance)
    else:
        term1 = 0.0

    return -(term1 + term2 + term3 + term4)


def df_obs(_x, uintptr_t c, work_array):
    """Derivative of the objective function which optimises obs.

    Parameters
    ----------
    _x : np.ndarray of float64
        The obs values for this word.
    c: uintptr_t
        An pointer's value or address where config structure is stored in the memory.
    work_array:
        Additional work memory

    Returns
    -------
    np.ndarray of float64
        The derivative of the objective function evaluated at point `x`.

    """
    cdef StateSpaceLanguageModelConfig * config = <StateSpaceLanguageModelConfig *> c
    cdef REAL_t *x = <REAL_t *> (np.PyArray_DATA(_x))


    memcpy(config[0].obs + config[0].num_time_slices * config[0].word, x, config[0].num_time_slices * sizeof(REAL_t))

    compute_post_mean(config[0].mean, config[0].fwd_mean, config[0].fwd_variance, config[0].obs,
                      config[0].word, config[0].num_time_slices, config[0].obs_variance, config[0].chain_variance)

    compute_obs_deriv(config[0].deriv, config[0].mean, config[0].mean_deriv_mtx, config[0].variance,
                      config[0].zeta, config[0].totals, config[0].word_counts,
                      config[0].chain_variance, config[0].word, config[0].num_time_slices)

    for i in range(config[0].num_time_slices):
        config[0].deriv[i] = -config[0].deriv[i]

    cdef REAL_t *temp_ptr = <REAL_t *>(np.PyArray_DATA(work_array))

    memcpy(temp_ptr, config[0].deriv, config[0].num_time_slices * sizeof(REAL_t))

    return work_array


def fit_sslm(model, np_sstats):
    """Fits variational distribution.

    This is essentially the m-step.
    Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
    statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
    :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

    Parameters
    ----------
    model
        An instance of SSLM model
    sstats : numpy.ndarray
        Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
        current time slice, expected shape (vocab_len, num_time_slices).

    Returns
    -------
    float
        The lower bound for the true posterior achieved using the fitted approximate distribution.

    """

    # Initialize C structures based on Python instance of the model
    cdef StateSpaceLanguageModelConfig* config = <StateSpaceLanguageModelConfig *>malloc(sizeof(StateSpaceLanguageModelConfig))
    init_sslm_config(config, model)

    cdef int W = config[0].vocab_len
    cdef REAL_t old_bound = 0.0
    cdef REAL_t sslm_fit_threshold = 0.000001
    cdef int sslm_max_iter = 2
    cdef REAL_t converged = sslm_fit_threshold + 1

    cdef int w

    for w in range(W):
        config[0].word = w
        compute_post_variance(config[0].variance, config[0].fwd_variance, config[0].obs_variance,
                              config[0].chain_variance, w, config[0].num_time_slices)

    cdef REAL_t *sstats = <REAL_t *> (np.PyArray_DATA(np_sstats))

    # column sum of sstats
    np_totals = np_sstats.sum(axis=0)
    cdef REAL_t *totals = <REAL_t *> (np.PyArray_DATA(np_totals))

    cdef int iter_ = 0

    cdef REAL_t bound = compute_bound(config, sstats, totals)

    while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
        iter_ += 1
        old_bound = bound
        update_obs(config, sstats, totals)

        bound = compute_bound(config, sstats, totals)

        converged = abs((bound - old_bound) / old_bound)

    compute_expected_log_prob(config[0].e_log_prob, config[0].zeta, config[0].mean,
                              W, config[0].num_time_slices)

    free(config)
    return bound
