cimport numpy as np
ctypedef np.float64_t REAL_t



cdef struct StateSpaceLanguageModelConfig:
    int num_time_slices, vocab_len
    REAL_t chain_variance, obs_variance

    REAL_t * obs
    REAL_t * mean
    REAL_t * variance

    REAL_t * fwd_mean
    REAL_t * fwd_variance
    REAL_t * zeta
    REAL_t * e_log_prob
    # T
    REAL_t * word_counts
    # T
    REAL_t * totals

    REAL_t * deriv
    # T * (T+1)
    REAL_t * mean_deriv_mtx
    int word

