import numpy as np

from scipy.special.cython_special cimport gammaln
from scipy.special.cython_special cimport psi

from libc.math cimport exp, log, abs
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

cimport numpy as np


ctypedef np.float64_t REAL_t


cdef update_phi(
        REAL_t * gamma, REAL_t *phi, REAL_t * log_phi,
        int * word_ids, REAL_t * lda_topics, const int num_topics,
        const int doc_length
     ):

    """Update variational multinomial parameters, based on a document and a time-slice.

    This is done based on the original Blei-LDA paper, where:
    log_phi := beta * exp(Ψ(gamma)), over every topic for every word.

    TODO: incorporate lee-sueng trick used in
    **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

    Parameters
    ----------
    gamma : size of num_topics. in-parameter

    phi : size of (max_doc_len, num_topics). int/out-parameter

    log_phi: size of (max_doc_len, num_topics). in/out-parameter

    word_ids: size of doc_length. in-parameter

    lda_topics: size of (vocab_len, num_topics). in-parameter

    num_topics: number of topics in the model. in-parameter

    doc_length: length of the document (number of words in the document). in-parameter

    """

    cdef int k, i

    # digamma values
    cdef REAL_t * dig = <REAL_t *> malloc(num_topics * sizeof(REAL_t))
    if dig == NULL:
        raise

    for k in range(num_topics):
        dig[k] = psi(gamma[k])

    cdef REAL_t *log_phi_row = NULL
    cdef REAL_t *phi_row = NULL

    for i in range(doc_length):
        for k in range(num_topics):
            log_phi[i * num_topics + k] = dig[k] + lda_topics[word_ids[i] * num_topics + k]

        log_phi_row = log_phi + i * num_topics

        phi_row = phi + i * num_topics

        # log normalize
        v = log_phi_row[0]
        for i in range(1, num_topics):
            v = log(exp(v) + exp(log_phi_row[i]))

        # subtract every element by v
        for i in range(num_topics):
            log_phi_row[i] = log_phi_row[i] - v

        for i in range(num_topics):
            phi_row[i] = exp(log_phi_row[i])

    free(dig)


cdef update_phi_fixed():
    return


cdef update_gamma(
        REAL_t * gamma, const REAL_t * phi, const REAL_t *lda_alpha,
        const int *word_counts, const int num_topics, const int doc_length
     ):
    """Update variational dirichlet parameters.

    This operations is described in the original Blei LDA paper:
    gamma = alpha + sum(phi), over every topic for every word.

    Parameters
    ----------
    gamma: size of num_topics. 
    
    phi: size of (max_doc_len, num_topics). 
    
    lda_alpha: size of num_topics. 
    
    word_counts: size of doc_length. 
    
    num_topics: number of topics in the model 
    
    doc_length: length of the document

    """
    memcpy(gamma, lda_alpha, num_topics * sizeof(REAL_t))

    cdef int i, k
    # TODO BLAS matrix*vector
    for i in range(doc_length):
        for k in range(num_topics):
            gamma[k] += phi[i * num_topics + k] * word_counts[i]


cdef REAL_t compute_lda_lhood(
        REAL_t * lhood, const REAL_t *gamma, const REAL_t * phi, const REAL_t *log_phi,
        const REAL_t * lda_alpha, const REAL_t *lda_topics,
        int *word_counts, int *word_ids,
        const int num_topics, const int doc_length
     ):
    """Compute the log likelihood bound.
    Parameters
    ----------
    gamma: size of num_topics
    
    lhood: size of num_topics + 1
    
    phi: size of (max_doc_len, num_topics).
    
    log_phi: size of (max_doc_len, num_topics). in-parameter
    
    lda_alpha: size of num_topics. in-parameter

    lda_topics: size of (vocab_len, num_topics). in-parameter
    
    word_counts: size of doc_len
    
    word_ids: size of doc_len 
    
    num_topics: number of topics in the model 
    
    doc_length: length of the document 

    Returns
    -------
    float
        The optimal lower bound for the true posterior using the approximate distribution.

    """
    cdef int i

    cdef REAL_t gamma_sum = 0.0
    for i in range(num_topics):
        gamma_sum += gamma[i]

    cdef REAL_t alpha_sum = 0.0
    for i in range(num_topics):
        alpha_sum += lda_alpha[i]

    cdef REAL_t lhood_v = gammaln(alpha_sum) - gammaln(gamma_sum)
    lhood[num_topics] = lhood_v

    cdef REAL_t digsum = psi(gamma_sum)
    cdef REAL_t lhood_term, e_log_theta_k

    for k in range(num_topics):

        e_log_theta_k = psi(gamma[k]) - digsum

        lhood_term = (lda_alpha[k] - gamma[k]) * e_log_theta_k + gammaln(gamma[k]) - gammaln(lda_alpha[k])

        # TODO: check why there's an IF
        for i in range(doc_length):
            if phi[i * num_topics + k] > 0:
                lhood_term += \
                    word_counts[i] * phi[i * num_topics + k] \
                    *  (e_log_theta_k + lda_topics[word_ids[i] * num_topics + k]
                        - log_phi[i * num_topics + k])

        lhood[k] = lhood_term
        lhood_v += lhood_term

    return lhood_v

cdef init_lda_post(
        REAL_t *gamma, REAL_t * phi, const int *word_counts,
        const REAL_t *lda_alpha,
        const int doc_length, const int num_topics
     ):

    """Initialize variational posterior. """

    cdef int i, j

    cdef int total = 0

    # BLAS sum of absolute numbers
    for i in range(doc_length):
        total += word_counts[i]

    cdef REAL_t init_value = lda_alpha[0] + float(total) / num_topics

    for i in range(num_topics):
        gamma[i] = init_value

    init_value = 1.0 / num_topics

    for i in range(doc_length):
        phi_doc = phi + i * num_topics
        for j in range(num_topics):
            phi_doc[j] = init_value


def fit_lda_post(
        self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-8,
        lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None
    ):
    """Posterior inference for lda.

    Parameters
    ----------

    Returns
    -------
    float
        The optimal lower bound for the true posterior using the approximate distribution.
    """

    cdef int i

    ###############
    # Setup C structures
    ###############

    cdef int num_topics = self.lda.num_topics
    cdef int vocab_len = len(self.lda.id2word)

    cdef int doc_length = len(self.doc)
    cdef int max_doc_len = doc_length

    # TODO adopt implementation to avoid memory allocation for every document.
    #  E.g. create numpy array field of Python class. Be careful with the array size, it should be at least
    #  the length of the longest document
    cdef int * word_ids = <int *> malloc(doc_length * sizeof(int))
    if word_ids == NULL:
        raise
    cdef int * word_counts = <int *> malloc(doc_length * sizeof(int))
    if word_counts == NULL:
        raise
    # TODO Would it be better to create numpy array first?
    for i in range(doc_length):
        word_ids[i] = self.doc[i][0]
        word_counts[i] = self.doc[i][1]

    cdef REAL_t * gamma = <REAL_t *> (np.PyArray_DATA(self.gamma))
    cdef REAL_t * phi = <REAL_t *> (np.PyArray_DATA(self.phi))
    cdef REAL_t * log_phi = <REAL_t *> (np.PyArray_DATA(self.log_phi))
    cdef REAL_t * lhood  = <REAL_t *> (np.PyArray_DATA(self.lhood))

    cdef REAL_t * lda_topics = <REAL_t *> (np.PyArray_DATA(self.lda.topics))
    cdef REAL_t * lda_alpha = <REAL_t *> (np.PyArray_DATA(self.lda.alpha))

    ###############
    # Finished setup of c structures here
    ###############

    init_lda_post(gamma, phi, word_counts, lda_alpha, doc_length, num_topics)

    # sum of counts in a doc
    cdef REAL_t total = sum(count for word_id, count in self.doc)

    cdef REAL_t lhood_v = compute_lda_lhood(
        lhood, gamma, phi, log_phi,
        lda_alpha, lda_topics, word_counts, word_ids,
        num_topics, doc_length
    )

    cdef REAL_t lhood_old = 0.0
    cdef REAL_t converged = 0.0
    cdef int iter_ = 0

    # TODO Why first iteration starts here is done outside of the loop?
    iter_ += 1
    lhood_old = lhood_v
    update_gamma(gamma, phi, lda_alpha, word_counts, num_topics, doc_length)

    model = "DTM"

    # if model == "DTM" or sslm is None:
    update_phi(gamma, phi, log_phi, word_ids, lda_topics, num_topics, doc_length)

    lhood_v = compute_lda_lhood(
        lhood, gamma, phi, log_phi,
        lda_alpha, lda_topics, word_counts, word_ids,
        num_topics, doc_length
    )

    converged = abs((lhood_old - lhood_v) / (lhood_old * total))

    while converged > LDA_INFERENCE_CONVERGED and iter_ <= lda_inference_max_iter:

        iter_ += 1
        lhood_old = lhood_v
        update_gamma(gamma, phi, lda_alpha, word_counts, num_topics, doc_length)
        model = "DTM"

        update_phi(gamma, phi, log_phi, word_ids, lda_topics, num_topics, doc_length)

        lhood_v = compute_lda_lhood(
            lhood, gamma, phi, log_phi,
            lda_alpha, lda_topics, word_counts, word_ids,
            num_topics, doc_length
        )

        converged = np.fabs((lhood_old - lhood_v) / (lhood_old * total))

    free(word_ids)
    free(word_counts)

    return lhood_v
