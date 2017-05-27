# -*- coding: utf-8 -*-
"""
Univariate features selection.

The functions and classes are used for feature selection in the cosine ESA model.
They are forked from Scikit-learn.

Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay.
         L. Buitinck
License: BSD 3 clause

Modified by K. Jeschkies to work on iterables and have constant memory usage
"""


from abc import ABCMeta, abstractmethod

from collections import defaultdict
import logging
import numpy as np
from scipy import stats

from sklearn.feature_selection import SelectKBest
from sklearn.utils import array2d, atleast2d_or_csr, deprecated, \
        check_arrays, safe_asarray, safe_sqr, safe_mask
from sklearn.utils.extmath import safe_sparse_dot

logger = logging.getLogger('gensim.models.esamodel')

######################################################################
# Scoring functions


# The following function is a rewriting of sklearn.univariate_selection.f_oneway
# The memory usage is independent of samples and only depends on the number of
# features.
def if_classif(X_y, n_features):
    """Compute the Anova F-value for the provided sample

    Parameters
    ----------
    X_y : Tuples of (X, y) with 
          X {array-like, sparse matrix} shape = [n_samples, n_features]
          The set of regressors that will tested sequentially
          y array of shape(n_samples)
          The data matrix

    Returns
    -------
    F : array, shape = [n_features,]
        The set of F values
    pval : array, shape = [n_features,]
        The set of p-values
    """
    
    n_samples = 0
    n_samples_per_class = defaultdict(lambda: 0)
    
    sums_args_d = defaultdict(lambda: np.zeros(shape=(n_features))) 
    ss_alldata = np.zeros(shape=(n_features))
    
    for X, y in X_y:
        if(n_samples % 100) == 0:
            logger.info("Processing doc #%d..." % n_samples)
            
        n_samples += 1
        n_samples_per_class[y] += 1
        
        ss_alldata[:] += X[:]**2
        sums_args_d[y][:] += X[:]
        
    n_classes = len(sums_args_d.keys())
    
    #Convert dictionary to numpy array
    sums_args = np.array(list(row for row in sums_args_d.itervalues()))
    
    square_of_sums_alldata = safe_sqr(reduce(lambda x, y: x + y, sums_args))
    square_of_sums_args = [safe_sqr(s) for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.
    for k, y in enumerate(n_samples_per_class.keys()):
        ssbn += square_of_sums_args[k] / n_samples_per_class[y]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = stats.fprob(dfbn, dfwn, f)
    return f, prob


######################################################################
# Specific filters
######################################################################


class iSelectKBest(SelectKBest):
    """Filter: Select the k lowest p-values.

       Modified to have constant memory usage.

    Parameters
    ----------
    score_func: callable
        Function taking two arrays X and y, and returning 2 arrays:
        both scores and pvalues

    k: int, optional
        Number of top features to select.

    Notes
    -----
    Ties between features with equal p-values will be broken in an unspecified
    way.

    """
        
    def fit(self, X_y, n_features):
        """
        Evaluate the function
        
        Parameters
        ==========
        X_y: iterable over tuples (X, y) 
        n_features: number of features. Length of X
        """
        scores = self.score_func(X_y, n_features)
        self.scores_ = scores[0]
        self.pvalues_ = scores[1]
        return self

    def _get_support_mask(self):
        k = self.k
        if k > len(self.pvalues_):
            raise ValueError("cannot select %d features among %d"
                             % (k, len(self.pvalues_)))

        # XXX This should be refactored; we're getting an array of indices
        # from argsort, which we transform to a mask, which we probably
        # transform back to indices later.
        mask = np.zeros(self.pvalues_.shape, dtype=bool)
        mask[np.argsort(self.pvalues_)[:k]] = 1
        return mask

