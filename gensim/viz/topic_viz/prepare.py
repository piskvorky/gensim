from __future__ import absolute_import
from past.builtins import basestring
from collections import namedtuple
import json
import logging
from joblib import Parallel, delayed, cpu_count
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from .utils import NumPyEncoder
try:
	from sklearn.manifold import MDS, TSNE
	sklearn_present = True
except ImportError:
	sklearn_present = False



def _jensen_shannon(_P, _Q):
	_M = 0.5 * (_P + _Q)
	return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def _pcoa(pair_dists, n_components=2):
	"""Principal Coordinate Analysis,
	aka Classical Multidimensional Scaling
	"""
	# code referenced from skbio.stats.ordination.pcoa
	# https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

	# pairwise distance matrix is assumed symmetric
	pair_dists = np.asarray(pair_dists, np.float64)

	# perform SVD on double centred distance matrix
	n = pair_dists.shape[0]
	H = np.eye(n) - np.ones((n, n)) / n
	B = - H.dot(pair_dists ** 2).dot(H) / 2
	eigvals, eigvecs = np.linalg.eig(B)

	# Take first n_components of eigenvalues and eigenvectors
	# sorted in decreasing order
	ix = eigvals.argsort()[::-1][:n_components]
	eigvals = eigvals[ix]
	eigvecs = eigvecs[:, ix]

	# replace any remaining negative eigenvalues and associated eigenvectors with zeroes
	# at least 1 eigenvalue must be zero
	eigvals[np.isclose(eigvals, 0)] = 0
	if np.any(eigvals < 0):
		ix_neg = eigvals < 0
		eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
		eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

	return np.sqrt(eigvals) * eigvecs


def js_PCoA(distributions):
	"""Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis
	(aka Classical Multidimensional Scaling)
	Parameters
	----------
	distributions : array-like, shape (`n_dists`, `k`)
		Matrix of distributions probabilities.
	Returns
	-------
	pcoa : array, shape (`n_dists`, 2)
	"""
	dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
	return _pcoa(dist_matrix)

def js_MMDS(distributions, **kwargs):
	"""Dimension reduction via Jensen-Shannon Divergence & Metric Multidimensional Scaling
	Parameters
	----------
	distributions : array-like, shape (`n_dists`, `k`)
		Matrix of distributions probabilities.
	**kwargs : Keyword argument to be passed to `sklearn.manifold.MDS()`
	Returns
	-------
	mmds : array, shape (`n_dists`, 2)
	"""
	dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
	model = MDS(n_components=2, random_state=0, dissimilarity='precomputed', **kwargs)
	return model.fit_transform(dist_matrix)

def js_TSNE(distributions, **kwargs):
	"""Dimension reduction via Jensen-Shannon Divergence & t-distributed Stochastic Neighbor Embedding
	Parameters
	----------
	distributions : array-like, shape (`n_dists`, `k`)
		Matrix of distributions probabilities.
	**kwargs : Keyword argument to be passed to `sklearn.manifold.TSNE()`
	Returns
	-------
	tsne : array, shape (`n_dists`, 2)
	"""
	dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
	model = TSNE(n_components=2, random_state=0, metric='precomputed', **kwargs)
	return model.fit_transform(dist_matrix)


def _doc_coordinates(mds, doc_topic_dists, doc_tag, doc_texts):
	K = doc_topic_dists.shape[0]
	mds_res = mds(doc_topic_dists)
	assert mds_res.shape == (K, 2)
	mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'docs': doc_tag, 'doc_texts':doc_texts, 'Freq': [10]*K}, index=range(1, K+1))
	mds_df.reset_index(level=0, inplace=True)
	return mds_df

def _topic_coordinates(mds, topic_word_dists, topic_proportion):
	K = topic_word_dists.shape[0]
	mds_res = mds(topic_word_dists)
	assert mds_res.shape == (K, 2)
	mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'topics': range(1, K+1), 'Freq': topic_proportion * 100}, index=range(1, K+1))
	mds_df.reset_index(level=0, inplace=True)
	return mds_df

def _word_coordinates(mds, word_topic_dists, vocab, word_proportion):
	K = word_topic_dists.shape[0]
	mds_res = mds(word_topic_dists)
	assert mds_res.shape == (K, 2)
	mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'vocab': vocab, 'Freq': word_proportion * 100}, index=range(1, K+1))
	mds_df.reset_index(level=0, inplace=True)
	return mds_df