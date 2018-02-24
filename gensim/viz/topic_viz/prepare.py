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


def _info(dists, fst, scnd):
	dists = dists / dists.sum()
	dists = np.array(dists)
	pd_data = pd.DataFrame(dists, index=range(1, dists.shape[0]+1), columns=range(1, dists.shape[1]+1))
	pd_data = pd_data.stack().reset_index().rename(columns={'level_0':fst,'level_1':scnd, 0:'Freq'})

	return pd_data


def prepare(doc_topic_dists, doc_word_dists, topic_word_dists, word_topic_dists, 
			doc_tag, doc_texts, doc_lengths, vocab, mds=js_PCoA):
	"""Transforms the topic model distributions and related corpus data into
	the data structures needed for the visualization.
	"""
	# parse mds
	if isinstance(mds, basestring):
		mds = mds.lower()
		if mds == 'pcoa':
			mds = js_PCoA
		elif mds in ('mmds', 'tsne'):
			if sklearn_present:
				mds_opts = {'mmds': js_MMDS, 'tsne': js_TSNE}
				mds = mds_opts[mds]
			else:
				logging.warning('sklearn not present, switch to PCoA')
				mds = js_PCoA
		else:
			logging.warning('Unknown mds `%s`, switch to PCoA' % mds)
			mds = js_PCoA

	topic_freq = np.dot(doc_topic_dists.T, doc_lengths)
	topic_proportion = topic_freq / topic_freq.sum()

	word_freq = np.dot(topic_word_dists.T, topic_freq)
	word_proportion = word_freq / word_freq.sum()

	word_doc_dists = doc_word_dists.T
	topic_doc_dists = doc_topic_dists.T

	doc_topic_info = _info(doc_topic_dists, 'Doc', 'Topic')
	doc_word_info = _info(doc_word_dists, 'Doc', 'Word')
	topic_doc_info = _info(topic_doc_dists, 'Topic', 'Doc')
	topic_word_info = _info(topic_word_dists, 'Topic', 'Word')
	word_doc_info = _info(word_doc_dists, 'Word', 'Doc')
	word_topic_info = _info(word_topic_dists, 'Word', 'Topic')

	doc_coordinates = _doc_coordinates(mds, doc_topic_dists, doc_tag, doc_texts)
	topic_coordinates = _topic_coordinates(mds, topic_word_dists, topic_proportion)
	word_coordinates = _word_coordinates(mds, word_topic_dists, vocab, word_proportion)

	return PreparedData(doc_coordinates, topic_coordinates, word_coordinates, doc_topic_info, doc_word_info, topic_doc_info, topic_word_info, word_doc_info, word_topic_info)


class PreparedData(namedtuple('PreparedData', ['doc_coordinates', 'topic_coordinates', 'word_coordinates', 'doc_topic_info', 'doc_word_info', 
											   'topic_doc_info', 'topic_word_info', 'word_doc_info', 'word_topic_info'])):
	def to_dict(self):
		return {'doc_mds': self.doc_coordinates.to_dict(orient='list'),
			   'topic_mds': self.topic_coordinates.to_dict(orient='list'),
			   'word_mds': self.word_coordinates.to_dict(orient='list'),
			   'doc_topic.info': self.doc_topic_info.to_dict(orient='list'),
			   'doc_word.info': self.doc_word_info.to_dict(orient='list'),
			   'topic_doc.info': self.topic_doc_info.to_dict(orient='list'),
			   'topic_word.info': self.topic_word_info.to_dict(orient='list'),
			   'word_doc.info': self.word_doc_info.to_dict(orient='list'),
			   'word_topic.info': self.word_topic_info.to_dict(orient='list')
			   }

	def to_json(self):
		return json.dumps(self.to_dict(), cls=NumPyEncoder)	