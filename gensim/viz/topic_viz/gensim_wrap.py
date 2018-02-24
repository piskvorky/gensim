"""
LDAvis Gensim
===============
Helper functions to visualize LDA models trained by Gensim
"""

from __future__ import absolute_import
import funcy as fp
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from past.builtins import xrange
from . import prepare as vis_prepare


def _extract_data(topic_model, corpus, dictionary, texts=None, doc_topic_dists=None):
   import gensim

   if not gensim.matutils.ismatrix(corpus):
      corpus_csc = gensim.matutils.corpus2csc(corpus, num_terms=len(dictionary))
   else:
      corpus_csc = corpus
      # Need corpus to be a streaming gensim list corpus for len and inference functions below:
      corpus = gensim.matutils.Sparse2Corpus(corpus_csc)

   doc_word_dists = corpus_csc.todense().T
   doc_word_dists = doc_word_dists / doc_word_dists.sum(axis=1)

   vocab = [token for token, token_id in sorted(dictionary.token2id.items(), key=lambda value: value[1])]

   # TODO: add the hyperparam to smooth it out? no beta in online LDA impl.. hmm..
   # for now, I'll just make sure we don't ever get zeros...
   beta = 0.01
   fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
   term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
   term_freqs[term_freqs == 0] = beta
   doc_lengths = corpus_csc.sum(axis=0).A.ravel()

   assert term_freqs.shape[0] == len(dictionary), 'Term frequencies and dictionary have different shape {} != {}'.format(term_freqs.shape[0], len(dictionary))
   assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(corpus))

   if hasattr(topic_model, 'lda_alpha'):
       num_topics = len(topic_model.lda_alpha)
   else:
       num_topics = topic_model.num_topics

   if doc_topic_dists is None:
      # If its an HDP model.
      if hasattr(topic_model, 'lda_beta'):
          gamma = topic_model.inference(corpus)
      else:
          gamma, _ = topic_model.inference(corpus)
      doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
   else:
      if isinstance(doc_topic_dists, list):
         doc_topic_dists = gensim.matutils.corpus2dense(doc_topic_dists, num_topics).T
      elif issparse(doc_topic_dists):
         doc_topic_dists = doc_topic_dists.T.todense()
      doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)

   assert doc_topic_dists.shape[1] == num_topics, 'Document topics and number of topics do not match {} != {}'.format(doc_topic_dists.shape[1], num_topics)

   term_topic_dists = []
   for word_id in list(dictionary.token2id.values()):
    values = []
    for topic_id in range(0, num_topics):
      values.append(topic_model.expElogbeta[topic_id][word_id])
    term_topic_dists.append(values)
   term_topic_dists = np.array(term_topic_dists)
   # term_topic_dists = term_topic_dists / term_topic_dists.sum(axis=1)

   # get the topic-term distribution straight from gensim without
   # iterating over tuples
   if hasattr(topic_model, 'lda_beta'):
       topic = topic_model.lda_beta
   else:
       topic = topic_model.state.get_lambda()
   topic = topic / topic.sum(axis=1)[:, None]
   topic_term_dists = topic[:, fnames_argsort]

   assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]

   # convert tokenised texts of documents to list of strings
   texts = [' '.join(doc) for doc in texts]

   return {'doc_topic_dists': doc_topic_dists, 'doc_word_dists': doc_word_dists, 'topic_word_dists': topic_term_dists,
           'word_topic_dists': term_topic_dists, 'doc_tag': range(1, doc_topic_dists.shape[0]+1), 'doc_texts': texts,
           'doc_lengths': doc_lengths, 'vocab': vocab}

def prepare(topic_model, corpus, dictionary, texts=None, doc_topic_dist=None, **kwargs):
    """Transforms the Gensim TopicModel and related corpus and dictionary into
    the data structures needed for the visualization.

    Parameters
    ----------
    topic_model : gensim.models.ldamodel.LdaModel
        An already trained Gensim LdaModel. The other gensim model types are
        not supported (PRs welcome).

    corpus : array-like list of bag of word docs in tuple form or scipy CSC matrix
        The corpus in bag of word form, the same docs used to train the model.
        The corpus is transformed into a csc matrix internally, if you intend to
        call prepare multiple times it is a good idea to first call
        `gensim.matutils.corpus2csc(corpus)` and pass in the csc matrix instead.

    For example: [(50, 3), (63, 5), ....]

    dictionary: gensim.corpora.Dictionary
        The dictionary object used to create the corpus. Needed to extract the
        actual terms (not ids).

    doc_topic_dist (optional): Document topic distribution from LDA (default=None)
        The document topic distribution that is eventually visualised, if you will
        be calling `prepare` multiple times it's a good idea to explicitly pass in
        `doc_topic_dist` as inferring this for large corpora can be quite
        expensive.

    **kwargs :
        additional keyword arguments are passed through to :func:`pyldavis.prepare`.

    Returns
    -------
    prepared_data : PreparedData
        the data structures used in the visualization

    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb

    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    opts = fp.merge(_extract_data(topic_model, corpus, dictionary, texts, doc_topic_dist), kwargs)
    return vis_prepare(**opts)

