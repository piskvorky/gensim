r"""
Ensemble LDA
============

Introduces Gensim's EnsembleLda model

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# This tutorial will explain how to use the EnsembleLDA model class.
#
# EnsembleLda is a method of finding and generating stable topics from the results of multiple topic models,
# it can be used to remove topics from your results that are noise and are not reproducible.
#

###############################################################################
# Corpus
# ------
# We will use the gensim downloader api to get a small corpus for training our ensemble.
#
# The preprocessing is similar to :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py`,
# so it won't be explained again in detail.
#

import gensim.downloader as api
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = api.load('text8')

dictionary = Dictionary()
for doc in docs:
    dictionary.add_documents([[lemmatizer.lemmatize(token) for token in doc]])
dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in docs]

###############################################################################
# Training
# --------
#
# Training the ensemble works very similar to training a single model,
#
# You can use any model that is based on LdaModel, such as LdaMulticore, to train the Ensemble.
# In experiments, LdaMulticore showed better results.
#

from gensim.models import LdaModel
topic_model_class = LdaModel

###############################################################################
# Any arbitrary number of models can be used, but it should be a multiple of your workers so that the
# load can be distributed properly. In this example, 4 processes will train 8 models each.
#

ensemble_workers = 4
num_models = 8

###############################################################################
# After training all the models, some distance computations are required which can take quite some
# time as well. You can speed this up by using workers for that as well.
#

distance_workers = 4

###############################################################################
# All other parameters that are unknown to EnsembleLda are forwarded to each LDA Model, such as
#
num_topics = 20
passes = 2

###############################################################################
# Now start the training
#
# Since 20 topics were trained on each of the 8 models, we expect there to be 160 different topics.
# The number of stable topics which are clustered from all those topics is smaller.
#

from gensim.models import EnsembleLda
ensemble = EnsembleLda(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=passes,
    num_models=num_models,
    topic_model_class=LdaModel,
    ensemble_workers=ensemble_workers,
    distance_workers=distance_workers
)

print(len(ensemble.ttda))
print(len(ensemble.get_topics()))

###############################################################################
# Tuning
# ------
#
# Different from LdaModel, the number of resulting topics varies greatly depending on the clustering parameters.
#
# You can provide those in the ``recluster()`` function or the ``EnsembleLda`` constructor.
#
# Play around until you get as many topics as you desire, which however may reduce their quality.
# If your ensemble doesn't have enough topics to begin with, you should make sure to make it large enough.
#
# Having an epsilon that is smaller than the smallest distance doesn't make sense.
# Make sure to chose one that is within the range of values in ``asymmetric_distance_matrix``.
#

import numpy as np
shape = ensemble.asymmetric_distance_matrix.shape
without_diagonal = ensemble.asymmetric_distance_matrix[~np.eye(shape[0], dtype=bool)].reshape(shape[0], -1)
print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())

ensemble.recluster(eps=0.09, min_samples=2, min_cores=2)

print(len(ensemble.get_topics()))

###############################################################################
# Increasing the Size
# -------------------
#
# If you have some models lying around that were trained on a corpus based on the same dictionary,
# they are compatible and you can add them to the ensemble.
#
# By setting num_models of the EnsembleLda constructor to 0 you can also create an ensemble that is
# entirely made out of your existing topic models with the following method.
#
# Afterwards the number and quality of stable topics might be different depending on your added topics and parameters.
#

from gensim.models import LdaMulticore

model1 = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=9,
    passes=4,
)

model2 = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=11,
    passes=2,
)

# add_model supports various types of input, check out its docstring
ensemble.add_model(model1)
ensemble.add_model(model2)

ensemble.recluster()

print(len(ensemble.ttda))
print(len(ensemble.get_topics()))
