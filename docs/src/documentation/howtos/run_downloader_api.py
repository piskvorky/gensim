r"""
How to download pre-trained models and corpora
==============================================

Demonstrates simple and quick access to common corpora, models, and other data.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# One of Gensim's features is simple and easy access to some common data.
# The `gensim-data <https://github.com/RaRe-Technologies/gensim-data>`_ project stores a variety of corpora, models and other data.
# Gensim has a :py:mod:`gensim.downloader` module for programmatically accessing this data.
# The module leverages a local cache that ensures data is downloaded at most once.
#
# This tutorial:
#
# * Retrieves the text8 corpus, unless it is already on your local machine
# * Trains a Word2Vec model from the corpus (see :ref:`sphx_glr_auto_examples_tutorials_run_doc2vec_lee.py` for a detailed tutorial)
# * Leverages the model to calculate word similarity
# * Demonstrates using the API to load other models and corpora
#
# Let's start by importing the api module.
#
import gensim.downloader as api

###############################################################################
#
# Now, lets download the text8 corpus and load it to memory (automatically)
#
corpus = api.load('text8')

###############################################################################
# In this case, corpus is an iterable.
# If you look under the covers, it has the following definition:

import inspect
print(inspect.getsource(corpus.__class__))

###############################################################################
# For more details, look inside the file that defines the Dataset class for your particular resource.
#
print(inspect.getfile(corpus.__class__))


###############################################################################
#
# As the corpus has been downloaded and loaded, let's create a word2vec model of our corpus.
#

from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus)

###############################################################################
#
# Now that we have our word2vec model, let's find words that are similar to 'tree'
#


print(model.most_similar('tree'))

###############################################################################
#
# You can use the API to download many corpora and models. You can get the list of all the models and corpora that are provided, by using the code below:
#


import json
info = api.info()
print(json.dumps(info, indent=4))

###############################################################################
# There are two types of data: corpora and models.
print(info.keys())

###############################################################################
# Let's have a look at the available corpora:
for corpus_name, corpus_data in sorted(info['corpora'].items()):
    print(
        '%s (%d records): %s' % (
            corpus_name,
            corpus_data.get('num_records', -1),
            corpus_data['description'][:40] + '...',
        )
    )

###############################################################################
# ... and the same for models:
for model_name, model_data in sorted(info['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )

###############################################################################
#
# If you want to get detailed information about the model/corpus, use:
#


fake_news_info = api.info('fake-news')
print(json.dumps(fake_news_info, indent=4))

###############################################################################
#
# Sometimes, you do not want to load the model to memory. You would just want to get the path to the model. For that, use :
#


print(api.load('glove-wiki-gigaword-50', return_path=True))

###############################################################################
#
# If you want to load the model to memory, then:
#


model = api.load("glove-wiki-gigaword-50")
model.most_similar("glass")

###############################################################################
#
# In corpora, the corpus is never loaded to memory, all corpuses wrapped to special class ``Dataset`` and provide ``__iter__`` method
#


