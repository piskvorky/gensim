r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# The purpose of this tutorial is to demonstrate how to train and tune an LDA model.
#
# In this tutorial we will:
#
# * Load input data.
# * Pre-process that data.
# * Transform documents into bag-of-words vectors.
# * Train an LDA model.
#
# This tutorial will **not**:
#
# * Explain how Latent Dirichlet Allocation works
# * Explain how the LDA model performs inference
# * Teach you all the parameters and options for Gensim's LDA implementation
#
# If you are not familiar with the LDA model or how to use it in Gensim, I (Olavur Mortensen)
# suggest you read up on that before continuing with this tutorial. Basic
# understanding of the LDA model should suffice. Examples:
#
# * `Introduction to Latent Dirichlet Allocation <http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation>`_
# * Gensim tutorial: :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`
# * Gensim's LDA model API docs: :py:class:`gensim.models.LdaModel`
#
# I would also encourage you to consider each step when applying the model to
# your data, instead of just blindly applying my solution. The different steps
# will depend on your data and possibly your goal with the model.
#
# Data
# ----
#
# I have used a corpus of NIPS papers in this tutorial, but if you're following
# this tutorial just to learn about LDA I encourage you to consider picking a
# corpus on a subject that you are familiar with. Qualitatively evaluating the
# output of an LDA model is challenging and can require you to understand the
# subject matter of your corpus (depending on your goal with the model).
#
# NIPS (Neural Information Processing Systems) is a machine learning conference
# so the subject matter should be well suited for most of the target audience
# of this tutorial.  You can download the original data from Sam Roweis'
# `website <http://www.cs.nyu.edu/~roweis/data.html>`_.  The code below will
# also do that for you.
#
# .. Important::
#     The corpus contains 1740 documents, and not particularly long ones.
#     So keep in mind that this tutorial is not geared towards efficiency, and be
#     careful before applying the code to a large dataset.
#

import io
import os.path
import re
import tarfile

import smart_open

def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')

docs = list(extract_documents())

###############################################################################
# So we have a list of 1740 documents, where each document is a Unicode string.
# If you're thinking about using your own corpus, then you need to make sure
# that it's in the same format (list of Unicode strings) before proceeding
# with the rest of this tutorial.
#
print(len(docs))
print(docs[0][:500])

###############################################################################
# Pre-process and vectorize the documents
# ---------------------------------------
#
# As part of preprocessing, we will:
#
# * Tokenize (split the documents into tokens).
# * Lemmatize the tokens.
# * Compute bigrams.
# * Compute a bag-of-words representation of the data.
#
# First we tokenize the text using a regular expression tokenizer from NLTK. We
# remove numeric tokens and tokens that are only a single character, as they
# don't tend to be useful, and the dataset contains a lot of them.
#
# .. Important::
#
#    This tutorial uses the nltk library for preprocessing, although you can
#    replace it with something else if you want.
#

# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

###############################################################################
# We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
# stemmer in this case because it produces more readable words. Output that is
# easy to read is very desirable in topic modelling.
#

# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

###############################################################################
# We find bigrams in the documents. Bigrams are sets of two adjacent words.
# Using bigrams we can get phrases like "machine_learning" in our output
# (spaces are replaced with underscores); without bigrams we would only get
# "machine" and "learning".
#
# Note that in the code below, we find bigrams and then add them to the
# original data, because we would like to keep the words "machine" and
# "learning" as well as the bigram "machine_learning".
#
# .. Important::
#     Computing n-grams of large dataset can be very computationally
#     and memory intensive.
#


# Compute bigrams.
from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

###############################################################################
# We remove rare words and common words based on their *document frequency*.
# Below we remove words that appear in less than 20 documents or in more than
# 50% of the documents. Consider trying to remove words only based on their
# frequency, or maybe combining that with this approach.
#

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

###############################################################################
# Finally, we transform the documents to a vectorized form. We simply compute
# the frequency of each word, including the bigrams.
#

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

###############################################################################
# Let's see how many tokens and documents we have to train on.
#

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

###############################################################################
# Training
# --------
#
# We are ready to train the LDA model. We will first discuss how to set some of
# the training parameters.
#
# First of all, the elephant in the room: how many topics do I need? There is
# really no easy answer for this, it will depend on both your data and your
# application. I have used 10 topics here because I wanted to have a few topics
# that I could interpret and "label", and because that turned out to give me
# reasonably good results. You might not need to interpret all your topics, so
# you could use a large number of topics, for example 100.
#
# ``chunksize`` controls how many documents are processed at a time in the
# training algorithm. Increasing chunksize will speed up training, at least as
# long as the chunk of documents easily fit into memory. I've set ``chunksize =
# 2000``, which is more than the amount of documents, so I process all the
# data in one go. Chunksize can however influence the quality of the model, as
# discussed in Hoffman and co-authors [2], but the difference was not
# substantial in this case.
#
# ``passes`` controls how often we train the model on the entire corpus.
# Another word for passes might be "epochs". ``iterations`` is somewhat
# technical, but essentially it controls how often we repeat a particular loop
# over each document. It is important to set the number of "passes" and
# "iterations" high enough.
#
# I suggest the following way to choose iterations and passes. First, enable
# logging (as described in many Gensim tutorials), and set ``eval_every = 1``
# in ``LdaModel``. When training the model look for a line in the log that
# looks something like this::
#
#    2016-06-21 15:40:06,753 - gensim.models.ldamodel - DEBUG - 68/1566 documents converged within 400 iterations
#
# If you set ``passes = 20`` you will see this line 20 times. Make sure that by
# the final passes, most of the documents have converged. So you want to choose
# both passes and iterations to be high enough for this to happen.
#
# We set ``alpha = 'auto'`` and ``eta = 'auto'``. Again this is somewhat
# technical, but essentially we are automatically learning two parameters in
# the model that we usually would have to specify explicitly.
#


# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

###############################################################################
# We can compute the topic coherence of each topic. Below we display the
# average topic coherence and print the topics in order of topic coherence.
#
# Note that we use the "Umass" topic coherence measure here (see
# :py:func:`gensim.models.ldamodel.LdaModel.top_topics`), Gensim has recently
# obtained an implementation of the "AKSW" topic coherence measure (see
# accompanying blog post, http://rare-technologies.com/what-is-topic-coherence/).
#
# If you are familiar with the subject of the articles in this dataset, you can
# see that the topics below make a lot of sense. However, they are not without
# flaws. We can see that there is substantial overlap between some topics,
# others are hard to interpret, and most of them have at least some terms that
# seem out of place. If you were able to do better, feel free to share your
# methods on the blog at http://rare-technologies.com/lda-training-tips/ !
#

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

###############################################################################
# Things to experiment with
# -------------------------
#
# * ``no_above`` and ``no_below`` parameters in ``filter_extremes`` method.
# * Adding trigrams or even higher order n-grams.
# * Consider whether using a hold-out set or cross-validation is the way to go for you.
# * Try other datasets.
#
# Where to go from here
# ---------------------
#
# * Check out a RaRe blog post on the AKSW topic coherence measure (http://rare-technologies.com/what-is-topic-coherence/).
# * pyLDAvis (https://pyldavis.readthedocs.io/en/latest/index.html).
# * Read some more Gensim tutorials (https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials).
# * If you haven't already, read [1] and [2] (see references).
#
# References
# ----------
#
# 1. "Latent Dirichlet Allocation", Blei et al. 2003.
# 2. "Online Learning for Latent Dirichlet Allocation", Hoffman et al. 2010.
#
