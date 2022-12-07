r"""
Doc2Vec Model
=============

Introduces Gensim's Doc2Vec model and demonstrates its use on the
`Lee Corpus <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`__.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Doc2Vec is a :ref:`core_concepts_model` that represents each
# :ref:`core_concepts_document` as a :ref:`core_concepts_vector`.  This
# tutorial introduces the model and demonstrates how to train and assess it.
#
# Here's a list of what we'll be doing:
#
# 0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
# 1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
# 2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
# 3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
# 4. Assess the model
# 5. Test the model on the test corpus
#
# Review: Bag-of-words
# --------------------
#
# .. Note:: Feel free to skip these review sections if you're already familiar with the models.
#
# You may be familiar with the `bag-of-words model
# <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ from the
# :ref:`core_concepts_vector` section.
# This model transforms each document to a fixed-length vector of integers.
# For example, given the sentences:
#
# - ``John likes to watch movies. Mary likes movies too.``
# - ``John also likes to watch football games. Mary hates football.``
#
# The model outputs the vectors:
#
# - ``[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]``
# - ``[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]``
#
# Each vector has 10 elements, where each element counts the number of times a
# particular word occurred in the document.
# The order of elements is arbitrary.
# In the example above, the order of the elements corresponds to the words:
# ``["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]``.
#
# Bag-of-words models are surprisingly effective, but have several weaknesses.
#
# First, they lose all information about word order: "John likes Mary" and
# "Mary likes John" correspond to identical vectors. There is a solution: bag
# of `n-grams <https://en.wikipedia.org/wiki/N-gram>`__
# models consider word phrases of length n to represent documents as
# fixed-length vectors to capture local word order but suffer from data
# sparsity and high dimensionality.
#
# Second, the model does not attempt to learn the meaning of the underlying
# words, and as a consequence, the distance between vectors doesn't always
# reflect the difference in meaning.  The ``Word2Vec`` model addresses this
# second problem.
#
# Review: ``Word2Vec`` Model
# --------------------------
#
# ``Word2Vec`` is a more recent model that embeds words in a lower-dimensional
# vector space using a shallow neural network. The result is a set of
# word-vectors where vectors close together in vector space have similar
# meanings based on context, and word-vectors distant to each other have
# differing meanings. For example, ``strong`` and ``powerful`` would be close
# together and ``strong`` and ``Paris`` would be relatively far.
#
# Gensim's :py:class:`~gensim.models.word2vec.Word2Vec` class implements this model.
#
# With the ``Word2Vec`` model, we can calculate the vectors for each **word** in a document.
# But what if we want to calculate a vector for the **entire document**\ ?
# We could average the vectors for each word in the document - while this is quick and crude, it can often be useful.
# However, there is a better way...
#
# Introducing: Paragraph Vector
# -----------------------------
#
# .. Important:: In Gensim, we refer to the Paragraph Vector model as ``Doc2Vec``.
#
# Le and Mikolov in 2014 introduced the `Doc2Vec algorithm <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__,
# which usually outperforms such simple-averaging of ``Word2Vec`` vectors.
#
# The basic idea is: act as if a document has another floating word-like
# vector, which contributes to all training predictions, and is updated like
# other word-vectors, but we will call it a doc-vector. Gensim's
# :py:class:`~gensim.models.doc2vec.Doc2Vec` class implements this algorithm.
#
# There are two implementations:
#
# 1. Paragraph Vector - Distributed Memory (PV-DM)
# 2. Paragraph Vector - Distributed Bag of Words (PV-DBOW)
#
# .. Important::
#   Don't let the implementation details below scare you.
#   They're advanced material: if it's too much, then move on to the next section.
#
# PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training
# a neural network on the synthetic task of predicting a center word based an
# average of both context word-vectors and the full document's doc-vector.
#
# PV-DBOW is analogous to Word2Vec SG. The doc-vectors are obtained by training
# a neural network on the synthetic task of predicting a target word just from
# the full document's doc-vector. (It is also common to combine this with
# skip-gram testing, using both the doc-vector and nearby word-vectors to
# predict a single target word, but only one at a time.)
#
# Prepare the Training and Test Data
# ----------------------------------
#
# For this tutorial, we'll be training our model using the `Lee Background
# Corpus
# <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
# included in gensim. This corpus contains 314 documents selected from the
# Australian Broadcasting Corporation’s news mail service, which provides text
# e-mails of headline stories and covers a number of broad topics.
#
# And we'll test our model by eye using the much shorter `Lee Corpus
# <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
# which contains 50 documents.
#

import os
import gensim
# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
lee_test_file = os.path.join(test_data_dir, 'lee.cor')

###############################################################################
# Define a Function to Read and Preprocess Text
# ---------------------------------------------
#
# Below, we define a function to:
#
# - open the train/test file (with latin encoding)
# - read the file line-by-line
# - pre-process each line (tokenize text into individual words, remove punctuation, set to lowercase, etc)
#
# The file we're reading is a **corpus**.
# Each line of the file is a **document**.
#
# .. Important::
#   To train the model, we'll need to associate a tag/number with each document
#   of the training corpus. In our case, the tag is simply the zero-based line
#   number.
#
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

###############################################################################
# Let's take a look at the training corpus
#
print(train_corpus[:2])

###############################################################################
# And the testing corpus looks like this:
#
print(test_corpus[:2])

###############################################################################
# Notice that the testing corpus is just a list of lists and does not contain
# any tags.
#

###############################################################################
# Training the Model
# ------------------
#
# Now, we'll instantiate a Doc2Vec model with a vector size with 50 dimensions and
# iterating over the training corpus 40 times. We set the minimum word count to
# 2 in order to discard words with very few occurrences. (Without a variety of
# representative examples, retaining such infrequent words can often make a
# model worse!) Typical iteration counts in the published `Paragraph Vector paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__
# results, using 10s-of-thousands to millions of docs, are 10-20. More
# iterations take more time and eventually reach a point of diminishing
# returns.
#
# However, this is a very very small dataset (300 documents) with shortish
# documents (a few hundred words). Adding training passes can sometimes help
# with such small datasets.
#
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

###############################################################################
# Build a vocabulary
model.build_vocab(train_corpus)

###############################################################################
# Essentially, the vocabulary is a list (accessible via
# ``model.wv.index_to_key``) of all of the unique words extracted from the training corpus.
# Additional attributes for each word are available using the ``model.wv.get_vecattr()`` method,
# For example, to see how many times ``penalty`` appeared in the training corpus:
#
print(f"Word 'penalty' appeared {model.wv.get_vecattr('penalty', 'count')} times in the training corpus.")

###############################################################################
# Next, train the model on the corpus.
# In the usual case, where Gensim installation found a BLAS library for optimized
# bulk vector operations, this training on this tiny 300 document, ~60k word corpus 
# should take just a few seconds. (More realistic datasets of tens-of-millions
# of words or more take proportionately longer.) If for some reason a BLAS library 
# isn't available, training uses a fallback approach that takes 60x-120x longer, 
# so even this tiny training will take minutes rather than seconds. (And, in that 
# case, you should also notice a warning in the logging letting you know there's 
# something worth fixing.) So, be sure your installation uses the BLAS-optimized 
# Gensim if you value your time.
#
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

###############################################################################
# Now, we can use the trained model to infer a vector for any piece of text
# by passing a list of words to the ``model.infer_vector`` function. This
# vector can then be compared with other vectors via cosine similarity.
#
vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print(vector)

###############################################################################
# Note that ``infer_vector()`` does *not* take a string, but rather a list of
# string tokens, which should have already been tokenized the same way as the
# ``words`` property of original training document objects.
#
# Also note that because the underlying training/inference algorithms are an
# iterative approximation problem that makes use of internal randomization,
# repeated inferences of the same text will return slightly different vectors.
#

###############################################################################
# Assessing the Model
# -------------------
#
# To assess our new model, we'll first infer new vectors for each document of
# the training corpus, compare the inferred vectors with the training corpus,
# and then returning the rank of the document based on self-similarity.
# Basically, we're pretending as if the training corpus is some new unseen data
# and then seeing how they compare with the trained model. The expectation is
# that we've likely overfit our model (i.e., all of the ranks will be less than
# 2) and so we should be able to find similar documents very easily.
# Additionally, we'll keep track of the second ranks for a comparison of less
# similar documents.
#
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

###############################################################################
# Let's count how each document ranks with respect to the training corpus
#
# NB. Results vary between runs due to random seeding and very small corpus
import collections

counter = collections.Counter(ranks)
print(counter)

###############################################################################
# Basically, greater than 95% of the inferred documents are found to be most
# similar to itself and about 5% of the time it is mistakenly most similar to
# another document. Checking the inferred-vector against a
# training-vector is a sort of 'sanity check' as to whether the model is
# behaving in a usefully consistent manner, though not a real 'accuracy' value.
#
# This is great and not entirely surprising. We can take a look at an example:
#
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

###############################################################################
# Notice above that the most similar document (usually the same text) is has a
# similarity score approaching 1.0. However, the similarity score for the
# second-ranked documents should be significantly lower (assuming the documents
# are in fact different) and the reasoning becomes obvious when we examine the
# text itself.
#
# We can run the next cell repeatedly to see a sampling other target-document
# comparisons.
#

# Pick a random document from the corpus and infer a vector from the model
import random
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

###############################################################################
# Testing the Model
# -----------------
#
# Using the same approach above, we'll infer the vector for a randomly chosen
# test document, and compare the document to our model by eye.
#

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

###############################################################################
# Conclusion
# ----------
#
# Let's review what we've seen in this tutorial:
#
# 0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
# 1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
# 2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
# 3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
# 4. Assess the model
# 5. Test the model on the test corpus
#
# That's it! Doc2Vec is a great way to explore relationships between documents.
#
# Additional Resources
# --------------------
#
# If you'd like to know more about the subject matter of this tutorial, check out the links below.
#
# * `Word2Vec Paper <https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_
# * `Doc2Vec Paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_
# * `Dr. Michael D. Lee's Website <http://faculty.sites.uci.edu/mdlee>`_
# * `Lee Corpus <http://faculty.sites.uci.edu/mdlee/similarity-data/>`__
# * `IMDB Doc2Vec Tutorial <doc2vec-IMDB.ipynb>`_
#
