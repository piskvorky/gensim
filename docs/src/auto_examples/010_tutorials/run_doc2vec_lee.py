r"""

.. _doc2vec_lee_py:

Doc2Vec Model
=============

Introduces Gensim's Doc2Vec model and demonstrates its use on the Lee Corpus.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Doc2Vec is an NLP tool for representing documents as a vector and is a
# generalizing of the Word2Vec method.  This tutorial will serve as an
# introduction to Doc2Vec and present ways to train and assess a Doc2Vec model.
#
# This tutorial will take you through the following steps:
#
# 1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
# 2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
# 3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
# 4. Assess the model
# 5. Test the model on the test corpus
#
# Getting Started
# ---------------
# 
# To get going, we'll need to have a set of documents to train our doc2vec
# model. In theory, a document could be anything from a short 140 character
# tweet, a single paragraph (i.e., journal article abstract), a news article,
# or a book. In NLP parlance a collection or set of documents is often referred
# to as a **corpus**. 
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
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

###############################################################################
# Define a Function to Read and Preprocess Text
# ---------------------------------------------
# 
# Below, we define a function to open the train/test file (with latin
# encoding), read the file line-by-line, pre-process each line using a simple
# gensim pre-processing tool (i.e., tokenize text into individual words, remove
# punctuation, set to lowercase, etc), and return a list of words. Note that,
# for a given file (aka corpus), each continuous line constitutes a single
# document and the length of each line (i.e., document) can vary. Also, to
# train the model, we'll need to associate a tag/number with each document of
# the training corpus. In our case, the tag is simply the zero-based line
# number.
# 
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

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
# Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and
# iterating over the training corpus 40 times. We set the minimum word count to
# 2 in order to discard words with very few occurrences. (Without a variety of
# representative examples, retaining such infrequent words can often make a
# model worse!) Typical iteration counts in published 'Paragraph Vectors'
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
# Essentially, the vocabulary is a dictionary (accessible via
# ``model.wv.vocab``\ ) of all of the unique words extracted from the training
# corpus along with the count (e.g., ``model.wv.vocab['penalty'].count`` for
# counts for the word ``penalty``\ ).
# 

###############################################################################
# Next, train the model on the corpus.
# If the BLAS library is being used, this should take no more than 3 seconds.
# If the BLAS library is not being used, this should take no more than 2
# minutes, so use BLAS if you value your time.
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
# Assessing Model
# ---------------
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
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
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
# another document. the checking of an inferred-vector against a
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
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

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
