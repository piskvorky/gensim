
# coding: utf-8
"""
plot_gensim_Quick_Start.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

#    # Getting Started with gensim

# This section introduces the basic concepts and terms needed to understand and use `gensim` and provides a simple usage example.
#
# ## Core Concepts and Simple Example
#
# At a very high-level, `gensim` is a tool for discovering the semantic structure of documents by examining the patterns of words (or higher-level structures such as entire sentences or documents). `gensim` accomplishes this by taking a *corpus*, a collection of text documents, and producing a *vector* representation of the text in the corpus. The vector representation can then be used to train a *model*, which is an algorithms to create different representations of the data, which are usually more semantic. These three concepts are key to understanding how `gensim` works so let's take a moment to explain what each of them means. At the same time, we'll work through a simple example that illustrates each of them.
#
# ### Corpus
#
# A *corpus* is a collection of digital documents. This collection is the input to `gensim` from which it will infer the structure of the documents, their topics, etc. The latent structure inferred from the corpus can later be used to assign topics to new documents which were not present in the training corpus. For this reason, we also refer to this collection as the *training corpus*. No human intervention (such as tagging the documents by hand) is required - the topic classification is [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning.html).
#
# For our corpus, we'll use a list of 9 strings, each consisting of only a single sentence.

# In[1]:

raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


# This is a particularly small example of a corpus for illustration purposes. Another example could be a list of all the plays written by Shakespeare, list of all wikipedia articles, or all tweets by a particular person of interest.
#
# After collecting our corpus, there are typically a number of preprocessing steps we want to undertake. We'll keep it simple and just remove some commonly used English words (such as 'the') and words that occur only once in the corpus. In the process of doing so, we'll [tokenise](https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)) our data. Tokenization breaks up the documents into words (in this case using space as a delimiter).

# In[2]:

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
processed_corpus


# Before proceeding, we want to associate each word in the corpus with a unique integer ID. We can do this using the `gensim.corpora.Dictionary` class. This dictionary defines the vocabulary of all words that our processing knows about.

# In[3]:

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)


# Because our corpus is small, there is only 12 different tokens in this `Dictionary`. For larger corpuses, dictionaries that contains hundreds of thousands of tokens are quite common.

# ### Vector
#
# To infer the latent structure in our corpus we need a way to represent documents that we can manipulate mathematically. One approach is to represent each document as a vector. There are various approaches for creating a vector representation of a document but a simple example is the *bag-of-words model*. Under the bag-of-words model each document is represented by a vector containing the frequency counts of each word in the dictionary. For example, given a dictionary containing the words `['coffee', 'milk', 'sugar', 'spoon']` a document consisting of the string `"coffee milk coffee"` could be represented by the vector `[2, 1, 0, 0]` where the entries of the vector are (in order) the occurrences of "coffee", "milk", "sugar" and "spoon" in the document. The length of the vector is the number of entries in the dictionary. One of the main properties of the bag-of-words model is that it completely ignores the order of the tokens in the document that is encoded, which is where the name bag-of-words comes from.
#
# Our processed corpus has 12 unique words in it, which means that each document will be represented by a 12-dimensional vector under the bag-of-words model. We can use the dictionary to turn tokenized documents into these 12-dimensional vectors. We can see what these IDs correspond to:

# In[4]:

print(dictionary.token2id)


# For example, suppose we wanted to vectorize the phrase "Human computer interaction" (note that this phrase was not in our original corpus). We can create the bag-of-word representation for a document using the `doc2bow` method of the dictionary, which returns a sparse representation of the word counts:

# In[5]:

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec


# The first entry in each tuple corresponds to the ID of the token in the dictionary, the second corresponds to the count of this token.

# Note that "interaction" did not occur in the original corpus and so it was not included in the vectorization. Also note that this vector only contains entries for words that actually appeared in the document. Because any given document will only contain a few words out of the many words in the dictionary, words that do not appear in the vectorization are represented as implicitly zero as a space saving measure.
#
# We can convert our entire original corpus to a list of vectors:

# In[6]:

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
bow_corpus


# Note that while this list lives entirely in memory, while in most applications you will want a more scalable solution. Luckily, `gensim` allows you to use any iterator that returns a single document vector at a time. See the documentation for more details.
#
# ### Model
#
# Now that we have vectorized our corpus we can begin to transform it using *models*. We use model as an abstract term referring to a transformation from one document representation to another. In `gensim` documents are represented as vectors so a model can be thought of as a transformation between two vector spaces. The details of this transformation are learned from the training corpus.
#
# One simple example of a model is [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).  The tf-idf model transforms vectors from the bag-of-words representation to a vector space where the frequency counts are weighted according to the relative rarity of each word in the corpus.
#
# Here's a simple example. Let's initialize the tf-idf model, training it on our corpus and transforming the string "system minors":

# In[7]:

from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)
# transform the "system minors" sting
tfidf[dictionary.doc2bow("system minors".lower().split())]


# The `tfidf` model agains returns a list of tuples, where the first entry is the token ID and the second entry is the tf-idf weighting. Note that the ID corresponding to "system" (which occurred 4 times in the original corpus) has been weighted lower than the ID corresponding to "minors" (which only occurred twice).
#
# `gensim` offers a number of different models/transformations. See [Transformations and Topics](https://radimrehurek.com/gensim/tut2.html) for details.
