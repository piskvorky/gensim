r"""
Getting Started with Gensim
===========================

A gentle introduction to Gensim.

This example demonstrates using gensim to:

    1. Create a toy corpus and store it in memory
    2. Create a Tf-Idf transformation of the corpus
    3. Calculate the similarity between all the documents in the corpus

Let's begin.
"""

###############################################################################
# First, let’s create a small corpus of nine documents and twelve features [1]_:

corpus = [
    [(0, 1.0), (1, 1.0), (2, 1.0)],
    [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
    [(0, 1.0), (4, 2.0), (7, 1.0)],
    [(3, 1.0), (5, 1.0), (6, 1.0)],
    [(9, 1.0)],
    [(9, 1.0), (10, 1.0)],
    [(9, 1.0), (10, 1.0), (11, 1.0)],
    [(8, 1.0), (10, 1.0), (11, 1.0)],
]

###############################################################################
# Each of the elements in `corpus` corresponds to a document.
# A document consists of `features`.
# In the above representation, we use tuples to represent features.
# So, the first document included features 0, 1, and 2 only.
# This representation is known as the `Vector Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_.
#
# If you’re not familiar with the vector space model, we’ll bridge the gap between raw strings, corpora and sparse vectors in the next tutorial on Corpora and Vector Spaces.
# If you’re familiar with the vector space model, you’ll probably know that the way you parse your documents and convert them to vectors has major impact on the quality of any subsequent applications.
#
# In `gensim`, a :dfn:`corpus` is simply an object which, when iterated over, returns its documents represented as sparse vectors. In this case we’re using a list of list of tuples.
#
# .. note::
#   In this example, the whole corpus is stored in memory, as a Python list.
#   However, the corpus interface only dictates that a corpus must support
#   iteration over its constituent documents. For very large corpora, it is
#   advantageous to keep the corpus on disk, and access its documents
#   sequentially, one at a time. All the operations and transformations are
#   implemented in such a way that makes them independent of the size of the
#   corpus, memory-wise.
#
# Next, let’s import gensim and initialize a :dfn:`transformation`:
#
from gensim import models

tfidf = models.TfidfModel(corpus)

###############################################################################
# A transformation is used to convert documents from one vector representation into another.
# Here, we used `Tf-Idf <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_, a simple transformation which takes documents represented as bag-of-words counts and applies a weighting which discounts common terms (or, equivalently, promotes rare terms). It also scales the resulting vector to unit length (in the `Euclidean norm <http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Euclidean_norm>`_).
#
# Transformations are covered in detail in the tutorial on Topics and Transformations.
#
# So, given a new vector corresponding to another document (*not* in the original corpus), we can get its representation via Tf-Idf as:

vec = [(0, 1), (4, 1)]
print(tfidf[vec])

###############################################################################
# To transform the whole corpus via TfIdf and index it, in preparation for similarity queries:
#
from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

###############################################################################
# and to query the similarity of our query vector vec against every document in the corpus:

sims = index[tfidf[vec]]
print(list(enumerate(sims)))

###############################################################################
# How to read this output? Document number zero (the first document) has a similarity score of 0.466=46.6%, the second document has a similarity score of 19.1% etc.
# We can make this slightly more readable by sorting:

for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)

###############################################################################
# Thus, according to TfIdf document representation and cosine similarity measure, the most similar to our query document vec is document no. 3, with a similarity score of 82.1%. Note that in the TfIdf representation, any documents which do not share any common features with vec at all (documents no. 4–8) get a similarity score of 0.0. See the Similarity Queries tutorial for more detail.
#
# .. [1] This is the same corpus as used in
#        `Deerwester et al. (1990): Indexing by Latent Semantic Analysis <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_, Table 2.

