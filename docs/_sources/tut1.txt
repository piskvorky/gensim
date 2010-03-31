.. _tut1:

Corpora and Vector Spaces
===================================


.. _first example:

Quick Example
-------------

First, let's import gensim and create a small corpus of nine documents [1]_:

>>> from gensim import corpora, models, similarities
>>> 
>>> corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
>>>           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
>>>           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
>>>           [(0, 1.0), (4, 2.0), (7, 1.0)],
>>>           [(3, 1.0), (5, 1.0), (6, 1.0)],
>>>           [(9, 1.0)],
>>>           [(9, 1.0), (10, 1.0)],
>>>           [(9, 1.0), (10, 1.0), (11, 1.0)],
>>>           [(8, 1.0), (10, 1.0), (11, 1.0)]]

Corpus is simply an object which, when iterated over, returns its documents represented
as sparse vectors. 

If you're familiar with the `Vector Space Model (VSM) <http://en.wikipedia.org/wiki/Vector_space_model>`_,
you'll probably know that the way you parse your documents and convert them to vectors 
has major impact on the quality of any subsequent applications. If you're not familiar
with VSM, we'll bridge the gap between raw texts and vectors in the `second example`_ a 
bit later.

.. note::
    In this example, the whole corpus is stored in memory, as a Python list. However, 
    the corpus interface only dictates that a corpus must support iteration over its 
    constituent documents. For very large corpora, it is advantageous to keep the 
    corpus on disk, and access its documents sequentially, one at a time. All the 
    operations and corpora transformations
    are implemented in such a way that makes them independent of the size of the corpus,
    RAM-wise.


Next, let's initialize a transformation:

>>> tfidf = models.TfidfModel(corpus)

A transformation is used to convert documents from one vector representation into another: 

>>> vec = [(0, 1), (4, 1)]
>>> print tfidf[vec]
[(0, 0.8075244), (4, 0.5898342)]

Here, we used `Tf-Idf <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_, a simple 
transformation which takes documents represented as bag-of-words counts and applies 
a weighting which discounts common terms (or, equivalently, promotes rare terms) and
scales the resulting vector to unit length.

To index and prepare the whole TfIdf corpus for similarity queries:

>>> index = similarities.SparseMatrixSimilarity(tfidf[corpus])

and to query the similarity of our vector ``vec`` against every document in the corpus:

>>> sims = index[tfidf[vec]]
>>> print list(enumerate(sims))
[(0, 0.4662244), (1, 0.19139354), (2, 0.24600551), (3, 0.82094586), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]

According to TfIdf and cosine similarity, the most similar to our query 
document `vec` is document no. 3, with a similarity score of 82.1%. Note
that in the TfIdf representation, all documents which do not share any common features
with ``vec`` at all (documents no. 4--8) get a similarity score of 0.0.


.. _second example:

From Strings to Vectors
------------------------

This time, let's start from documents represented as strings:

>>> from gensim import corpora, models, similarities
>>>
>>> documents = ["Human machine interface for lab abc computer applications",
>>>              "A survey of user opinion of computer system response time",
>>>              "The EPS user interface management system",
>>>              "System and human system engineering testing of EPS",
>>>              "Relation of user perceived response time to error measurement",
>>>              "The generation of random binary unordered trees",
>>>              "The intersection graph of paths in trees",
>>>              "Graph minors IV Widths of trees and well quasi ordering",
>>>              "Graph minors A survey"]


This is a tiny corpus of nine documents, each consisting of only a single sentence.

First, let's tokenize the documents, remove common words (using a toy stoplist) 
as well as words that only appear once in the corpus:

>>> # remove common words and tokenize
>>> stoplist = set('for a of the and to in'.split())
>>> texts = [[word for word in document.lower().split() if word not in stoplist] 
>>>          for document in documents]
>>>
>>> # remove words that appear only once
>>> allTokens = sum(texts, [])
>>> tokensOnce = set(word for word in set(allTokens) if allTokens.count(word) == 1)
>>> texts = [[word for word in text if word not in tokensOnce]
>>>          for text in texts]
>>>
>>> print texts
[['human', 'interface', 'computer'], 
 ['survey', 'user', 'computer', 'system', 'response', 'time'], 
 ['eps', 'user', 'interface', 'system'], 
 ['system', 'human', 'system', 'eps'], 
 ['user', 'response', 'time'], 
 ['trees'], 
 ['graph', 'trees'], 
 ['graph', 'minors', 'trees'], 
 ['graph', 'minors', 'survey']]

Your way of processing the documents will likely vary; here, we only split on whitespace
to tokenize, followed by lowercasing each word. In fact, we use this particular 
(simplistic and inefficient) setup to mimick the experiment done in Deerwester et al.'s 
original LSA article [1]_.

The ways to process documents are so varied and application- and language-dependent that we
decided to *not* constrain them by any interface. Instead, a document is represented
by the features extracted from it, not by its "surface" string form. How you get to
the features is up to you; what follows below is just one common scenario.

To convert documents to vectors, we will use a document representation called 
`bag-of-words <http://en.wikipedia.org/wiki/Bag_of_words>`_. In this representation, 
each vector element is a question-answer pair, in the style of:

 "How many times does the word `system` appear in the document? Once."

The :class:`gensim.corpora.Dictionary` class can be used to convert tokenized texts
to vectors.

>>> dictionary = corpora.Dictionary.fromDocuments(texts)
>>> print dictionary
Dictionary(12 unique tokens)

Here we assigned a unique integer id to all words appearing in the corpus by calling
:func:`fromDocuments`. This sweeps across the texts, collecting words and relevant statistics. 
In the end, there are twelve distinct words in the preprocessed corpus, so each document will 
be represented by twelve numbers (ie., by a 12-D vector). 
To see the mapping between words and their ids:
 
>>> print dictionary.token2id
{'minors': 11, 'graph': 10, 'system': 5, 'trees': 9, 'eps': 8, 'computer': 0, 
'survey': 4, 'user': 7, 'human': 1, 'time': 6, 'interface': 2, 'response': 3}

To actually convert tokenized documents to vectors:

>>> newDoc = "Human computer interaction"
>>> newVec = dictionary.doc2bow(newDoc.lower().split())
>>> print newVec # the word "interaction" does not appear in the dictionary and is ignored
[(0, 1), (1, 1)]

The function :func:`doc2bow` simply counts the number of occurences of 
each distinct word, converts the word to its integer word id
and returns the result as a sparse vector. 

>>> corpus = [dictionary.doc2bow(text) for text in texts]
>>> print corpus
[[(0, 1.0), (1, 1.0), (2, 1.0)],
 [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
 [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
 [(0, 1.0), (4, 2.0), (7, 1.0)],
 [(3, 1.0), (5, 1.0), (6, 1.0)],
 [(9, 1.0)],
 [(9, 1.0), (10, 1.0)],
 [(9, 1.0), (10, 1.0), (11, 1.0)],
 [(8, 1.0), (10, 1.0), (11, 1.0)]]

For example, the vector feature with ``id=10`` stands for the question "How many 
times does the word `graph` appear in the document?". The answer is "zero" for 
the first six documents and "one" for the remaining three. As a matter of fact, 
we have arrived at exactly the same corpus of vectors as in the `first example`_.

 
To finish the example, we transform our ``"Human computer interaction"`` document
via `Latent Semantic Indexing <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
into a 2-D space:

>>> lsi = models.LsiModel(corpus, numTopics = 2)
>>>
>>> newVecLsi = lsi[newVec]
>>> print newVecLsi
[(0, -0.461821), (1, 0.0700277)]

and print proximity of this query document against every one of the nine original 
documents, in the same 2-D LSI space:

>>> index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and "index" it
>>> sims = index[newVecLsi] # perform similarity query against the corpus
>>> print list(enumerate(sims))
[(0, 0.99809301), (1, 0.93748635), (2, 0.99844527), (3, 0.9865886), (4, 0.90755945), 
(5, -0.12416792), (6, -0.1063926), (7, -0.098794639), (8, 0.05004178)]

.. decades of research in a single sentence (including 3 effective lines of code) there :-) yay

The thing to note here is that documents no. 2 (``"The EPS user interface management system"``)
and 4 (``"Relation of user perceived response time to error measurement"``) would never be returned by
a standard boolean fulltext search, because they do not share any common words with ``"Human 
computer interaction"``. However, after applying LSI, we can observe that both of 
them received high similarity scores, which corresponds better to our intuition of
them sharing a "computer-related" topic with the query. In fact, this is the reason 
why we apply transformations and do topic modeling in the first place.


Corpus Formats
---------------

There exist several file formats for storing a collection of vectors to disk.
`Gensim` implements them via the *streaming corpus interface* mentioned earlier:
documents are read from disk in a lazy fashion, one document at a time, without the whole
corpus being read into main memory at once.

One of the more notable formats is the `Market Matrix format <http://math.nist.gov/MatrixMarket/formats.html>`_.
To save a corpus in the Matrix Market format:

>>> from gensim import corpora
>>> corpora.MmCorpus.saveCorpus('/tmp/corpus.mm', corpus)

Other formats include `Joachim's SVMlight format <svmlight.joachims.org/>`_, 
`Blei's LDA-C format <www.cs.princeton.edu/~blei/lda-c/>`_ and 
`GibbsLDA++ format <http://gibbslda.sourceforge.net/>`_. 

Conversely, to load a corpus iterator from a Matrix Market file:

>>> corpus = corpora.MmCorpus('/tmp/corpus.mm')
>>> print list(corpus) # convert from MmCorpus object (document stream) to plain Python list
[[(0, 1.0), (1, 1.0), (2, 1.0)],
 [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
 [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
 [(0, 1.0), (4, 2.0), (7, 1.0)],
 [(3, 1.0), (5, 1.0), (6, 1.0)],
 [(9, 1.0)],
 [(9, 1.0), (10, 1.0)],
 [(9, 1.0), (10, 1.0), (11, 1.0)],
 [(8, 1.0), (10, 1.0), (11, 1.0)]]
 
and to save it in Blei's LDA-C format again,

>>> corpora.BleiCorpus.saveCorpus('/tmp/corpus.lda-c', corpus)

In this way, `gensim` can also be used as a simple I/O format conversion tool.

For a complete reference, see the :doc:`API documentation <apiref>`.


------

.. [1]  This is the same corpus as used in 
        `Deerwester et al. (1990): Indexing by Latent Semantic Analysis <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_, Table 2.

