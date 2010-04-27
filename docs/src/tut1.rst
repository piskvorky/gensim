.. _tut1:

Corpora and Vector Spaces
===================================

Don't forget to set

>>> import logging
>>> logging.root.setLevel(logging.INFO) # will suppress DEBUG level events

if you want to see logging events.


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
by the features extracted from it, not by its "surface" string form: how you get to
the features is up to you. Below we describe one common, general-purpose approach (called 
:dfn:`bag-of-words`), but keep in mind that different application domains call for 
different features, and, as always, it's `garbage in, garbage out <http://en.wikipedia.org/wiki/Garbage_In,_Garbage_Out>`_...

To convert documents to vectors, we will use a document representation called 
`bag-of-words <http://en.wikipedia.org/wiki/Bag_of_words>`_. In this representation, 
each document is represented by one vector where each vector element represents
a question-answer pair, in the style of:

 "How many times does the word `system` appear in the document? Once."

It is advantageous to represent the questions only by their (integer) ids. The mapping 
between the questions and ids is called a dictionary:

>>> dictionary = corpora.Dictionary.fromDocuments(texts)
>>> dictionary.save('/tmp/dictionary.pkl') # store the dictionary, for future reference
>>> print dictionary
Dictionary(12 unique tokens)

Here we assigned a unique integer id to all words appearing in the corpus by calling
:func:`Dictionary.fromDocuments`. This sweeps across the texts, collecting words 
and relevant statistics. In the end, we see there are twelve distinct words in the 
processed corpus, which means each document will be represented by twelve numbers (ie., by a 12-D vector). 
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
and returns the result as a sparse vector. The sparse vector ``[(0, 1), (1, 1)]``
therefore reads: in the document `"Human computer interaction"`, the words `computer`
(id 0) and `human` (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.

>>> corpus = [dictionary.doc2bow(text) for text in texts]
>>> corpora.MmCorpus.saveCorpus('/tmp/deerwester.mm', corpus) # store to disk, for later use
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

By now it should be clear that the vector feature with ``id=10`` stands for the question "How many 
times does the word `graph` appear in the document?" and that the answer is "zero" for 
the first six documents and "one" for the remaining three. As a matter of fact, 
we have arrived at exactly the same corpus of vectors as in the :ref:`first-example`.

And that is all there is to it! At least as far as bag-of-words representation is concerned. 
Of course, what we do with such corpus is another question; it is not at all clear
how counting the frequency of distinct words could be useful. As it turns out, it isn't, and 
we will need to apply a transformation on this simple representation first, before
we can use it to compute any meaningful document vs. document similarities. 
Transformations are covered in the :doc:`next tutorial <tut2>`, but before that, let's
briefly turn our attention to *corpus persistency*.


.. _corpus-formats:

Corpus Formats
---------------

There exist several file formats for storing a Vector Space corpus (~sequence of vectors) to disk.
`Gensim` implements them via the *streaming corpus interface* mentioned earlier:
documents are read from disk in a lazy fashion, one document at a time, without the whole
corpus being read into main memory at once.

One of the more notable formats is the `Market Matrix format <http://math.nist.gov/MatrixMarket/formats.html>`_.
To save a corpus in the Matrix Market format:

>>> from gensim import corpora
>>>
>>> corpus = [[(1, 0.5)], []] # create a toy corpus of 2 documents (one of them empty, for the heck of it)
>>>
>>> corpora.MmCorpus.saveCorpus('/tmp/corpus.mm', corpus)

Other formats include `Joachim's SVMlight format <svmlight.joachims.org/>`_, 
`Blei's LDA-C format <www.cs.princeton.edu/~blei/lda-c/>`_ and 
`GibbsLDA++ format <http://gibbslda.sourceforge.net/>`_. 

>>> corpora.SvmLightCorpus.saveCorpus('/tmp/corpus.svmlight', corpus)
>>> corpora.BleiCorpus.saveCorpus('/tmp/corpus.lda-c', corpus)
>>> corpora.LowCorpus.saveCorpus('/tmp/corpus.low', corpus)


Conversely, to load a corpus iterator from a Matrix Market file:

>>> corpus = corpora.MmCorpus('/tmp/corpus.mm')

Corpus objects are streams, so typically you won't be able to print them directly:

>>> print corpus
MmCorpus(2 documents, 2 features, 1 non-zero entries)

Instead, to view the contents of a corpus:

>>> # one way of printing a corpus: load it entirely into memory
>>> print list(corpus) # calling list() will convert any sequence to a plain Python list
[[(1, 0.5)], []]

or

>>> # another way of doing it: print one document at a time, making use of the streaming interface
>>> for doc in corpus:
>>>     print doc
[(1, 0.5)]
[]

The second way is obviously more memory-friendly, but for testing and development
purposes, nothing beats the simplicity of calling ``list(corpus)``.

To save the same corpus in Blei's LDA-C format,

>>> corpora.BleiCorpus.saveCorpus('/tmp/corpus.lda-c', corpus)

In this way, `gensim` can also be used as a simple **I/O format conversion tool**: 
just load a document stream using one format and immediately save it in another format.

For a complete reference, see the :doc:`API documentation <apiref>`. Or continue
to the next tutorial on :doc:`tut2`.


------

.. [1]  This is the same corpus as used in 
        `Deerwester et al. (1990): Indexing by Latent Semantic Analysis <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_, Table 2.

