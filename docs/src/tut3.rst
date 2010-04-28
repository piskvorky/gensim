.. _tut3:

Similarity Queries
===========================


Don't forget to set

>>> import logging
>>> logging.root.setLevel(logging.INFO) # will suppress DEBUG level events

if you want to see logging events.

Similarity interface
--------------------------

In the previous tutorials on :doc:`tut1` and :doc:`tut2`, we covered what it means 
to create a corpus in the Vector Space Model and how to transform it between different
vector spaces. A common reason for such a charade is that we want to determine 
**similarity between pairs of documents**, or the **similarity between a specific document
and a set of other documents** (such as a user query vs. indexed documents).

To show how this can be done in gensim, let us consider the same corpus as in the 
previous examples (which really originally comes from Deerwester et al.'s 
`"Indexing by Latent Semantic Analysis" <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_ 
seminal 1990 article):

>>> from gensim import corpora, models, similarities
>>> dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
>>> corpus = corpora.MmCorpus('/tmp/deerwester.mm')
>>> print corpus
MmCorpus(9 documents, 12 features, 28 non-zero entries)

To follow Deerwester's example, we first use this tiny corpus to define a 2-dimensional 
LSI space:

>>> lsi = models.LsiModel(corpus, id2word = dictionary.id2token, numTopics = 2)
  
Now suppose a user typed in the query `"Human computer interaction"`. We would 
like to sort our nine corpus documents in decreasing order of relevance to this query. 
Unlike modern search engines, here we only concentrate on a single aspect of possible 
similarities -- on apparent semantic relatedness of their texts (words). No hyperlinks,
no random-walk static ranks, just an extension over a boolean keyword match:

>>> doc = "Human computer interaction"
>>> vec_bow = dictionary.doc2bow(doc.lower().split())
>>> vec_lsi = lsi[vec_bow] # convert the query to LSI space
>>> print vec_lsi # result is already scaled by singular values
[(0, -0.461821), (1, 0.070028)]

In addition, we will be considering `cosine similarity <http://en.wikipedia.org/wiki/Cosine_similarity>`_
to determine the similarity of two vectors. Cosine similarity is a standard measure 
in Vector Space Modeling, but wherever the vectors represent probability distributions, 
`different similarity measures <http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`_
may be more appropriate.

Initializing query structures
++++++++++++++++++++++++++++++++

To prepare for similarity queries, we need to enter all documents which we want
to compare against subsequent queries. In our case, they are the nine documents, 
converted to a 2-D latent space:

>>> index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and "index" it

.. warning::
  The class :class:`similarities.MatrixSimilarity` is only appropriate when the whole
  set of vectors fits into memory. For example, a corpus of one million documents
  would require 1GB of RAM in a 256-dimensional LSI space, when used with this class. 
  Without 1GB of free RAM, you would need to use the :class:`similarities.Similarity` class.
  This class operates in constant memory, in a streaming (and more gensim-like) 
  fashion, but is also much slower than :class:`similarities.MatrixSimilarity`, which uses
  fast level-2 `BLAS routines <http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_
  to determine similarities. We are currently searching for a lightweight Python
  framework for distributed computing, which will allow us to perform fast queries
  without compromising gensim's memory independence, by spreading the query load 
  over many computers.

Index persistency is handled via the :func:`save` and :func:`load` functions:

>>> index.save('/tmp/deerwester.index')
>>> index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')


Performing queries
+++++++++++++++++++++

To obtain similarities of our query document against the nine indexed documents:

>>> sims = index[vec_lsi] # perform a similarity query against the corpus
>>> print list(enumerate(sims))
[(0, 0.99809301), (1, 0.93748635), (2, 0.99844527), (3, 0.9865886), (4, 0.90755945), 
(5, -0.12416792), (6, -0.1063926), (7, -0.098794639), (8, 0.05004178)]

Cosine measure returns similarities in the range `<-1, 1>` (the greater, the more similar),
so that the first document has a score of 0.99809301 etc.

With some standard Python magic we sort these similarities into descending 
order, and obtain the final answer to the query `"Human computer interaction"`:

>>> sims = sorted(list(enumerate(sims)), key = lambda item: -item[1])
>>> print sims
[(2, 0.99844527), # The EPS user interface management system
(0, 0.99809301), # Human machine interface for lab abc computer applications
(3, 0.9865886), # System and human system engineering testing of EPS
(1, 0.93748635), # A survey of user opinion of computer system response time
(4, 0.90755945), # Relation of user perceived response time to error measurement
(8, 0.050041795), # Graph minors A survey
(7, -0.098794639), # Graph minors IV Widths of trees and well quasi ordering
(6, -0.1063926), # The intersection graph of paths in trees
(5, -0.12416792)] # The generation of random binary unordered trees

(We added the original documents in their original "string form" to the output, to 
improve clarity.)

The thing to note here is that documents no. 2 (``"The EPS user interface management system"``)
and 4 (``"Relation of user perceived response time to error measurement"``) would never be returned by
a standard boolean fulltext search, because they do not share any common words with ``"Human 
computer interaction"``. However, after applying LSI, we can observe that both of 
them received quite high similarity scores, which corresponds better to our intuition of
them sharing a "computer-related" topic with the query. In fact, this is the reason 
why we apply transformations and do topic modeling in the first place.


Where next?
------------

Congratulations, you have finished the tutorials -- now you know how gensim works :-)
To delve into more details, you can browse through the :doc:`API documentation <apiref>`.

Please remember that gensim is an experimental package, aimed at the NLP research community.
This means that:

* there certainly are parts that may be implemented more efficiently (in C, for example), and there may also be bugs in the code
* your **feedback is most welcome** and appreciated, be it in code and idea contributions, bug reports or just user stories.

