.. _tut2:

Topics and Transformations
===========================


Don't forget to set

>>> import logging
>>> logging.root.setLevel(logging.INFO) # will suppress DEBUG level events

if you want to see logging events.

Transformation interface
--------------------------

In the previous tutorial on :doc:`tut1`, we created a corpus of documents represented
as a stream of vectors. To continue, let's fire up gensim and use that corpus:

>>> from gensim import corpora, models, similarities
>>> dictionary = corpora.Dictionary.load('/tmp/dictionary.pkl')
>>> corpus = corpora.MmCorpus('/tmp/deerwester.mm')
>>> print corpus
MmCorpus(9 documents, 12 features, 28 non-zero entries)

In this tutorial, we will show how to transform documents from one vector representation
into another. This process serves two goals:

1. To bring out hidden structure in the corpus, discover relationships between
   the original features and use them to describe the documents in a new and 
   (hopefully) more realistic way.
2. To make the document representation more compact. This both improves efficiency
   (new representation consumes less resources) and efficacy (marginal data 
   trends are ignored, so that transformations can be thought of as noise-reduction). 

The transformations are standard Python objects, typically initialized by means of 
a :dfn:`training corpus`:

>>> tfidf = models.TfidfModel(corpus)

We used our old corpus to initialize (train) the transformation model. In case of TfIdf, the 
"training" consists of going through the corpus once and computing document frequencies
of all its features.

From now on, ``tfidf`` is treated as a read-only object that can be used to convert 
any vector from the old representation (bag-of-words counts) to the new representation
(TfIdf):

>>> doc_bow = [(0, 1), (1, 1)]
>>> print tfidf[doc_bow]
[(0, 0.70710678), (1, 0.70710678)]

To apply a transformation to the whole corpus:

>>> corpus_tfidf = tfidf[corpus]

This creates a standard corpus (stream of documents), which can be iterated over,
saved to disk etc.

.. warning::
  Calling ``transformation[corpus]`` only creates a wrapper around the old ``corpus``
  document stream -- actual conversions are done on-the-fly, during document iteration. 
  Conversion at the time of calling ``corpus2 = transformation[corpus]`` would mean
  storing the result in memory, which contradicts gensim's objective of memory-indepedence.
  If you will be iterating over the transformed ``corpus2`` multiple times, and the 
  transformation is costly, :ref:`store the resulting corpus to disk first <corpus-formats>` and continue
  using that.

Transformations can also be serialized, one on top of another, in a sort of chain:

>>> lsi = models.LsiModel(corpus_tfidf, id2word = dictionary.id2token, numTopics = 2) # initialize LSI transformation
>>> corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

Here we transformed our Tf-Idf corpus via `Latent Semantic Indexing <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
into a latent 2-D space. Now you're probably wondering what these two latent 
dimensions stand for, so let's inspect with :func:`models.LsiModel.printTopics`:

>>> for topicNo in range(lsi.numTopics):
>>>     print 'topic %i: %s' % (topicNo, lsi.printTopic(topicNo))
topic 0: -0.703 * "trees" + -0.538 * "graph" + -0.402 * "minors" + -0.187 * "survey" + -0.061 * "system" + -0.060 * "time" + -0.060 * "response" + -0.058 * "user" + -0.049 * "computer" + -0.035 * "interface" + -0.035 * "eps" + -0.030 * "human"
topic 1: 0.460 * "system" + 0.373 * "user" + 0.332 * "eps" + 0.328 * "interface" + 0.320 * "time" + 0.320 * "response" + 0.293 * "computer" + 0.280 * "human" + 0.171 * "survey" + -0.161 * "trees" + -0.076 * "graph" + -0.029 * "minors"

It appears that according to LSI, "trees", "graphs" and "minors" are all related 
words (and contribute the most to the direction of the first topic), while the 
second topic practically concerns itself with all the other words. As expected, 
the first five documents are more strongly related to the second topic and the 
remaining four documents to the first topic:

>>> for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here 
>>>     print doc
[(0, -0.066), (1, 0.520)] # "Human machine interface for lab abc computer applications"
[(0, -0.197), (1, 0.761)] # "A survey of user opinion of computer system response time"
[(0, -0.090), (1, 0.724)] # "The EPS user interface management system"
[(0, -0.076), (1, 0.632)] # "System and human system engineering testing of EPS"
[(0, -0.102), (1, 0.574)] # "Relation of user perceived response time to error measurement"
[(0, -0.703), (1, -0.161)] # "The generation of random binary unordered trees"
[(0, -0.877), (1, -0.168)] # "The intersection graph of paths in trees"
[(0, -0.910), (1, -0.141)] # "Graph minors IV Widths of trees and well quasi ordering"
[(0, -0.617), (1, 0.054)] # "Graph minors A survey"


.. note::

  Transformations are initialized to convert between two specific vector 
  spaces. Failure to use the same input feature space (such as applying a different string 
  preprocessing, or using bag-of-words vectors where TfIdf vectors are expected) 
  will result in feature mismatch during the transformation, and consequently in 
  either garbage output and/or runtime exceptions.

Model persistency is handled via the :func:`save` and :func:`load` functions:

>>> lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
>>> lsi = models.LsiModel.load('/tmp/model.lsi')


The next question might be: just how exactly similar are those documents to each other?
Is there a way to formalize the similarity, so that for a given document, we can
order some other set of documents according to their similarity? Similarity queries
are covered in the :doc:`next tutorial <tut3>`.

Available transformations
--------------------------

Gensim implements several popular Vector Space Model algorithms:

* `Latent Semantic Indexing, LSI, LSA <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
  transforms documents from either bag-of-words or (preferrably) TfIdf-weighted space into
  a latent space of a lower dimensionality. For the toy corpus above we used only 
  2 latent dimensions, but on real corpora, target dimensionality of 200--500 is recommended
  as a "golden standard" [1]_.

* `Random Projections, RP <www.cis.hut.fi/ella/publications/randproj_kdd.pdf>`_ aim to
  reduce vector space dimensionality. This is a very efficient (both memory- and
  CPU-friendly) approach to approximating TfIdf through randomness. Recommended
  target dimensionality is again in the hundreds/thousands, depending on your dataset.

* `Latent Dirichlet Allocation, LDA <http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_
  is yet another transformation, from bag-of-words counts into a topic space of low 
  dimensionality (a few hundreds). LDA is **much** slower than the other algorithms,
  and we are currently looking into ways of making it faster (see eg. [2]_). If you'd like
  to contribute, `let us know <mailto:radimrehurek@seznam.cz>`_!

Adding new :abbr:`VSM (Vector Space Model)` transformations (such as different weighting schemes) is rather trivial;
see the :doc:`API reference <apiref>` or directly the Python code for more info and examples.

It is probably worth repeating that these are all unique, **incremental** implementations, 
which do not require the whole training corpus to be present in main memory at once.
With memory taken care of, we are now investigating available lightweight Python 
frameworks for distributed computing, to improve CPU efficiency, too. 
If you feel you could contribute, please `let us know <mailto:radimrehurek@seznam.cz>`_! 


------

.. [1] Bradford, R.B., 2008. An empirical study of required dimensionality for large-scale latent semantic indexing applications.

.. [2] Asuncion, A., 2009. On Smoothing and Inference for Topic Models.
