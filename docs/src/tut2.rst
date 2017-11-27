.. _tut2:

Topics and Transformations
===========================


Don't forget to set

>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if you want to see logging events.

Transformation interface
--------------------------

In the previous tutorial on :doc:`tut1`, we created a corpus of documents represented
as a stream of vectors. To continue, let's fire up gensim and use that corpus:

>>> from gensim import corpora, models, similarities
>>> if (os.path.exists("/tmp/deerwester.dict")):
>>>    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
>>>    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
>>>    print("Used files generated from first tutorial")
>>> else:
>>>    print("Please run first tutorial to generate data set")

MmCorpus(9 documents, 12 features, 28 non-zero entries)

In this tutorial, I will show how to transform documents from one vector representation
into another. This process serves two goals:

1. To bring out hidden structure in the corpus, discover relationships between
   words and use them to describe the documents in a new and
   (hopefully) more semantic way.
2. To make the document representation more compact. This both improves efficiency
   (new representation consumes less resources) and efficacy (marginal data
   trends are ignored, noise-reduction).

Creating a transformation
++++++++++++++++++++++++++

The transformations are standard Python objects, typically initialized by means of
a :dfn:`training corpus`:

>>> tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

We used our old corpus from tutorial 1 to initialize (train) the transformation model. Different
transformations may require different initialization parameters; in case of TfIdf, the
"training" consists simply of going through the supplied corpus once and computing document frequencies
of all its features. Training other models, such as Latent Semantic Analysis or Latent Dirichlet
Allocation, is much more involved and, consequently, takes much more time.

.. note::

  Transformations always convert between two specific vector
  spaces. The same vector space (= the same set of feature ids) must be used for training
  as well as for subsequent vector transformations. Failure to use the same input
  feature space, such as applying a different string preprocessing, using different
  feature ids, or using bag-of-words input vectors where TfIdf vectors are expected, will
  result in feature mismatch during transformation calls and consequently in either
  garbage output and/or runtime exceptions.


Transforming vectors
+++++++++++++++++++++

From now on, ``tfidf`` is treated as a read-only object that can be used to convert
any vector from the old representation (bag-of-words integer counts) to the new representation
(TfIdf real-valued weights):

>>> doc_bow = [(0, 1), (1, 1)]
>>> print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors
[(0, 0.70710678), (1, 0.70710678)]

Or to apply a transformation to a whole corpus:

>>> corpus_tfidf = tfidf[corpus]
>>> for doc in corpus_tfidf:
...     print(doc)
[(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
[(0, 0.44424552527467476), (3, 0.44424552527467476), (4, 0.44424552527467476), (5, 0.32448702061385548), (6, 0.44424552527467476), (7, 0.32448702061385548)]
[(2, 0.5710059809418182), (5, 0.41707573620227772), (7, 0.41707573620227772), (8, 0.5710059809418182)]
[(1, 0.49182558987264147), (5, 0.71848116070837686), (8, 0.49182558987264147)]
[(3, 0.62825804686700459), (6, 0.62825804686700459), (7, 0.45889394536615247)]
[(9, 1.0)]
[(9, 0.70710678118654746), (10, 0.70710678118654746)]
[(9, 0.50804290089167492), (10, 0.50804290089167492), (11, 0.69554641952003704)]
[(4, 0.62825804686700459), (10, 0.45889394536615247), (11, 0.62825804686700459)]

In this particular case, we are transforming the same corpus that we used
for training, but this is only incidental. Once the transformation model has been initialized,
it can be used on any vectors (provided they come from the same vector space, of course),
even if they were not used in the training corpus at all. This is achieved by a process called
folding-in for LSA, by topic inference for LDA etc.

.. note::
  Calling ``model[corpus]`` only creates a wrapper around the old ``corpus``
  document stream -- actual conversions are done on-the-fly, during document iteration.
  We cannot convert the entire corpus at the time of calling ``corpus_transformed = model[corpus]``,
  because that would mean storing the result in main memory, and that contradicts gensim's objective of memory-indepedence.
  If you will be iterating over the transformed ``corpus_transformed`` multiple times, and the
  transformation is costly, :ref:`serialize the resulting corpus to disk first <corpus-formats>` and continue
  using that.

Transformations can also be serialized, one on top of another, in a sort of chain:

>>> lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
>>> corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

Here we transformed our Tf-Idf corpus via `Latent Semantic Indexing <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
into a latent 2-D space (2-D because we set ``num_topics=2``). Now you're probably wondering: what do these two latent
dimensions stand for? Let's inspect with :func:`models.LsiModel.print_topics`:

  >>> lsi.print_topics(2)
  topic #0(1.594): -0.703*"trees" + -0.538*"graph" + -0.402*"minors" + -0.187*"survey" + -0.061*"system" + -0.060*"response" + -0.060*"time" + -0.058*"user" + -0.049*"computer" + -0.035*"interface"
  topic #1(1.476): -0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"response" + -0.320*"time" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"

(the topics are printed to log -- see the note at the top of this page about activating
logging)

It appears that according to LSI, "trees", "graph" and "minors" are all related
words (and contribute the most to the direction of the first topic), while the
second topic practically concerns itself with all the other words. As expected,
the first five documents are more strongly related to the second topic while the
remaining four documents to the first topic:

>>> for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
...     print(doc)
[(0, -0.066), (1, 0.520)] # "Human machine interface for lab abc computer applications"
[(0, -0.197), (1, 0.761)] # "A survey of user opinion of computer system response time"
[(0, -0.090), (1, 0.724)] # "The EPS user interface management system"
[(0, -0.076), (1, 0.632)] # "System and human system engineering testing of EPS"
[(0, -0.102), (1, 0.574)] # "Relation of user perceived response time to error measurement"
[(0, -0.703), (1, -0.161)] # "The generation of random binary unordered trees"
[(0, -0.877), (1, -0.168)] # "The intersection graph of paths in trees"
[(0, -0.910), (1, -0.141)] # "Graph minors IV Widths of trees and well quasi ordering"
[(0, -0.617), (1, 0.054)] # "Graph minors A survey"


Model persistency is achieved with the :func:`save` and :func:`load` functions:

>>> lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
>>> lsi = models.LsiModel.load('/tmp/model.lsi')


The next question might be: just how exactly similar are those documents to each other?
Is there a way to formalize the similarity, so that for a given input document, we can
order some other set of documents according to their similarity? Similarity queries
are covered in the :doc:`next tutorial <tut3>`.

.. _transformations:

Available transformations
--------------------------

Gensim implements several popular Vector Space Model algorithms:

* `Term Frequency * Inverse Document Frequency, Tf-Idf <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_
  expects a bag-of-words (integer values) training corpus during initialization.
  During transformation, it will take a vector and return another vector of the
  same dimensionality, except that features which were rare in the training corpus
  will have their value increased.
  It therefore converts integer-valued vectors into real-valued ones, while leaving
  the number of dimensions intact. It can also optionally normalize the resulting
  vectors to (Euclidean) unit length.

  >>> model = models.TfidfModel(corpus, normalize=True)

* `Latent Semantic Indexing, LSI (or sometimes LSA) <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
  transforms documents from either bag-of-words or (preferrably) TfIdf-weighted space into
  a latent space of a lower dimensionality. For the toy corpus above we used only
  2 latent dimensions, but on real corpora, target dimensionality of 200--500 is recommended
  as a "golden standard" [1]_.

  >>> model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

  LSI training is unique in that we can continue "training" at any point, simply
  by providing more training documents. This is done by incremental updates to
  the underlying model, in a process called `online training`. Because of this feature, the
  input document stream may even be infinite -- just keep feeding LSI new documents
  as they arrive, while using the computed transformation model as read-only in the meanwhile!

  >>> model.add_documents(another_tfidf_corpus) # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
  >>> lsi_vec = model[tfidf_vec] # convert some new document into the LSI space, without affecting the model
  >>> ...
  >>> model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
  >>> lsi_vec = model[tfidf_vec]
  >>> ...

  See the :mod:`gensim.models.lsimodel` documentation for details on how to make
  LSI gradually "forget" old observations in infinite streams. If you want to get dirty,
  there are also parameters you can tweak that affect speed vs. memory footprint vs. numerical
  precision of the LSI algorithm.

  `gensim` uses a novel online incremental streamed distributed training algorithm (quite a mouthful!),
  which I published in [5]_. `gensim` also executes a stochastic multi-pass algorithm
  from Halko et al. [4]_ internally, to accelerate in-core part
  of the computations.
  See also :doc:`wiki` for further speed-ups by distributing the computation across
  a cluster of computers.

* `Random Projections, RP <http://www.cis.hut.fi/ella/publications/randproj_kdd.pdf>`_ aim to
  reduce vector space dimensionality. This is a very efficient (both memory- and
  CPU-friendly) approach to approximating TfIdf distances between documents, by throwing in a little randomness.
  Recommended target dimensionality is again in the hundreds/thousands, depending on your dataset.

  >>> model = models.RpModel(tfidf_corpus, num_topics=500)

* `Latent Dirichlet Allocation, LDA <http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_
  is yet another transformation from bag-of-words counts into a topic space of lower
  dimensionality. LDA is a probabilistic extension of LSA (also called multinomial PCA),
  so LDA's topics can be interpreted as probability distributions over words. These distributions are,
  just like with LSA, inferred automatically from a training corpus. Documents
  are in turn interpreted as a (soft) mixture of these topics (again, just like with LSA).

  >>> model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)

  `gensim` uses a fast implementation of online LDA parameter estimation based on [2]_,
  modified to run in :doc:`distributed mode <distributed>` on a cluster of computers.

* `Hierarchical Dirichlet Process, HDP <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_
  is a non-parametric bayesian method (note the missing number of requested topics):

  >>> model = models.HdpModel(corpus, id2word=dictionary)

  `gensim` uses a fast, online implementation based on [3]_.
  The HDP model is a new addition to `gensim`, and still rough around its academic edges -- use with care.

Adding new :abbr:`VSM (Vector Space Model)` transformations (such as different weighting schemes) is rather trivial;
see the :doc:`API reference <apiref>` or directly the `Python code <https://github.com/piskvorky/gensim/blob/develop/gensim/models/tfidfmodel.py>`_
for more info and examples.

It is worth repeating that these are all unique, **incremental** implementations,
which do not require the whole training corpus to be present in main memory all at once.
With memory taken care of, I am now improving :doc:`distributed`,
to improve CPU efficiency, too.
If you feel you could contribute (by testing, providing use-cases or code),
please `let me know <mailto:radimrehurek@seznam.cz>`_.

Continue on to the next tutorial on :doc:`tut3`.

------

.. [1] Bradford. 2008. An empirical study of required dimensionality for large-scale latent semantic indexing applications.

.. [2] Hoffman, Blei, Bach. 2010. Online learning for Latent Dirichlet Allocation.

.. [3] Wang, Paisley, Blei. 2011. Online variational inference for the hierarchical Dirichlet process.

.. [4] Halko, Martinsson, Tropp. 2009. Finding structure with randomness.

.. [5] Řehůřek. 2011. Subspace tracking for Latent Semantic Analysis.
