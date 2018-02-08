:orphan:

.. _changes_080:

Change Set for 0.8.0
============================

Release 0.8.0 concludes the 0.7.x series, which was about API consolidation and performance.
In 0.8.x, I'd like to extend `gensim` with new functionality and features.

Codestyle Changes
------------------

Codebase was modified to comply with `PEP8: Style Guide for Python Code <http://www.python.org/dev/peps/pep-0008/>`_.
This means the 0.8.0 API is **backward incompatible** with the 0.7.x series.

That's not as tragic as it sounds, gensim was almost there anyway. The changes are few and pretty straightforward:

1. the `numTopics` parameter is now `num_topics`
2. `addDocuments()` method becomes `add_documents()`
3. `toUtf8()` => `to_utf8()`
4. ... you get the idea: replace `camelCase` with `lowercase_with_underscores`.

If you stored a model that is affected by this to disk, you'll need to rename its attributes manually:

>>> lsa = gensim.models.LsiModel.load('/some/path') # load old <0.8.0 model
>>> lsa.num_terms, lsa.num_topics = lsa.numTerms, lsa.numTopics # rename attributes
>>> del lsa.numTerms, lsa.numTopics # clean up old attributes (optional)
>>> lsa.save('/some/path') # save again to disk, as 0.8.0 compatible

Only attributes (variables) need to be renamed; method names (functions) are not affected, due to the way `pickle` works.

Similarity Queries
-------------------

Improved speed and scalability of :doc:`similarity queries <tut2>`.

The `Similarity` class can now index corpora of arbitrary size more efficiently.
Internally, this is done by splitting the index into several smaller pieces ("shards") that fit in RAM
and can be processed independently. In addition, documents can now be added to a `Similarity` index dynamically.

There is also a new way to query the similarity indexes:

>>> index = MatrixSimilarity(corpus) # create an index
>>> sims = index[document] # get cosine similarity of query "document" against every document in the index
>>> sims = index[chunk_of_documents] # new syntax!

Advantage of the last line (querying multiple documents at the same time) is faster execution.

This faster execution is also utilized *automatically for you* if you're using the ``for sims in index: ...`` syntax
(which returns pairwise similarities of documents in the index).

To see the speed-up on your machine, run ``python -m gensim.test.simspeed`` (and compare to my results `here <http://groups.google.com/group/gensim/msg/4f6f171a869e4fca?>`_ to see how your machine fares).

.. note::
  This current functionality of querying is as far as I wanted to get with gensim.
  More optimizations and smarter indexing are certainly possible, but I'd like to
  focus on other features now. Pull requests are still welcome though :)

Check out the :mod:`updated documentation <gensim.similarities.docsim>` of the similarity classes for more info.

Simplified Directory Structure
--------------------------------

Instead of the java-esque ``ROOT_DIR/src/gensim`` directory structure of gensim,
the packages now reside directly in ``ROOT_DIR/gensim`` (no superfluous ``src``). See the new structure `on github <https://github.com/piskvorky/gensim>`_.

Other changes (that you're unlikely to notice unless you look)
----------------------------------------------------------------------

* Improved efficiency of ``lsi[corpus]`` transformations (documents are chunked internally for better performance).
* Large matrices (numpy/scipy.sparse, in `LsiModel`, `Similarity` etc.) are now mmapped to/from disk when doing `save/load`. The `cPickle` approach used previously was too `buggy <http://groups.google.com/group/gensim/browse_thread/thread/3c4c6c0f76c5938c#>`_ and `slow <http://dieter.plaetinck.be/poor_mans_pickle_implementations_benchmark.html>`_.
* Renamed `chunks` parameter to `chunksize` (i.e. `LsiModel(corpus, num_topics=100, chunksize=20000)`). This better reflects its purpose: size of a chunk=number of documents to be processed at once.
* Also improved memory efficiency of LSI and LDA model generation (again).
* Removed SciPy 0.6 from the list of supported SciPi versions (need >=0.7 now).
* Added more unit tests.
* Several smaller fixes; see the `commit history <https://github.com/piskvorky/gensim/commits/0.8.0>`_ for full account.

.. admonition:: Future Directions?

   If you have ideas or proposals for new features for 0.8.x, now is the time to let me know:
   `gensim mailing list <http://groups.google.com/group/gensim>`_.
