.. _changes_080:

Change Set for 0.8.0
============================

Release 0.8.0 concludes the 0.7.x series, which was about API consolidation and performance.
In 0.8.x, I'd like to extend `gensim` with new functionality and features.

Codestyle Changes
------------------

Codebase was modified to comply with `PEP8: Style Guide for Python Code <http://www.python.org/dev/peps/pep-0008/>`_.
This means the 0.8.0 API is **backward incompatible** with the 0.7.x series.

That's not as tragic as it sounds, the changes were actually very few and pretty straightforward:

1. the `numTopics` parameter is now `num_topics`
2. `addDocuments()` method becomes `add_documents()`
3. `printTopics()` => `print_topics()`
4. ... you get the idea: replace `camelCase` with `lowercase_with_underscores`.

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
This faster execution is also utilized automatically if you're using the ``for sims in index: ...`` syntax,
which returns pairwise similarities of documents within the index.

To see the speed-up on your machine, run ``python -m gensim.test.simspeed`` (compare to my results `here <http://groups.google.com/group/gensim/msg/4f6f171a869e4fca?>`_).

.. note::
  This current functionality of querying is as far as I wanted to get with gensim.
  More optimizations and smarter indexing are certainly possible, but I'd like to
  focus on other features now. Pull requests are still welcome though :)

Check out the :mod:`updated documentation <gensim.similarities.docsim>` of the similarity classes for more info.

Simplified Directory Structure
--------------------------------

Instead of the java-esque ``ROOT_DIR/src/gensim`` directory structure of gensim,
the packages now reside directly in ``ROOT_DIR/gensim`` (no superfluous ``src``). See the new structure `on github <https://github.com/piskvorky/gensim>`_.

Other
-------

* Improved efficiency of ``lsi[corpus]`` transformations (documents are chunked internally for better performance).
* Also improved memory efficiency of LSI model generation (again).
* Several smaller fixes; see the `commit history <https://github.com/piskvorky/gensim/commits/0.8.0>`_ for full account.

.. admonition:: Future Directions?

   If you have ideas or proposals for new features for 0.8.x, now is the time to let me know:
   `gensim mailing list <http://groups.google.com/group/gensim>`_.
