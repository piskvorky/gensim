:orphan:

.. _simserver:

Document Similarity Server
=============================

The 0.7.x series of `gensim <http://radimrehurek.com/gensim/>`_ was about improving performance and consolidating API.
0.8.x will be about new features --- 0.8.1, first of the series, is a **document similarity service**.

The source code itself has been moved from gensim to its own, dedicated package, named `simserver`.
Get it from `PyPI <http://pypi.python.org/pypi/simserver>`_ or clone it on `Github <https://github.com/piskvorky/gensim-simserver>`_.

What is a document similarity service?
---------------------------------------

Conceptually, a service that lets you :

1. train a semantic model from a corpus of plain texts (no manual annotation and mark-up needed)
2. index arbitrary documents using this semantic model
3. query the index for similar documents (the query can be either an id of a document already in the index, or an arbitrary text)


>>> from simserver import SessionServer
>>> server = SessionServer('/tmp/my_server') # resume server (or create a new one)

>>> server.train(training_corpus, method='lsi') # create a semantic model
>>> server.index(some_documents) # convert plain text to semantic representation and index it
>>> server.find_similar(query) # convert query to semantic representation and compare against index
>>> ...
>>> server.index(more_documents) # add to index: incremental indexing works
>>> server.find_similar(query)
>>> ...
>>> server.delete(ids_to_delete) # incremental deleting also works
>>> server.find_similar(query)
>>> ...

.. note::
    "Semantic" here refers to semantics of the crude, statistical type --
    `Latent Semantic Analysis <http://en.wikipedia.org/wiki/Latent_semantic_analysis>`_,
    `Latent Dirichlet Allocation <http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_ etc.
    Nothing to do with the semantic web, manual resource tagging or detailed linguistic inference.


What is it good for?
---------------------

Digital libraries of (mostly) text documents. More generally, it helps you annotate,
organize and navigate documents in a more abstract way, compared to plain keyword search.

How is it unique?
-----------------

1. **Memory independent**. Gensim has unique algorithms for statistical analysis that allow
   you to create semantic models of arbitrarily large training corpora (larger than RAM) very quickly
   and in constant RAM.
2. **Memory independent (again)**. Indexing shards are stored as files to disk/mmapped back as needed,
   so you can index very large corpora. So again, constant RAM, this time independent of the number of indexed documents.
3. **Efficient**. Gensim makes heavy use of Python's NumPy and SciPy libraries to make indexing and
   querying efficient.
4. **Robust**. Modifications of the index are transactional, so you can commit/rollback an
   entire indexing session. Also, during the session, the service is still available
   for querying (using its state from when the session started). Power failures leave
   service in a consistent state (implicit rollback).
5. **Pure Python**. Well, technically, NumPy and SciPy are mostly wrapped C and Fortran, but
   `gensim <http://radimrehurek.com/gensim/>`_ itself is pure Python. No compiling, installing or root priviledges needed.
6. **Concurrency support**. The underlying service object is thread-safe and can
   therefore be used as a daemon server: clients connect to it via RPC and issue train/index/query requests remotely.
7. **Cross-network, cross-platform and cross-language**. While the Python server runs
   over TCP using `Pyro <http://irmen.home.xs4all.nl/pyro/>`_,
   clients in Java/.NET are trivial thanks to `Pyrolite <http://irmen.home.xs4all.nl/pyrolite/>`_.

The rest of this document serves as a tutorial explaining the features in more detail.

-----

Prerequisites
----------------------

It is assumed you have `gensim` properly :doc:`installed <install>`. You'll also
need the `sqlitedict <http://pypi.python.org/pypi/sqlitedict>`_ package that wraps
Python's sqlite3 module in a thread-safe manner::

    $ sudo easy_install -U sqlitedict

To test the remote server capabilities, install Pyro4 (Python Remote Objects, at
version 4.8 as of this writing)::

    $ sudo easy_install Pyro4

.. note::
    Don't forget to initialize logging to see logging messages::

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

What is a document?
-------------------

In case of text documents, the service expects::

>>> document = {'id': 'some_unique_string',
>>>             'tokens': ['content', 'of', 'the', 'document', '...'],
>>>             'other_fields_are_allowed_but_ignored': None}

This format was chosen because it coincides with plain JSON and is therefore easy to serialize and send over the wire, in almost any language.
All strings involved must be utf8-encoded.


What is a corpus?
-----------------

A sequence of documents. Anything that supports the `for document in corpus: ...`
iterator protocol. Generators are ok. Plain lists are also ok (but consume more memory).

>>> from gensim import utils
>>> texts = ["Human machine interface for lab abc computer applications",
>>>          "A survey of user opinion of computer system response time",
>>>          "The EPS user interface management system",
>>>          "System and human system engineering testing of EPS",
>>>          "Relation of user perceived response time to error measurement",
>>>          "The generation of random binary unordered trees",
>>>          "The intersection graph of paths in trees",
>>>          "Graph minors IV Widths of trees and well quasi ordering",
>>>          "Graph minors A survey"]
>>> corpus = [{'id': 'doc_%i' % num, 'tokens': utils.simple_preprocess(text)}
>>>           for num, text in enumerate(texts)]

Since corpora are allowed to be arbitrarily large, it is
recommended client splits them into smaller chunks before uploading them to the server:

>>> utils.upload_chunked(server, corpus, chunksize=1000) # send 1k docs at a time

Wait, upload what, where?
-------------------------

If you use the similarity service object (instance of :class:`simserver.SessionServer`) in
your code directly---no remote access---that's perfectly fine. Using the service remotely, from a different process/machine, is an
option, not a necessity.

Document similarity can also act as a long-running service, a daemon process on a separate machine. In that
case, I'll call the service object a *server*.

But let's start with a local object. Open your `favourite shell <http://ipython.org/>`_ and::

>>> from gensim import utils
>>> from simserver import SessionServer
>>> service = SessionServer('/tmp/my_server/') # or wherever

That initialized a new service, located in `/tmp/my_server` (you need write access rights to that directory).

.. note::
   The service is fully defined by the content of its location directory ("`/tmp/my_server/`").
   If you use an existing location, the service object will resume
   from the index found there. Also, to "clone" a service, just copy that
   directory somewhere else. The copy will be a fully working duplicate of the
   original service.


Model training
---------------

We can start indexing right away:

>>> service.index(corpus)
AttributeError: must initialize model for /tmp/my_server/b before indexing documents

Oops, we can not. The service indexes documents in a semantic representation, which
is different to the plain text we give it. We must teach the service how to convert
between plain text and semantics first::

>>> service.train(corpus, method='lsi')

That was easy. The `method='lsi'` parameter meant that we trained a model for
`Latent Semantic Indexing <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_
and default dimensionality (400) over a `tf-idf <http://en.wikipedia.org/wiki/Tf–idf>`_
representation of our little `corpus`, all automatically. More on that later.

Note that for the semantic model to make sense, it should be trained
on a corpus that is:

* Reasonably similar to the documents you want to index later. Training on a corpus
  of recipes in French when all indexed documents will be about programming in English
  will not help.
* Reasonably large (at least thousands of documents), so that the statistical analysis has
  a chance to kick in. Don't use my example corpus here of 9 documents in production O_o

Indexing documents
------------------

>>> service.index(corpus) # index the same documents that we trained on...

Indexing can happen over any documents, but I'm too lazy to create another example corpus, so we index the same 9 docs used for training.

Delete documents with::

  >>> service.delete(['doc_5', 'doc_8']) # supply a list of document ids to be removed from the index

When you pass documents that have the same id as some already indexed document,
the indexed document is overwritten by the new input (=only the latest counts;
document ids are always unique per service)::

  >>> service.index(corpus[:3]) # overall index size unchanged (just 3 docs overwritten)

The index/delete/overwrite calls can be arbitrarily interspersed with queries.
You don't have to index **all** documents first to start querying, indexing can be incremental.

Querying
---------

There are two types of queries:

1. by id:

   .. code-block:: python

     >>> print(service.find_similar('doc_0'))
     [('doc_0', 1.0, None), ('doc_2', 0.30426699, None), ('doc_1', 0.25648531, None), ('doc_3', 0.25480536, None)]

   >>> print(service.find_similar('doc_5')) # we deleted doc_5 and doc_8, remember?
   ValueError: document 'doc_5' not in index

   In the resulting 3-tuples, `doc_n` is the document id we supplied during indexing,
   `0.30426699` is the similarity of `doc_n` to the query, but what's up with that `None`, you ask?
   Well, you can associate each document with a "payload", during indexing.
   This payload object (anything pickle-able) is later returned during querying.
   If you don't specify `doc['payload']` during indexing, queries simply return `None` in the result tuple, as in our example here.

2. or by document (using `document['tokens']`; id is ignored in this case):

   .. code-block:: python

     >>> doc = {'tokens': utils.simple_preprocess('Graph and minors and humans and trees.')}
     >>> print(service.find_similar(doc, min_score=0.4, max_results=50))
     [('doc_7', 0.93350589, None), ('doc_3', 0.42718196, None)]

Remote access
-------------

So far, we did everything in our Python shell, locally. I very much like `Pyro <http://irmen.home.xs4all.nl/pyro/>`_,
a pure Python package for Remote Procedure Calls (RPC), so I'll illustrate remote
service access via Pyro. Pyro takes care of all the socket listening/request routing/data marshalling/thread
spawning, so it saves us a lot of trouble.

To create a similarity server, we just create a :class:`simserver.SessionServer` object and register it
with a Pyro daemon for remote access. There is a small `example script <https://github.com/piskvorky/gensim-simserver/blob/master/simserver/run_simserver.py>`_
included with simserver, run it with::

  $ python -m simserver.run_simserver /tmp/testserver

You can just `ctrl+c` to terminate the server, but leave it running for now.

Now open your Python shell again, in another terminal window or possibly on another machine, and::

>>> import Pyro4
>>> service = Pyro4.Proxy(Pyro4.locateNS().lookup('gensim.testserver'))

Now `service` is only a proxy object: every call is physically executed wherever
you ran the `run_server.py` script, which can be a totally different computer
(within a network broadcast domain), but you don't even know::

>>> print(service.status())
>>> service.train(corpus)
>>> service.index(other_corpus)
>>> service.find_similar(query)
>>> ...

It is worth mentioning that Irmen, the author of Pyro, also released
`Pyrolite <http://irmen.home.xs4all.nl/pyrolite/>`_ recently. That is a package
which allows you to create Pyro proxies also from Java and .NET, in addition to Python.
That way you can call remote methods from there too---the client doesn't have to be in Python.

Concurrency
-----------

Ok, now it's getting interesting. Since we can access the service remotely, what
happens if multiple clients create proxies to it at the same time? What if they
want to modify the server index at the same time?

Answer: the `SessionServer` object is thread-safe, so that when each client spawns a request
thread via Pyro, they don't step on each other's toes.

This means that:

1. There can be multiple simultaneous `service.find_similar` queries (or, in
   general, multiple simultaneus calls that are "read-only").
2. When two clients issue modification calls (`index`/`train`/`delete`/`drop_index`/...)
   at the same time, an internal lock serializes them -- the later call has to wait.
3. While one client is modifying the index, all other clients' queries still see
   the original index. Only once the modifications are committed do they become
   "visible".

What do you mean, visible?
--------------------------

The service uses transactions internally. This means that each modification is
done over a clone of the service. If the modification session fails for whatever
reason (exception in code; power failure that turns off the server; client unhappy
with how the session went), it can be rolled back. It also means other clients can
continue querying the original index during index updates.

The mechanism is hidden from users by default through auto-committing (it was already happening
in the examples above too), but auto-committing can be turned off explicitly::

  >>> service.set_autosession(False)
  >>> service.train(corpus)
  RuntimeError: must open a session before modifying SessionServer
  >>> service.open_session()
  >>> service.train(corpus)
  >>> service.index(corpus)
  >>> service.delete(doc_ids)
  >>> ...

None of these changes are visible to other clients, yet. Also, other clients'
calls to index/train/etc will block until this session is committed/rolled back---there
cannot be two open sessions at the same time.

To end a session::

  >>> service.rollback() # discard all changes since open_session()

or::

  >>> service.commit() # make changes public; now other clients can see changes/acquire the modification lock


Other stuff
------------

TODO Custom document parsing (in lieu of `utils.simple_preprocess`). Different models (not just `lsi`). Optimizing the index with `service.optimize()`.
TODO add some hard numbers; example tutorial for some bigger collection, e.g. for `arxiv.org <http://aura.fi.muni.cz:8080/>`_ or wikipedia.

