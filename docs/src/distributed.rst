.. _distributed:

Distributed Computing
===================================

Why distributed computing?
---------------------------

Need to build semantic representation of a corpus that is millions of documents large and it's
taking forever? Have several idle machines at your disposal that you could use? 
`Distributed computing <http://en.wikipedia.org/wiki/Distributed_computing>`_ tries 
to accelerate computations by splitting a given task into several smaller subtasks,
passing them on to several computing nodes in parallel.

In the context of `gensim`, computing nodes are computers identified by their IP address/port,
and communication happens over TCP/IP. The whole collection of available machines is called
a *cluster*. The distribution is very coarse grained (not
much communication going on), so the network is allowed to be of relatively high latency.

.. warning::
  The primary reason for using distributed computing is making things run faster. In `gensim`,
  most of the time consuming stuff is done inside low-level routines for linear algebra, inside
  NumPy, independent of any `gensim` code.
  **Installing a fast** `BLAS (Basic Linear Algebra) <http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ **library
  for NumPy can improve performance up to 8 times!** So before you start buying those extra computers, 
  consider installing a fast, threaded BLAS first. Options include your vendor's BLAS library (Intel's MKL, 
  AMD's ACML, OS X's vecLib, Sun's Sunperf, ...) or some open-source alternative (GotoBLAS, ALTAS).

  To see what BLAS and LAPACK your NumPy's using, type into your shell::
  
    python -c 'import numpy; numpy.show_config()' 

Distributed computing in `gensim`
-----------------------------------

As always, `gensim` strives for a clear and straightforward API (see :ref:`design`).
To this end, *you do not need to make any changes in your code at all* in order to
run it in a distributed manner!

What you need to do is run a worker script (see below) on each of your cluster nodes prior
to starting your computation. Running this script tells `gensim` that it may use the node
as a slave to delegate some work to it. During initialization, the algorithms
inside `gensim` will automatically try to look for and enslave all available worker nodes.
If at least one worker is found, things will run in the distributed mode; if not, in serial node.

To remove a node from your cluster, simply kill its worker script process.

For communication between nodes, `gensim` uses `Pyro (PYthon Remote Objects)
<http://pypi.python.org/pypi/Pyro>`_, version >= 4.1. This is a library for low-level socket communication
and remote procedure calls (RPC) in Python. `Pyro` is a pure-Python library, so installation
is quite painless and only involves copying its `*.py` files somewhere onto your Python's import path::

  sudo easy_install Pyro

You don't have to install `Pyro` to run `gensim`, but if you don't, you won't be able
to access the distributed features (i.e., everything will always run in serial mode).
Currently, network broadcasting is used to discover and connect all communicating nodes,
so the nodes must lie within the same `broadcast domain <http://en.wikipedia.org/wiki/Broadcast_domain>`_.

Available distributed algorithms
---------------------------------

Currently, there is only distributed *Latent Semantic Analysis (LSA, LSI)*. Distributed *Latent Dirichlet Allocation* is on its way.

Distributed LSA
+++++++++++++++++

We will show how to run distributed Latent Semantic Analysis on an example. Let's say 
we have 5 computers at our disposal, all in the same broadcast domain.
To start with, install `gensim` and `Pyro` on each one of them with::

  $ sudo easy_install gensim[distributed]

Let's say our example cluster consists of five dual-core computers with loads of 
memory. We will therefore run **two** worker scripts on four of the physical machines, 
creating **eight** logical worker nodes::

  $ python -m gensim.models.lsi_worker &

This will execute `gensim`'s `lsi_worker.py` script, to be run twice on each computer.
This lets `gensim` know that it can run two jobs on each of the four computers in 
parallel, so that the computation will be done faster (but also taking up twice 
as much memory on each machine).

Next, pick one computer that will be a job scheduler, in charge of worker 
synchronization, and on it, start Pyro's name server and an `LSA dispatcher`::

  $ python -m Pyro.naming &
  $ python -m gensim.models.lsi_dispatcher &

The dispatcher can be run on the same machine as one of the worker nodes, or it 
can be another, distinct computer within the same broadcast domain.
The dispatcher will be in charge of negotiating all computations, queueing and 
distributing ("dispatching") individual jobs to the workers (not doing much with 
CPU most of the time), so pick a computer with ample memory. Computations never
"talk" to worker nodes directly, only through the dispatcher.

In our example, we will use the fifth computer to act as the dispatcher and run 
the `lsi_dispatcher` and `Pyro.naming` scripts from there.

And that's it! The cluster is set up and running, ready to accept jobs. To remove
a worker later on, simply terminate its `lsi_worker` process. To add another worker, run another
`lsi_worker` (this will not affect a computation that is already running). If you terminate
`lsi_dispatcher`, you won't be able to run computations until you run it again on 
some node (surviving workers can be re-used though).

So let's test our setup and run one computation of distributed LSA. Open a Python 
shell on a worker node (again, this can be done on any computer
in the same `broadcast domain <http://en.wikipedia.org/wiki/Broadcast_domain>`_, 
our choice is incidental) and try::

>>> from gensim import corpora, models, utils
>>> import logging
>>> logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
>>> 
>>> corpus = corpora.MmCorpus('/tmp/deerwester.mm') # load a corpus of nine documents, from the Tutorials
>>> id2word = corpora.Dictionary.load('/tmp/deerwester.dict').id2token
>>>
>>> lsi = models.LsiModel(corpus, id2word, numTopics = 200, chunks = 1) # run distributed LSA on nine documents

This uses the corpus and feature-token mapping created in the :doc:`tut1` tutorial.
If you look at the log in your Python session, you should see a line similar to::

  2010-08-09 23:44:25,746 : INFO : using distributed version with 8 workers

which means all went well. You can also check the logs coming from your worker and dispatcher
processes --- this is especially helpful in case of problems.
To check the LSA results, let's print the first two latent topics:

>>> for i in xrange(2): lsi.printTopic(i, topN = 5)
0.644 * "survey" + 0.404 * "response" + 0.301 * "user" + 0.265 * "time" + 0.265 * "system"
0.623 * "graph" + 0.490 * "trees" + 0.451 * "minors" + 0.274 * "eps" + -0.167 * "survey"

Success! But a corpus of nine documents is no challenge for our powerful cluster...
In fact, we had to lower the job size (`chunks` parameter) to a single document 
at a time, otherwise all documents would be processed at once by a single worker.

So let's run LSA on **one million documents** instead::

>>> corpus1m = utils.RepeatCorpus(corpus, 1000000) # inflate the corpus to 1M documents, by repeating it over&over
>>> lsi1m = models.LsiModel(corpus1m, id2word, numTopics = 200, serial_only = False) # run distributed LSA on 1 million documents!

>>> for i in xrange(2): lsi1m.printTopic(i, topN = 5)
-0.644 * "survey" + -0.404 * "response" + -0.301 * "user" + -0.265 * "time" + -0.265 * "system
0.623 * "graph" + 0.490 * "trees" + 0.451 * "minors" + 0.274 * "eps" + -0.167 * "survey"

The `serial_only` parameter instructs `gensim` whether to run in serial or distributed mode.
Setting it to `True` will result in LSA running inside the active Python shell, without
any inter-node communication whatsoever, even if there are worker nodes available. 
Setting `serial_only=False` forces distributed mode (raising an exception in
case of failure). And finally, leaving `serial_only` unspecified tells `gensim`
to try running in distributed mode, or, failing that, run in serial mode.

On my Macbook (all 8 "distributed" workers operating on a single physical machine), 
the log from 1M LSA looks like::

  2010-08-10 02:46:35,087 : INFO : using distributed version with 8 workers
  2010-08-10 02:46:35,087 : INFO : updating SVD with new documents
  2010-08-10 02:46:35,202 : INFO : dispatched documents up to #10000
  2010-08-10 02:46:35,296 : INFO : dispatched documents up to #20000
  ...
  2010-08-10 02:46:46,524 : INFO : dispatched documents up to #990000
  2010-08-10 02:46:46,694 : INFO : dispatched documents up to #1000000
  2010-08-10 02:46:46,694 : INFO : reached the end of input; now waiting for all remaining jobs to finish
  2010-08-10 02:46:47,195 : INFO : all jobs finished, downloading final projection
  2010-08-10 02:46:47,200 : INFO : decomposition complete

Due to the small vocabulary size and trivial structure of our "one-million corpus", the computation
of LSA still takes only 12 seconds. To really stress-test our cluster, Wikipedia FIXME TODO.


