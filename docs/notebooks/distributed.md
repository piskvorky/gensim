Distributed Computing
=====================

Why distributed computing?
--------------------------

Need to build semantic representation of a corpus that is millions of documents large and it's taking forever? Have several idle machines at your disposal that you could use? [Distributed computing][1] tries to accelerate computations by splitting a given task into several smaller subtasks, passing them on to several computing nodes in parallel.

In the context of gensim, computing nodes are computers identified by their IP address/port, and communication happens over TCP/IP. The whole collection of available machines is called a *cluster*. The distribution is very coarse grained (not much communication going on), so the network is allowed to be of relatively high latency.

> **Warning**

> The primary reason for using distributed computing is making things run faster. In gensim, most of the time consuming stuff is done inside low-level routines for linear algebra, inside NumPy, independent of any gensim code. **Installing a fast** [BLAS (Basic Linear Algebra)][2] **library for NumPy can improve performance up to 15 times!** So before you start buying those extra computers, consider installing a fast, threaded BLAS that is optimized for your particular machine (as opposed to a generic, binary-distributed library). Options include your vendor's BLAS library (Intel's MKL, AMD's ACML, OS X's vecLib, Sun's Sunperf, ...) or some open-source alternative (GotoBLAS, ALTAS).
>
> To see what BLAS and LAPACK you are using, type into your shell:
>
>     $ python -c 'import scipy; scipy.show_config()'

Prerequisites
-------------

For communication between nodes, gensim uses [Pyro (PYthon Remote Objects)][3], version &gt;= `4.27`. This is a library for low-level socket communication and remote procedure calls (RPC) in Python. Pyro is a pure-Python library, so its installation is quite painless and only involves copying its `*.py` files somewhere onto your Python's import path:

    sudo easy_install Pyro4

You don't have to install Pyro to run gensim, but if you don't, you won't be able to access the distributed features (i.e., everything will always run in serial mode, the examples on this page don't apply).

Core concepts
-------------

As always, gensim strives for a clear and straightforward API (see [Features][4]). To this end, *you do not need to make any changes in your code at all* in order to run it over a cluster of computers!

What you need to do is run a [worker][5] script (*see below*) on each of your cluster nodes prior to starting your computation. Running this script tells gensim that it may use the node as a slave to delegate some work to it. During initialization, the algorithms inside gensim will try to look for and enslave all available worker nodes.

**Node**
A logical working unit. Can correspond to a single physical machine, but you can also run multiple workers on one machine, resulting in multiple logical nodes.

**Cluster**
Several nodes which communicate over TCP/IP. Nodes can lie in different [broadcast domains][6]. Here is how to [configure](https://github.com/RaRe-Technologies/gensim/pull/782).

**Worker**
A process which is created on each node. To remove a node from your cluster, simply kill its worker process.

**Dispatcher**
The dispatcher will be in charge of negotiating all computations, queueing and distributing ("dispatching") individual jobs to the workers. Computations never "talk" to worker nodes directly, only through this dispatcher. Unlike workers, there can only be one active dispatcher at a time in the cluster.

Available distributed algorithms
---------------------------------
* [Distributed Latent Semantic Analysis][7]
* [Distributed Latent Dirichlet Allocation][8]


[1]: http://en.wikipedia.org/wiki/Distributed_computing
[2]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[3]: http://pypi.python.org/pypi/Pyro4
[4]: http://radimrehurek.com/gensim/intro.html#design
[5]: http://radimrehurek.com/gensim/distributed.html#term-worker
[6]: http://en.wikipedia.org/wiki/Broadcast_domain
[7]: http://radimrehurek.com/gensim/dist_lsi.html
[8]: http://radimrehurek.com/gensim/dist_lda.html
