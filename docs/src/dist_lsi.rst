.. _dist_lsi:

Distributed Latent Semantic Analysis
============================================


.. note::
  See :doc:`distributed` for an introduction to distributed computing in `gensim`.


Setting up the cluster
_______________________

We will show how to run distributed Latent Semantic Analysis by means of an example.
Let's say we have 5 computers at our disposal, all on the same network segment (=reachable
by network broadcast). To start with, install `gensim` and set up `Pyro` on each computer with::

  $ sudo easy_install gensim[distributed]
  $ export PYRO_SERIALIZERS_ACCEPTED=pickle
  $ export PYRO_SERIALIZER=pickle

Then run Pyro’s name server on exactly one of the machines (doesn’t matter which one)::

  $ python -m Pyro4.naming -n 0.0.0.0 &

Let's say our example cluster consists of dual-core computers with loads of
memory. We will therefore run **two** worker scripts on four of the physical machines,
creating **eight** logical worker nodes::

  $ python -m gensim.models.lsi_worker &

This will execute `gensim`'s `lsi_worker.py` script (to be run twice on each of the
four computer).
This lets `gensim` know that it can run two jobs on each of the four computers in
parallel, so that the computation will be done faster, while also taking up twice
as much memory on each machine.

Next, pick one computer that will be a job scheduler in charge of worker
synchronization, and on it, run `LSA dispatcher`. In our example, we will use the
fifth computer to act as the dispatcher and from there run::

  $ python -m gensim.models.lsi_dispatcher &

In general, the dispatcher can be run on the same machine as one of the worker nodes, or it
can be another, distinct computer (within the same broadcast domain). The dispatcher
won't be doing much with CPU most of the time, but pick a computer with ample memory.

And that's it! The cluster is set up and running, ready to accept jobs. To remove
a worker later on, simply terminate its `lsi_worker` process. To add another worker, run another
`lsi_worker` (this will not affect a computation that is already running, the additions/deletions are not dynamic).
If you terminate `lsi_dispatcher`, you won't be able to run computations until you run it again
(surviving worker processes can be re-used though).


Running LSA
____________

So let's test our setup and run one computation of distributed LSA. Open a Python
shell on one of the five machines (again, this can be done on any computer
in the same `broadcast domain <http://en.wikipedia.org/wiki/Broadcast_domain>`_,
our choice is incidental) and try::

    >>> from gensim import corpora, models, utils
    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> corpus = corpora.MmCorpus('/tmp/deerwester.mm') # load a corpus of nine documents, from the Tutorials
    >>> id2word = corpora.Dictionary.load('/tmp/deerwester.dict')

    >>> lsi = models.LsiModel(corpus, id2word=id2word, num_topics=200, chunksize=1, distributed=True) # run distributed LSA on nine documents

This uses the corpus and feature-token mapping created in the :doc:`tut1` tutorial.
If you look at the log in your Python session, you should see a line similar to::

  2010-08-09 23:44:25,746 : INFO : using distributed version with 8 workers

which means all went well. You can also check the logs coming from your worker and dispatcher
processes --- this is especially helpful in case of problems.
To check the LSA results, let's print the first two latent topics::

    >>> lsi.print_topics(num_topics=2, num_words=5)
    topic #0(3.341): 0.644*"system" + 0.404*"user" + 0.301*"eps" + 0.265*"time" + 0.265*"response"
    topic #1(2.542): 0.623*"graph" + 0.490*"trees" + 0.451*"minors" + 0.274*"survey" + -0.167*"system"

Success! But a corpus of nine documents is no challenge for our powerful cluster...
In fact, we had to lower the job size (`chunksize` parameter above) to a single document
at a time, otherwise all documents would be processed by a single worker all at once.

So let's run LSA on **one million documents** instead::

    >>> # inflate the corpus to 1M documents, by repeating its documents over&over
    >>> corpus1m = utils.RepeatCorpus(corpus, 1000000)
    >>> # run distributed LSA on 1 million documents
    >>> lsi1m = models.LsiModel(corpus1m, id2word=id2word, num_topics=200, chunksize=10000, distributed=True)

    >>> lsi1m.print_topics(num_topics=2, num_words=5)
    topic #0(1113.628): 0.644*"system" + 0.404*"user" + 0.301*"eps" + 0.265*"time" + 0.265*"response"
    topic #1(847.233): 0.623*"graph" + 0.490*"trees" + 0.451*"minors" + 0.274*"survey" + -0.167*"system"

The log from 1M LSA should look like::

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
of LSA still takes only 12 seconds. To really stress-test our cluster, let's do
Latent Semantic Analysis on the English Wikipedia.

Distributed LSA on Wikipedia
++++++++++++++++++++++++++++++

First, download and prepare the Wikipedia corpus as per :doc:`wiki`, then load
the corpus iterator with::

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary)
    >>> id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print(mm)
    MmCorpus(3199665 documents, 100000 features, 495547400 non-zero entries)

Now we're ready to run distributed LSA on the English Wikipedia::

    >>> # extract 400 LSI topics, using a cluster of nodes
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400, chunksize=20000, distributed=True)

    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.print_topics(10)
    2010-11-03 16:08:27,602 : INFO : topic #0(200.990): -0.475*"delete" + -0.383*"deletion" + -0.275*"debate" + -0.223*"comments" + -0.220*"edits" + -0.213*"modify" + -0.208*"appropriate" + -0.194*"subsequent" + -0.155*"wp" + -0.117*"notability"
    2010-11-03 16:08:27,626 : INFO : topic #1(143.129): -0.320*"diff" + -0.305*"link" + -0.199*"image" + -0.171*"www" + -0.162*"user" + 0.149*"delete" + -0.147*"undo" + -0.144*"contribs" + -0.122*"album" + 0.113*"deletion"
    2010-11-03 16:08:27,651 : INFO : topic #2(135.665): -0.437*"diff" + -0.400*"link" + -0.202*"undo" + -0.192*"user" + -0.182*"www" + -0.176*"contribs" + 0.168*"image" + -0.109*"added" + 0.106*"album" + 0.097*"copyright"
    2010-11-03 16:08:27,677 : INFO : topic #3(125.027): -0.354*"image" + 0.239*"age" + 0.218*"median" + -0.213*"copyright" + 0.204*"population" + -0.195*"fair" + 0.195*"income" + 0.167*"census" + 0.165*"km" + 0.162*"households"
    2010-11-03 16:08:27,701 : INFO : topic #4(116.927): -0.307*"image" + 0.195*"players" + -0.184*"median" + -0.184*"copyright" + -0.181*"age" + -0.167*"fair" + -0.162*"income" + -0.151*"population" + -0.136*"households" + -0.134*"census"
    2010-11-03 16:08:27,728 : INFO : topic #5(100.326): 0.501*"players" + 0.318*"football" + 0.284*"league" + 0.193*"footballers" + 0.142*"image" + 0.133*"season" + 0.119*"cup" + 0.113*"club" + 0.110*"baseball" + 0.103*"f"
    2010-11-03 16:08:27,754 : INFO : topic #6(92.298): -0.411*"album" + -0.275*"albums" + -0.217*"band" + -0.214*"song" + -0.184*"chart" + -0.163*"songs" + -0.160*"singles" + -0.149*"vocals" + -0.139*"guitar" + -0.129*"track"
    2010-11-03 16:08:27,780 : INFO : topic #7(83.811): -0.248*"wikipedia" + -0.182*"keep" + 0.180*"delete" + -0.167*"articles" + -0.152*"your" + -0.150*"my" + 0.144*"film" + -0.130*"we" + -0.123*"think" + -0.120*"user"
    2010-11-03 16:08:27,807 : INFO : topic #8(78.981): 0.588*"film" + 0.460*"films" + -0.130*"album" + -0.127*"station" + 0.121*"television" + 0.115*"poster" + 0.112*"directed" + 0.110*"actors" + -0.096*"railway" + 0.086*"movie"
    2010-11-03 16:08:27,834 : INFO : topic #9(78.620): 0.502*"kategori" + 0.282*"categoria" + 0.248*"kategorija" + 0.234*"kategorie" + 0.172*"категория" + 0.165*"categoría" + 0.161*"kategoria" + 0.148*"categorie" + 0.126*"kategória" + 0.121*"catégorie"

In serial mode, creating the LSI model of Wikipedia with this **one-pass algorithm**
takes about 5.25h on my laptop (OS X, C2D 2.53GHz, 4GB RAM with `libVec`).
In distributed mode with four workers (Linux, dual-core Xeons of 2Ghz, 4GB RAM
with `ATLAS`), the wallclock time taken drops to 1 hour and 41 minutes. You can
read more about various internal settings and experiments in my `research
paper <http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf>`_.

