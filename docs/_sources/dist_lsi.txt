.. _dist_lsi:

Distributed Latent Semantic Analysis
============================================


.. note::
  See :doc:`distributed` for an introduction to distributed computing in `gensim`.


Setting up the cluster
_______________________

We will show how to run distributed Latent Semantic Analysis by means of an example. 
Let's say we have 5 computers at our disposal, all in the same broadcast domain.
To start with, install `gensim` and `Pyro` on each one of them with::

  $ sudo easy_install gensim[distributed]

and run Pyro's name server on exactly *one* of the machines (doesn't matter which one)::

  $ python -m Pyro.naming &

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
can be another, distinct computer within the same broadcast domain. The dispatcher
won't be  doing much with CPU most of the time, but pick a computer with ample memory.

And that's it! The cluster is set up and running, ready to accept jobs. To remove
a worker later on, simply terminate its `lsi_worker` process. To add another worker, run another
`lsi_worker` (this will not affect a computation that is already running). If you terminate
`lsi_dispatcher`, you won't be able to run computations until you run it again on 
some node (surviving workers can be re-used though).


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
    >>> id2word = corpora.Dictionary.load('/tmp/deerwester.dict').id2token

    >>> lsi = models.LsiModel(corpus, id2word, numTopics=200, chunks=1, distributed=True) # run distributed LSA on nine documents

This uses the corpus and feature-token mapping created in the :doc:`tut1` tutorial.
If you look at the log in your Python session, you should see a line similar to::

  2010-08-09 23:44:25,746 : INFO : using distributed version with 8 workers

which means all went well. You can also check the logs coming from your worker and dispatcher
processes --- this is especially helpful in case of problems.
To check the LSA results, let's print the first two latent topics::

    >>> lsi.printTopics(numTopics=2, numWords=5)
    0.644 * "survey" + 0.404 * "response" + 0.301 * "user" + 0.265 * "time" + 0.265 * "system"
    0.623 * "graph" + 0.490 * "trees" + 0.451 * "minors" + 0.274 * "eps" + -0.167 * "survey"

Success! But a corpus of nine documents is no challenge for our powerful cluster...
In fact, we had to lower the job size (`chunks` parameter above) to a single document 
at a time, otherwise all documents would be processed by a single worker all at once.

So let's run LSA on **one million documents** instead::

    >>> # inflate the corpus to 1M documents, by repeating it over&over
    >>> corpus1m = utils.RepeatCorpus(corpus, 1000000) 
    >>> # run distributed LSA on 1 million documents
    >>> lsi1m = models.LsiModel(corpus1m, id2word, numTopics=200, chunks=10000, distributed=True)

    >>> lsi1m.printTopics(numTopics=2, numWords=5)
    -0.644 * "survey" + -0.404 * "response" + -0.301 * "user" + -0.265 * "time" + -0.265 * "system
    0.623 * "graph" + 0.490 * "trees" + 0.451 * "minors" + 0.274 * "eps" + -0.167 * "survey"

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

First, download and prepare the Wikipedia corpus as per :doc:`wiki`, then load 
the corpus iterator with::
 
    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level logging.INFO)
    
    >>> # load id->word mapping (the dictionary)
    >>> id2word = gensim.corpora.wikicorpus.WikiCorpus.loadDictionary('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print mm
    MmCorpus(3199665 documents, 100000 features, 495547400 non-zero entries)

Now we're ready to run distributed LSA on the English Wikipedia::

    >>> # extract 400 LSI topics
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, numTopics=400, chunks=40000, distributed=True)
    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.printDebug(numTopics=10)
    2010-09-01 10:12:55,058 : INFO : computing word-topic salience for 10 topics
    2010-09-01 10:13:13,644 : INFO : topic #0(199.154): debate(0.274), subsequent(0.194), deletion(0.381), appropriate(0.208), proposed(0.103), modify(0.213), comments(0.223), edits(0.220), delete(0.478), archived(0.109), ..., 
    2010-09-01 10:13:14,482 : INFO : topic #1(141.388): diff(0.238), link(0.226), together(0.023), every(0.034), image(0.234), become(0.015), fair(0.124), ever(0.012), alone(0.019), changing(0.004), ..., debate(-0.089), subsequent(-0.062), deletion(-0.116), appropriate(-0.066), modify(-0.068)
    2010-09-01 10:13:15,323 : INFO : topic #2(132.771): diff(0.486), link(0.437), undo(0.221), added(0.124), resolves(0.065), blacklist(0.073), linkwatcher(0.062), reporting(0.060), reports(0.052), spamming(0.059), ..., together(-0.017), age(-0.087), alone(-0.017), older(-0.030), every(-0.027)
    2010-09-01 10:13:16,163 : INFO : topic #3(125.087): age(0.243), residing(0.055), older(0.086), household(0.103), householder(0.056), capita(0.056), median(0.221), families(0.135), versus(0.051), households(0.164), ..., fair(-0.193), image(-0.349), licensing(-0.102), resolution(-0.123), copyright(-0.210)
    2010-09-01 10:13:17,007 : INFO : topic #4(116.531): householder(0.045), residing(0.043), median(0.177), capita(0.045), household(0.080), households(0.130), makeup(0.044), versus(0.039), families(0.102), income(0.155), ..., ever(-0.012), six(-0.017), become(-0.014), coming(-0.007), twice(-0.007)
    2010-09-01 10:13:17,842 : INFO : topic #5(99.225): players(0.500), goalkeepers(0.009), goalkeeper(0.015), football(0.323), defender(0.017), forwards(0.016), league(0.286), midfielder(0.019), striker(0.012), fullbacks(0.003), ..., everything(-0.008), come(-0.014), gone(-0.006), nowhere(-0.002), listen(-0.003)
    2010-09-01 10:13:18,695 : INFO : topic #6(91.274): lyrically(0.002), album(0.414), rerecorded(0.001), daydream(0.001), tracklisting(0.006), catchy(0.002), anthemic(0.000), charting(0.008), studio(0.051), musically(0.003), ..., initiated(-0.003), consult(-0.001), necessary(-0.008), existing(-0.007), strongly(-0.004)
    2010-09-01 10:13:19,559 : INFO : topic #7(83.057): why(0.066), regards(0.009), else(0.027), understand(0.026), occured(0.001), gotten(0.004), honestly(0.005), noticed(0.009), reply(0.010), need(0.065), ..., subsequent(-0.071), debate(-0.096), proposed(-0.037), delete(-0.181), appropriate(-0.066)
    2010-09-01 10:13:20,412 : INFO : topic #8(78.190): 분류(0.097), kategória(0.117), kategori(0.471), categoría(0.153), kategoria(0.151), ja(0.093), категория(0.161), kategorie(0.219), categorie(0.139), kategorija(0.234), ..., formerly(-0.003), serving(-0.006), officially(-0.002), station(-0.061), newly(-0.001)
    2010-09-01 10:13:21,246 : INFO : topic #9(78.150): film(0.556), directorial(0.007), directed(0.104), remake(0.007), cassavetes(0.001), theatrically(0.003), ebert(0.004), projectionist(0.001), starring(0.046), films(0.427), ..., 분류(-0.036), kategória(-0.044), kategori(-0.175), categoría(-0.057), kategoria(-0.056)


In serial mode, creating the LSI model of Wikipedia with this one-pass algorithm 
takes about 8.5h on my laptop (OS X, dual-core 2.53GHz, 4GB RAM with `libVec`). 
In distributed mode with six workers (Linux, dual-core Xeons of 2Ghz, 4GB RAM 
with `ATLAS`), the wallclock time taken drops to 2 hours and 23 minutes.
