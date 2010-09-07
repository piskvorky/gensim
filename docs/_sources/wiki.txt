.. _wiki:

Experiments on Wikipedia 
============================================

To test `gensim` performance, we run it against the English version of Wikipedia.

This page describes the process of obtaining and processing Wikipedia, so that
anyone can reproduce our results.


Preparing the corpus
_______________________

1. First, download the dump of all Wikipedia articles from http://download.wikimedia.org/enwiki/ 
   (you want a file like `enwiki-latest-pages-articles.xml.bz2`). This file is about 6GB in size
   and contains (compressed version of) all articles from the English Wikipedia.

2. Convert the articles to plain text (process Wiki markup) and store the result as 
   sparse TF-IDF vectors. In Python, this is easy to do on-the-fly and we don't 
   even need to uncompress the whole archive to disk. We will use the fact that 
   the computation can be run incrementally, over the compressed file, when testing 
   the single pass LSA algorithm. 
   We will however also store the resulting TF-IDF vectors to disk, 
   because this reduces time for multi-pass algorithms and is the most common 
   scenario in practise::

   $ python -m gensim.corpora.wikicorpus

.. note::
  This pre-processing step makes two passes over the 6GB wiki dump (one to extract 
  the dictionary, one to create and store the sparse vectors) and takes about 
  15 hours on my laptop, so you may want to go have a coffee or two. 
  Also, you will need about 15GB of free disk space to store the sparse output vectors.

Latent Sematic Analysis
--------------------------
 
First let's load the corpus iterator and dictionary, created in the second step above::

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary), one of the results of step 2 above
    >>> id2word = gensim.corpora.wikicorpus.WikiCorpus.loadDictionary('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print mm
    MmCorpus(3199665 documents, 100000 features, 495547400 non-zero entries)

We see that our corpus contains 3.2M documents, 100K features (distinct 
tokens) and 0.5G non-zero entries in the sparse TF-IDF matrix. The corpus contains
about 1.92 billion tokens in total.

Now we're ready to compute LSA of the English Wikipedia::

    >>> # extract 400 LSI topics; by default, this uses the two-pass algorithm with double precision and oversampling
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, numTopics=400)
    
    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.printDebug(numTopics=10)
    computing word-topic salience for 10 topics
    topic #0(191.470): deletion(0.369), subsequent(0.189), debate(0.265), appropriate(0.200), comments(0.217), modify(0.206), edits(0.213), delete(0.456), proposed(0.098), archived(0.106), ..., rajah(-0.003), ribbing(-0.001), wess(-0.002), abdel(-0.003), hazy(-0.002)
    topic #1(131.265): diff(0.200), link(0.191), image(0.214), fair(0.113), copyright(0.126), licensing(0.057), undo(0.093), age(0.091), resolution(0.068), males(0.052), ..., subsequent(-0.062), deletion(-0.113), modify(-0.068), appropriate(-0.064), debate(-0.083)
    topic #2(122.543): diff(0.454), link(0.409), undo(0.203), resolves(0.059), linkwatcher(0.056), blacklist(0.066), spamming(0.054), imply(0.051), added(0.122), coibot(0.091), ..., age(-0.081), females(-0.049), males(-0.048), householder(-0.016), households(-0.046)
    topic #3(114.196): median(0.208), households(0.154), householder(0.052), males(0.147), females(0.147), income(0.187), age(0.231), household(0.097), residing(0.052), families(0.126), ..., image(-0.316), fair(-0.173), resolution(-0.112), copyright(-0.183), licensing(-0.087)
    topic #4(106.069): median(0.154), households(0.113), householder(0.038), females(0.109), males(0.108), income(0.136), household(0.071), residing(0.037), makeup(0.041), capita(0.037), ..., players(-0.196), football(-0.135), playing(-0.038), league(-0.104), season(-0.075)
    topic #5(86.145): players(0.442), football(0.289), league(0.247), footballers(0.178), footballer(0.059), season(0.109), player(0.088), f(0.103), cup(0.102), professional(0.047), ..., charts(-0.029), don(-0.037), songs(-0.057), stop(-0.020), band(-0.081)
    topic #6(78.923): album(0.360), albums(0.233), songs(0.142), track(0.118), listing(0.073), singles(0.142), song(0.182), chart(0.156), band(0.187), vocals(0.123), ..., serving(-0.022), provide(-0.021), complex(-0.021), leader(-0.022), areas(-0.023)
    topic #7(69.907): think(0.096), wikipedia(0.219), why(0.058), really(0.056), your(0.124), ve(0.068), my(0.118), sure(0.045), don(0.078), articles(0.152), ..., debate(-0.090), film(-0.148), subsequent(-0.057), proposed(-0.035), delete(-0.138)
    topic #8(65.305): kategori(0.403), kategorija(0.200), categoría(0.132), kategorie(0.187), 분류(0.082), categoria(0.225), kategoria(0.132), catégorie(0.098), категория(0.137), categorie(0.118), ..., bathed(-0.003), cautioned(-0.003), scooping(-0.002), matheson(-0.008), texans(-0.011)
    topic #9(63.094): film(0.386), films(0.293), directed(0.076), cast(0.048), starring(0.032), movie(0.049), actors(0.070), actor(0.053), drama(0.046), filmography(0.025), ..., z(-0.219), bytes(-0.218), contribs(-0.200), cydebot(-0.041), gif(-0.042)

Creating the LSI model of Wikipedia takes about 2.3h on my laptop (OS X, 
dual-core 2.53GHz, 4GB RAM with `libVec`), 
most of which is spent converting the sparse vectors from disk to a suitable 
representation in memory. The total processing time is therefore dominated by 
the preprocessing step of preparing the corpus, which took 15h. [1]_


One-pass Latent Semantic Analysis
------------------------------------

In the previous experiment, we had the whole corpus available beforehand, which allowed
us to make two passes over it and use the fast algorithm. But what if our documents
come as a non-repeatable stream, or what if the additional cost of iterating over the corpus 
multiple times is too high? This is not a very common scenario, but when it happens,
we need an algorithm that only makes a single one pass 
over the data, i.e., visits each document at most once.

We will simulate this scenario by running Latent Semantic Analysis directly 
over the `bzip`'d Wikipedia dump, converting each article to a sparse vector on
the fly as it is extracted from the archive, to be processed and then immediately 
forgotten::

    >>> import gensim
    >>> from gensim.corpora import wikicorpus
    
    >>> dictionary = wikicorpus.WikiCorpus.loadDictionary('wiki_en_wordids.txt', False)
    >>> wiki = wikicorpus.WikiCorpus("enwiki-latest-pages-articles.xml.bz2", dictionary=dictionary)
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=wiki, id2word=dictionary.id2word, numTopics=400, onepass=True)

There are two differences to how we ran LSA in the previous experiment:

1. With a single pass, we must know the feature set in advance; we cannot afford an extra pass over 
   the corpus just to collect statistics about word frequencies etc. Here we cheat
   a little and use the dictionary used in the two-pass algo; in reality, the feature
   set would need to be fixed in advance (or some dynamic feature hashing scheme used).
2. We cannot use TF-IDF weighting, because that again relies on the knowledge of
   corpus statistics (the number of documents, word document frequencies), which 
   are unknown in advance. We therefore run LSA directly on the raw bag-of-words counts.

Running the above takes about 22.5h on my laptop (OS X, dual-core 2.53GHz, 4GB 
RAM with `libVec`). This is comparable to the `15h+2.3h~=17.3h` for the two-phase 
algorithm above, where we first store the sparse vectors to disk (15h) and only then 
proceed with LSA (2.3h). In general, the two-pass algorithm is faster and preferrable 
wherever you can store your input vectors persistently.

If you need your results faster, the one-pass algorithm can be run over a cluster 
of computers, see the :doc:`distributed tutorial <distributed>`.

.. note::

  Actually, the two-pass algorithm can be easily parallelized, too. But the processing
  itself is blazing fast and most time is already spent on I/O conversions, 
  so the extra effort of sending data around over the network actually **slows 
  the algorithm down**. For this reason, there is no distributed variant of the two-pass algo.

--------------------

.. [1]
  Here we're mostly interested in performance, but it is interesting to look at the
  retrieved LSA concepts, too. I am no expert and don't see into Wiki's bowels, 
  but Brian Mingus had this to say about the result::

    There appears to be a lot of noise in your dataset. The first three topics
    in your list appear to be meta topics, concerning the administration and
    cleanup of Wikipedia. These show up because you didn't exclude templates
    such as these, some of which are included in most articles for quality
    control: http://en.wikipedia.org/wiki/Wikipedia:Template_messages/Cleanup
    
    The fourth and fifth topics clearly shows the influence of bots that import
    massive databases of cities, countries, etc. and their statistics such as
    population, capita, etc.
    
    The sixth shows the influence of sports bots, and the seventh of music bots.
    
  So the top ten concepts are apparently dominated by Wikipedia robots and templates; this is
  a good reminder that LSA is a powerful tool for data analysis, but not a magic pill. 
  As always, it's `garbage in, garbage out <http://en.wikipedia.org/wiki/Garbage_In,_Garbage_Out>`_...
  Btw, improvements to the Wiki markup parsing code are welcome :-)
