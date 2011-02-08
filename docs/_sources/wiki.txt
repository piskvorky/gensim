.. _wiki:

Experiments on Wikipedia 
============================================

To test `gensim` performance, we run it against the English version of Wikipedia.

This page describes the process of obtaining and processing Wikipedia, so that
anyone can reproduce our results. It is assumed you have `gensim` properly :doc:`installed <install>`.



Preparing the corpus
----------------------

1. First, download the dump of all Wikipedia articles from http://download.wikimedia.org/enwiki/ 
   (you want a file like `enwiki-latest-pages-articles.xml.bz2`). This file is about 6GB in size
   and contains (compressed version of) all articles from the English Wikipedia.

2. Convert the articles to plain text (process Wiki markup) and store the result as 
   sparse TF-IDF vectors. In Python, this is easy to do on-the-fly (the code is included in `gensim`), and we don't 
   even need to uncompress the whole archive to disk::

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

    >>> # extract 400 LSI topics; use the default one-pass algorithm
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, numTopics=400)
    
    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.printTopics(10)
    topic #0(200.540): 0.475*"delete" + 0.383*"deletion" + 0.275*"debate" + 0.223*"comments" + 0.221*"edits" + 0.213*"modify" + 0.208*"appropriate" + 0.195*"subsequent" + 0.155*"wp" + 0.116*"notability"
    topic #1(142.463): -0.292*"diff" + -0.277*"link" + -0.210*"image" + -0.160*"www" + 0.151*"delete" + -0.149*"user" + -0.134*"contribs" + -0.133*"undo" + -0.128*"album" + -0.115*"copyright"
    topic #2(134.758): -0.458*"diff" + -0.415*"link" + -0.210*"undo" + -0.201*"user" + -0.195*"www" + -0.186*"contribs" + 0.154*"image" + -0.115*"added" + 0.098*"album" + -0.096*"accounts"
    topic #3(124.902): 0.352*"image" + -0.240*"age" + -0.219*"median" + 0.212*"copyright" + -0.205*"population" + -0.196*"income" + 0.194*"fair" + -0.168*"census" + -0.166*"km" + -0.162*"households"
    topic #4(116.735): 0.309*"image" + -0.195*"players" + 0.185*"copyright" + 0.182*"median" + 0.178*"age" + 0.169*"fair" + 0.159*"income" + 0.149*"population" + 0.134*"households" + -0.133*"football"
    topic #5(100.069): -0.502*"players" + -0.317*"football" + -0.284*"league" + -0.194*"footballers" + -0.142*"image" + -0.132*"season" + -0.118*"cup" + -0.113*"club" + -0.110*"baseball" + 0.103*"album"
    topic #6(92.077): 0.411*"album" + 0.275*"albums" + 0.217*"band" + 0.214*"song" + 0.184*"chart" + 0.163*"songs" + 0.160*"singles" + 0.149*"vocals" + 0.139*"guitar" + 0.129*"track"
    topic #7(83.568): 0.248*"wikipedia" + 0.184*"keep" + -0.180*"delete" + 0.167*"articles" + 0.152*"your" + 0.150*"my" + -0.141*"film" + 0.130*"we" + 0.123*"think" + 0.121*"user"
    topic #8(78.753): 0.485*"kategori" + 0.273*"categoria" + 0.240*"kategorija" + 0.226*"kategorie" + 0.167*"категория" + 0.160*"categoría" + 0.156*"kategoria" + 0.146*"film" + 0.143*"categorie" + 0.122*"kategória"
    topic #9(78.701): -0.570*"film" + -0.444*"films" + 0.128*"kategori" + 0.126*"album" + 0.120*"station" + -0.118*"television" + -0.113*"poster" + -0.108*"directed" + -0.107*"actors" + 0.091*"railway"

Creating the LSI model of Wikipedia takes about 5 hours and 14 minutes on my laptop [1]_.
If you need your results even faster, see the tutorial on :doc:`distributed`.

We see that the total processing time is dominated by the preprocessing step of 
preparing the TF-IDF corpus, which took 15h. [2]_

The algorithm used in `gensim` only needs to see each input document once, so it 
is suitable for environments where the documents come as a non-repeatable stream, 
or where the cost of storing/iterating over the corpus multiple times is too high.


Latent Dirichlet Allocation
----------------------------

*Coming soon.*
 

--------------------

.. [1] My laptop = Core2Duo 2.53GHz MacBook Pro, 4GB RAM, OS X with `libVec`.

.. [2]
  Here we're mostly interested in performance, but it is interesting to look at the
  retrieved LSA concepts, too. I am no Wikipedia expert and don't see into Wiki's bowels, 
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
    
  So the top ten concepts are apparently dominated by Wikipedia robots and expanded 
  templates; this is a good reminder that LSA is a powerful tool for data analysis, 
  but no silver bullet. As always, it's `garbage in, garbage out 
  <http://en.wikipedia.org/wiki/Garbage_In,_Garbage_Out>`_...
  By the way, improvements to the Wiki markup parsing code are welcome :-)

