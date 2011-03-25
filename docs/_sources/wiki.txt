.. _wiki:

Experiments on the English Wikipedia
============================================

To test `gensim` performance, we run it against the English version of Wikipedia.

This page describes the process of obtaining and processing Wikipedia, so that
anyone can reproduce the results. It is assumed you have `gensim` properly :doc:`installed <install>`.



Preparing the corpus
----------------------

1. First, download the dump of all Wikipedia articles from http://download.wikimedia.org/enwiki/
   (you want a file like `enwiki-latest-pages-articles.xml.bz2`). This file is about 6GB in size
   and contains (a compressed version of) all articles from the English Wikipedia.

2. Convert the articles to plain text (process Wiki markup) and store the result as
   sparse TF-IDF vectors. In Python, this is easy to do on-the-fly and we don't
   even need to uncompress the whole archive to disk. There is a script included in
   `gensim` that does just that, run::

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

As with Latent Semantic Analysis above, first load the corpus iterator and dictionary::

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary), one of the results of step 2 above
    >>> id2word = gensim.corpora.wikicorpus.WikiCorpus.loadDictionary('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print mm
    MmCorpus(3199665 documents, 100000 features, 495547400 non-zero entries)

We will run online LDA (see Hoffman et al. [3]_), which is an algorithm that takes a chunk of documents,
updates the LDA model, takes another chunk, updates the model etc. Online LDA can be contrasted
with batch LDA, which processes the whole corpus (one full pass), then updates
the model, then another pass, another update... The difference is that given a
reasonably stationary document stream (not much topic drift), the online updates
over the smaller chunks (subcorpora) are pretty good in themselves, so that the
model estimation converges faster. As a result, we will perhaps only need a single full
pass over the corpus: if the corpus has 3 million articles, and we update once after
every 10,000 articles, this means we will have done 300 updates in one pass, quite likely
enough to have a very accurate topics estimate::

    >>> # extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, numTopics=100, update_every=1, chunks=10000, passes=1)
    using serial LDA version on this node
    running online LDA training, 100 topics, 1 passes over the supplied corpus of 3146817 documets, updating model once every 10000 documents
    ..

Unlike LSA, the topics coming from LDA are easier to interpret::

    >>> # print the most contributing words for 20 randomly selected topics
    >>> lda.printTopics(20)
    topic #0: 0.065*india + 0.043*indian + 0.019*sri + 0.012*tamil + 0.011*singh + 0.011*lanka + 0.010*temple + 0.009*delhi + 0.007*pradesh + 0.007*bangladesh
    topic #1: 0.083*game + 0.053*games + 0.051*video + 0.018*player + 0.010*players + 0.007*playstation + 0.006*pc + 0.006*entertainment + 0.005*nintendo + 0.005*online
    topic #2: 0.096*church + 0.032*catholic + 0.022*bishop + 0.018*christian + 0.015*roman + 0.014*saint + 0.013*churches + 0.011*cathedral + 0.011*parish + 0.010*christ
    topic #3: 0.037*island + 0.019*islands + 0.019*sea + 0.015*coast + 0.013*storm + 0.012*tropical + 0.012*bay + 0.010*ocean + 0.008*hurricane + 0.007*pacific
    topic #4: 0.189*class + 0.055*assessed + 0.048*rev + 0.046*stub + 0.032*quality + 0.032*low + 0.026*start + 0.025*added + 0.022*importance + 0.013*rating
    topic #5: 0.061*italian + 0.039*italy + 0.031*di + 0.015*milan + 0.013*serie + 0.012*il + 0.009*rome + 0.008*la + 0.008*roma + 0.008*del
    topic #6: 0.017*project + 0.014*development + 0.012*organization + 0.011*services + 0.010*community + 0.010*organizations + 0.009*management + 0.008*program + 0.008*association + 0.007*service
    topic #7: 0.051*language + 0.025*culture + 0.024*admins + 0.017*categories + 0.016*languages + 0.016*native + 0.015*groups + 0.014*cultural + 0.011*ethnic + 0.010*african
    topic #8: 0.043*women + 0.035*children + 0.018*child + 0.018*age + 0.018*mother + 0.016*young + 0.014*woman + 0.014*female + 0.013*sex + 0.012*wife
    topic #9: 0.032*business + 0.024*companies + 0.012*bank + 0.012*management + 0.011*industry + 0.010*inc + 0.010*products + 0.009*services + 0.008*co + 0.008*founded
    topic #10: 0.035*season + 0.028*football + 0.024*basketball + 0.015*game + 0.014*coach + 0.011*player + 0.010*conference + 0.010*players + 0.009*college + 0.009*hockey
    topic #11: 0.270*user + 0.167*link + 0.082*added + 0.077*username + 0.070*www + 0.023*accounts + 0.023*records + 0.021*involved + 0.016*reporting + 0.012*report
    topic #12: 0.202*diff + 0.010*david + 0.008*michael + 0.006*paul + 0.006*james + 0.005*smith + 0.005*robert + 0.005*mark + 0.004*peter + 0.004*chris
    topic #13: 0.017*episode + 0.012*character + 0.011*characters + 0.010*man + 0.009*comics + 0.008*episodes + 0.008*fictional + 0.007*comic + 0.005*season + 0.005*show
    topic #14: 0.043*college + 0.035*students + 0.032*education + 0.029*schools + 0.016*student + 0.010*campus + 0.009*educational + 0.009*program + 0.008*elementary + 0.008*training
    topic #15: 0.085*ireland + 0.065*irish + 0.025*dublin + 0.022*northern + 0.013*mac + 0.012*cork + 0.011*galway + 0.011*patrick + 0.009*kelly + 0.008*belfast
    topic #16: 0.019*pennsylvania + 0.018*ohio + 0.018*virginia + 0.016*florida + 0.015*illinois + 0.015*chicago + 0.013*jersey + 0.013*michigan + 0.011*washington + 0.010*georgia
    topic #17: 0.061*air + 0.031*aircraft + 0.026*force + 0.024*airport + 0.017*squadron + 0.015*flight + 0.013*flying + 0.011*aviation + 0.010*wing + 0.008*pilot
    topic #18: 0.064*california + 0.038*texas + 0.036*san + 0.022*los + 0.020*angeles + 0.016*oregon + 0.015*washington + 0.015*wisconsin + 0.014*colorado + 0.013*francisco
    topic #19: 0.021*anime + 0.020*manga + 0.017*dragon + 0.012*theme + 0.011*dvd + 0.011*super + 0.011*hunter + 0.009*ash + 0.009*dream + 0.009*angel

Creating this LDA model of Wikipedia takes about 11 hours on my laptop [1]_.
If you need your results faster, consider running :doc:`dist_lda` on a cluster of
computers.

Note two differences between the LDA and LSA runs: we asked LSA
to extract 400 topics, LDA only 100 topics (so the difference in speed is in fact
even greater). Secondly, the LSA implementation in `gensim` is truly online: if the nature of the input
stream changes in time, LSA will re-orient itself to reflect these changes, in a reasonably
small amount of updates. In contrast, LDA is not truly online (the name of the [3]_
article notwithstanding), as the impact of later updates on the model gradually
diminishes. If there is topic drift in the input document stream, LDA will get
confused and be increasingly slower at adjusting itself to the new state of affairs.

In short, be careful if using LDA to incrementally add new documents to the model
over time. **Batch usage of LDA**, where the entire training corpus is either known beforehand or does
not exihibit topic drift, **is ok and not affected**.

To run batch LDA (not online), train `LdaModel` with::

    >>> # extract 100 LDA topics, using 20 full passes, no online updates
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, numTopics=100, update_every=0, passes=20)

As usual, a trained model can used be to transform new, unseen documents (plain bag-of-words count vectors)
into LDA topic distributions:

    >>> doc_lda = lda[doc_bow]

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

.. [3] Hoffman, Blei, Bach. 2010. Online learning for Latent Dirichlet Allocation
   [`pdf <http://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_] [`code <http://www.cs.princeton.edu/~mdhoffma/>`_]

