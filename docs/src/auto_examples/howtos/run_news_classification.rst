.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_howtos_run_news_classification.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_howtos_run_news_classification.py:


How to Get Topics from News Articles
====================================

Demonstrates topic modeling on the Lee Corpus using a variety of topic models (LSI, HDP, LDA, etc).


News article classification is performed on a huge scale by news agencies all
over the world. We will be looking into how topic modeling can be used to
accurately classify news articles into different categories such as sports,
technology, politics etc.

This guide demonstrates training a topic model which can come up with topics
that can easily be interpreted by us. On top of assigning a topic to an
arbitrary document, this model can also discover hidden structure in the
corpus.

We will be using the Lee corpus which is a shortened version of the `Lee
Background Corpus
<http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF>`_. The
shortened version consists of 300 documents selected from the Australian
Broadcasting Corporation's news mail service. It consists of texts of
headline stories from around the year 2000-2001.

We will examine the following models:

- LSI (Latent Semantic Indexing)
- HDP (Hierarchical Dirichlet Process)
- LDA (Latent Dirichlet Allocation)
- LDA (tweaked with topic coherence to find optimal number of topics) and
- LDA as LSI with the help of topic coherence metrics

All of these models are in gensim and can be used easily. We will start by
training our models on the data, and then compare them against each other.
For LDA, we will also use the topic coherence metrics based on `Exploring
the Space of Topic Coherence Measures
<http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>`_ by Roder
et al.

Accompanying slides can be found `here <https://speakerdeck.com/dsquareindia/pycon-delhi-lightening>`_.



.. code-block:: default


    import os
    import operator
    import warnings
    warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
    from pprint import pprint

    import matplotlib.pyplot as plt
    import nltk
    import gensim
    import numpy as np
    from smart_open import open

    test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
    lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')







Analysing our corpus.


   - The first document talks about a bushfire that had occured in New South Wales.
   - The second talks about conflict between India and Pakistan in Kashmir.
   - The third talks about road accidents in the New South Wales area.
   - The fourth one talks about Argentina's economic and political crisis during that time.
   - The last one talks about the use of drugs by midwives in a Sydney hospital.

Our final topic model should be giving us keywords which we can easily
interpret and make a small summary out of. Without this the topic model
cannot be of much practical use.



.. code-block:: default


    nltk.download('stopwords')  # Let's make sure the NLTK 'stopword' package is downloaded & updated
    nltk.download('wordnet')  # Let's also download Wordnet, which will be used for lemmatization

    with open(lee_train_file, 'rb') as f:
        for n, l in enumerate(f):
            if n < 5:
                print(l[:100])

    def build_texts(fname):
        """
        Function to build tokenized texts from file.

        Parameters
        ----------
        fname: File to be read

        Yields
        -------
        Preprocessed lines.
        """
        with open(fname, 'rb') as f:
            for line in f:
                yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)

    train_texts = list(build_texts(lee_train_file))
    print("Number of training documents:", len(train_texts))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    b'Hundreds of people have been forced to vacate their homes in the Southern Highlands of New South Wal'
    b'Indian security forces have shot dead eight suspected militants in a night-long encounter in souther'
    b'The national road toll for the Christmas-New Year holiday period stands at 45, eight fewer than for '
    b"Argentina's political and economic crisis has deepened with the resignation of its interim President"
    b'Six midwives have been suspended at Wollongong Hospital, south of Sydney, for inappropriate use of n'
    Number of training documents: 300



Preprocessing our data. Remember: Garbage In Garbage Out
--------------------------------------------------------

This is the single most important step in setting up a good topic modeling
system. If the preprocessing is not good, the algorithm can't do much since
we would be feeding it a lot of noise. In this tutorial, we will be filtering
out the noise using the following steps in this order for each line:

#. Stopword removal using NLTK's english stopwords dataset.
#. Bigram collocation detection (frequently co-occuring tokens) using
   gensim's `Phrases <https://radimrehurek.com/gensim/models/phrases.html>`_.
   This is our first attempt to find some hidden structure in the corpus. You
   can even try trigram collocation detection.
#. Lemmatization (using :py:func:`gensim.utils.lemmatize`) to
   only keep the nouns. Lemmatization is generally better than stemming in the
   case of topic modeling since the words after lemmatization still remain
   understable. However, generally stemming might be preferred if the data is
   being fed into a vectorizer and isn't intended to be viewed.



.. code-block:: default

    bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection

    print(bigram[['new', 'york', 'example']])

    from gensim.utils import lemmatize
    from nltk.corpus import stopwords

    stops = set(stopwords.words('english'))  # set of stopwords from NLTK

    def process_texts(texts):
        """
        Process texts. Following are the steps we take:

        1. Stopword Removal.
        2. Collocation detection.
        3. Lemmatization (not stem since stemming can reduce the interpretability).

        Parameters
        ----------
        texts: Tokenized texts.

        Returns
        -------
        texts: Pre-processed tokenized texts.
        """
        texts = [[word for word in line if word not in stops] for line in texts]
        texts = [bigram[line] for line in texts]

        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        texts = [[word for word in lemmatizer.lemmatize(' '.join(line), pos='v').split()] for line in texts]
        return texts

    train_texts = process_texts(train_texts)
    print(train_texts[5:6][:20])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ['new_york', 'example']
    [['federal_government', 'says', 'safe', 'afghani', 'asylum_seekers', 'australia', 'return', 'home', 'environment', 'becomes', 'secure', 'government', 'suspended', 'applications', 'interim_government', 'established', 'kabul', 'foreign_affairs', 'minister_alexander', 'downer', 'refused', 'say', 'long', 'claims', 'process', 'put', 'hold', 'says', 'major', 'threat', 'people', 'seeking', 'asylum', 'longer', 'many', 'afghans', 'tried', 'get', 'australia', 'matter', 'britain', 'countries', 'north', 'west', 'europe', 'claimed', 'fleeing', 'taliban', 'said', 'well', 'taliban', 'longer', 'power', 'afghanistan', 'taliban', 'finished', 'meanwhile', 'mass', 'airlift', 'detainees', 'christmas', 'island', 'pacific', 'island', 'nauru', 'total', 'people', 'flown', 'island', 'two', 'operations', 'using', 'chartered', 'aircraft', 'second', 'airlift', 'today', 'delivered', 'asylum_seekers', 'nauru', 'await', 'processing', 'claims', 'temporary', 'visas', 'department', 'immigration', 'says', 'detainees', 'remaining', 'christmas', 'island', 'spokesman', 'says', 'decision', 'regarding', 'future', 'yet', 'made']]



Finalising our dictionary and corpus



.. code-block:: default

    from gensim.corpora import Dictionary
    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]







Topic modeling with LSI
-----------------------

This is a useful topic modeling algorithm in that it can rank topics by
itself. Thus it outputs topics in a ranked order. However it does require a
``num_topics`` parameter (set to 200 by default) to determine the number of
latent dimensions after the SVD.



.. code-block:: default

    from gensim.models import LsiModel
    lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
    for t in lsimodel.show_topics(num_topics=5):
        print(t)

    lsitopics = lsimodel.show_topics(formatted=False)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0, '0.542*"said" + 0.349*"says" + 0.127*"arafat" + 0.122*"palestinian" + 0.118*"people" + 0.117*"israeli" + 0.112*"two" + 0.110*"australian" + 0.110*"also" + 0.107*"australia"')
    (1, '-0.408*"says" + 0.322*"arafat" + 0.315*"palestinian" + 0.273*"israeli" + 0.192*"israel" + 0.173*"sharon" + -0.145*"australia" + -0.143*"australian" + 0.140*"west_bank" + 0.136*"hamas"')
    (2, '-0.349*"says" + 0.330*"said" + 0.203*"afghanistan" + 0.191*"bin_laden" + 0.178*"taliban" + 0.169*"pakistan" + -0.161*"australia" + -0.149*"arafat" + 0.126*"tora_bora" + -0.124*"israeli"')
    (3, '-0.293*"fire" + -0.240*"sydney" + 0.214*"says" + -0.184*"firefighters" + -0.170*"south" + -0.164*"new_south" + -0.164*"wales" + -0.163*"north" + -0.161*"fires" + 0.152*"afghanistan"')
    (4, '0.219*"said" + 0.171*"test" + 0.170*"match" + -0.155*"afghanistan" + -0.149*"government" + -0.148*"says" + 0.144*"first" + -0.142*"fire" + 0.139*"australia" + -0.135*"force"')



Topic modeling with `HDP <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_
-----------------------------------------------------------------------------------------------------

An HDP model is fully unsupervised. It can also determine the ideal number of
topics it needs through posterior inference.



.. code-block:: default

    from gensim.models import HdpModel
    hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
    for t in hdpmodel.show_topics():
        print(t)

    hdptopics = hdpmodel.show_topics(formatted=False)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0, '0.006*said + 0.005*sharon + 0.005*arafat + 0.004*palestinian + 0.003*government + 0.003*israeli + 0.002*west_bank + 0.002*terrorism + 0.002*gaza_strip + 0.002*called + 0.002*security + 0.002*suicide_attacks + 0.002*air_strikes + 0.002*attacks + 0.002*group + 0.002*war + 0.002*address + 0.002*official + 0.002*offices + 0.002*choosing')
    (1, '0.006*said + 0.003*airport + 0.003*taliban + 0.002*killed + 0.002*kandahar + 0.002*opposition + 0.002*nearly + 0.002*near + 0.002*around + 0.002*half + 0.002*city + 0.002*yallourn + 0.001*civilians + 0.001*end + 0.001*agha + 0.001*masood + 0.001*gul + 0.001*night + 0.001*left + 0.001*state')
    (2, '0.004*said + 0.003*afghanistan + 0.002*says + 0.002*powell + 0.002*australian + 0.002*also + 0.002*taliban + 0.002*southern + 0.002*united_states + 0.001*laden + 0.001*time + 0.001*officers + 0.001*osama_bin + 0.001*bin_laden + 0.001*less + 0.001*troops + 0.001*former + 0.001*mountains + 0.001*saviour + 0.001*rumsfeld')
    (3, '0.004*team + 0.003*rafter + 0.002*australia + 0.002*said + 0.002*says + 0.002*win + 0.002*good + 0.002*doubles + 0.002*make + 0.002*know + 0.001*expect + 0.001*racing + 0.001*today + 0.001*hewitt + 0.001*davis + 0.001*hour + 0.001*prizemoney + 0.001*new_zealand + 0.001*number + 0.001*todd')
    (4, '0.002*howard + 0.002*australia + 0.002*indonesia + 0.002*match + 0.002*says + 0.002*megawati + 0.002*summit + 0.001*day + 0.001*president + 0.001*said + 0.001*pieterse + 0.001*towards + 0.001*game + 0.001*test + 0.001*buchanan + 0.001*fast_bowler + 0.001*two + 0.001*win + 0.001*honours + 0.001*talks')
    (5, '0.004*company + 0.002*staff + 0.002*says + 0.002*cow + 0.002*entitlements + 0.002*disease + 0.002*austar + 0.002*said + 0.002*million + 0.002*confirmed + 0.002*japan + 0.001*administrators + 0.001*case + 0.001*receive + 0.001*third + 0.001*line + 0.001*albarran + 0.001*considerable + 0.001*foley + 0.001*traveland')
    (6, '0.003*match + 0.002*israeli + 0.002*rafter + 0.002*team + 0.002*says + 0.002*training + 0.002*tennis + 0.001*guarantee + 0.001*france + 0.001*australia + 0.001*said + 0.001*killed + 0.001*way + 0.001*arthurs + 0.001*members + 0.001*swinging + 0.001*government + 0.001*day + 0.001*want + 0.001*asked')
    (7, '0.003*afghan + 0.003*rabbani + 0.002*afghanistan + 0.002*says + 0.002*president + 0.002*agreement + 0.002*would + 0.002*government + 0.002*interim + 0.001*security + 0.001*factions + 0.001*bonn + 0.001*leaders + 0.001*talks + 0.001*northern_alliance + 0.001*international + 0.001*security_forces + 0.001*rosebury + 0.001*personnel + 0.001*real')
    (8, '0.003*australia + 0.003*says + 0.003*economy + 0.002*strong + 0.002*conference + 0.002*republic + 0.001*australian + 0.001*would + 0.001*canberra + 0.001*monarchy + 0.001*recession + 0.001*rural + 0.001*banksa + 0.001*quite + 0.001*process + 0.001*follow + 0.001*corowa + 0.001*taylor + 0.001*reverend + 0.001*said')
    (9, '0.002*adventure_world + 0.002*canyoning + 0.002*one + 0.002*three + 0.002*interlaken + 0.002*guides + 0.002*tourists + 0.001*court + 0.001*staff + 0.001*eight + 0.001*massive + 0.001*swiss + 0.001*changes + 0.001*convicted + 0.001*group + 0.001*says + 0.001*sentence + 0.001*allowed + 0.001*employed + 0.001*peace')
    (10, '0.003*israeli + 0.003*said + 0.002*palestinian + 0.002*source + 0.002*hamas + 0.002*two + 0.002*militants + 0.001*soldiers + 0.001*palestinian_security + 0.001*ismail + 0.001*west_bank + 0.001*police + 0.001*near + 0.001*leaders + 0.001*senior + 0.001*quest + 0.001*killed + 0.001*afp + 0.001*palestinians + 0.001*another')
    (11, '0.002*government + 0.002*said + 0.002*says + 0.002*help + 0.002*bid + 0.002*time + 0.002*lew + 0.001*famously + 0.001*assistance + 0.001*would + 0.001*per_cent + 0.001*review + 0.001*bin_laden + 0.001*federal_government + 0.001*dominance + 0.001*brewing + 0.001*progress + 0.001*trade + 0.001*lindsay + 0.001*leak')
    (12, '0.002*storm + 0.002*says + 0.002*sydney + 0.002*ses + 0.002*trees + 0.002*damage + 0.002*around + 0.002*said + 0.002*hornsby + 0.002*homes + 0.002*brought + 0.001*worst + 0.001*hit + 0.001*areas + 0.001*services + 0.001*storms + 0.001*slightly + 0.001*feet + 0.001*electricity + 0.001*energy')
    (13, '0.002*commission + 0.002*collapse + 0.002*one + 0.001*said + 0.001*australian + 0.001*says + 0.001*whether + 0.001*dickie + 0.001*begin + 0.001*expected + 0.001*today + 0.001*nabil + 0.001*need + 0.001*directors + 0.001*asic + 0.001*regime + 0.001*ascertain + 0.001*government + 0.001*witnesses + 0.001*regulatory')
    (14, '0.003*krishna + 0.003*hare + 0.003*benares + 0.003*ashes + 0.002*harrison + 0.002*ganges + 0.002*holy + 0.002*take + 0.001*river + 0.001*ceremony + 0.001*sect + 0.001*jackie + 0.001*devotees + 0.001*morning + 0.001*would + 0.001*members + 0.001*fans + 0.001*wore + 0.001*told + 0.001*said')
    (15, '0.001*band + 0.001*powerful + 0.001*lennon + 0.001*harrison + 0.001*songs + 0.001*known + 0.001*undertakings + 0.001*beatles + 0.001*salvation + 0.001*sun + 0.001*success + 0.001*said + 0.001*cranky + 0.001*beatle + 0.001*motor + 0.001*sustained + 0.001*found + 0.001*although + 0.001*hamas + 0.001*widely')
    (16, '0.003*friedli + 0.002*people + 0.002*replied + 0.002*know + 0.002*director + 0.002*company + 0.002*day + 0.002*said + 0.001*adventure_world + 0.001*swiss + 0.001*two + 0.001*trip + 0.001*along + 0.001*accused + 0.001*court + 0.001*think + 0.001*question + 0.001*asked + 0.001*trial + 0.001*first')
    (17, '0.001*one + 0.001*collapse + 0.001*third + 0.001*inspired + 0.001*made + 0.001*appleby + 0.001*parties + 0.001*future + 0.001*given + 0.001*recommendation + 0.001*masterminding + 0.001*disruptions + 0.001*best + 0.001*gabriel + 0.001*first + 0.001*culture + 0.001*report + 0.001*aware + 0.001*principles + 0.001*issue')
    (18, '0.003*harrison + 0.002*george + 0.002*said + 0.002*tonight + 0.002*memory + 0.002*died + 0.002*music + 0.001*beatle + 0.001*liverpool + 0.001*losing + 0.001*talented + 0.001*people + 0.001*insisted + 0.001*lord + 0.001*gaming + 0.001*say + 0.001*vigil + 0.001*planted + 0.001*silence + 0.001*internal')
    (19, '0.002*party + 0.002*lee + 0.002*said + 0.001*yesterday + 0.001*new_zealand + 0.001*coalition + 0.001*match + 0.001*two + 0.001*national + 0.001*says + 0.001*stage + 0.001*shane + 0.001*president + 0.001*atlanta + 0.001*parties + 0.001*outlined + 0.001*amalgamation + 0.001*test + 0.001*upon + 0.001*states')



Topic modeling using `LDA <https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_
----------------------------------------------------------------------------------------------------

This is one the most popular topic modeling algorithms today. It is a
generative model in that it assumes each document is a mixture of topics and
in turn, each topic is a mixture of words. To understand it better you can
watch `this <https://www.youtube.com/watch?v=DDq3OVp9dNA>`_ lecture by David
Blei. Let's choose 10 topics to initialize this.



.. code-block:: default

    from gensim.models import LdaModel
    ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
    ldatopics = ldamodel.show_topics(formatted=False)







pyLDAvis is a great way to visualize an LDA model. To summarize in short, the
area of the circles represent the prevelance of the topic. The length of the
bars on the right represent the membership of a term in a particular topic.
For more, see `LDAVis: A method for visualizing and interpreting topics
<http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf>`_.

Unfortunately, the visualization only works inside a Jupyter notebook.



.. code-block:: default


    try:
        import pyLDAvis.gensim
        get_ipython()
    except Exception:
        pass
    else:
        pyLDAvis.enable_notebook()
        pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)







Determining the optimal number of topics
----------------------------------------

**Introduction to topic coherence**\ :

.. role:: raw-html-m2r(raw)
   :format: html

:raw-html-m2r:`<img src="https://rare-technologies.com/wp-content/uploads/2016/06/pipeline.png">`

Topic coherence in essence measures the human interpretability of a topic
model. Traditionally `perplexity has been used
<http://qpleple.com/perplexity-to-evaluate-topic-models/>`_ to evaluate topic
models however this does not correlate with human annotations at times. Topic
coherence is another way to evaluate topic models with a much higher
guarantee on human interpretability. Thus this can be used to compare
different topic models among many other use-cases. Here's a short blog I
wrote explaining topic coherence:

`What is topic coherence? <https://rare-technologies.com/what-is-topic-coherence/>`_



.. code-block:: default

    from gensim.models import CoherenceModel


    def evaluate_graph(dictionary, corpus, texts, limit):
        """
        Function to display num_topics - LDA graph using c_v coherence

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        limit : topic limit

        Returns:
        -------
        lm_list : List of LDA topic models
        c_v : Coherence values corresponding to the LDA model with respective number of topics
        """
        c_v = []
        lm_list = []
        for num_topics in range(1, limit):
            lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            lm_list.append(lm)
            cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())

        # Show graph
        x = range(1, limit)
        plt.plot(x, c_v)
        plt.xlabel("num_topics")
        plt.ylabel("Coherence score")
        plt.legend(("c_v"), loc='best')
        plt.show()

        return lm_list, c_v

    lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=10)
    # pyLDAvis.gensim.prepare(lmlist[2], corpus, dictionary)
    lmtopics = lmlist[5].show_topics(formatted=False)

    lm, top_topics = ldamodel, ldatopics




.. image:: /auto_examples/howtos/images/sphx_glr_run_news_classification_001.png
    :class: sphx-glr-single-img




Evaluating all the topic models
-------------------------------

Any topic model which can come up with topic terms can be plugged into the
coherence pipeline. You can even plug in an `NMF topic model
<http://derekgreene.com/nmf-topic/>`_ created with scikit-learn.



.. code-block:: default


    lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]
    hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
    ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
    lmtopics = [[word for word, prob in topic] for topicid, topic in lmtopics]

    def create_coherence_model(topics):
        return CoherenceModel(
            topics=topics,
            texts=train_texts,
            dictionary=dictionary,
            window_size=10
        ).get_coherence()

    lsi_coherence = create_coherence_model(lsitopics[:10])
    hdp_coherence = create_coherence_model(hdptopics[:10])
    lda_coherence = create_coherence_model(ldatopics)
    lm_coherence = create_coherence_model(lmtopics)

    def evaluate_bar_graph(coherences, indices):
        """
        Function to plot bar graph.

        coherences: list of coherence values
        indices: Indices to be used to mark bars. Length of this and coherences should be equal.
        """
        assert len(coherences) == len(indices)
        n = len(coherences)
        x = np.arange(n)
        plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
        plt.xlabel('Models')
        plt.ylabel('Coherence Value')

    values = [lsi_coherence, hdp_coherence, lda_coherence, lm_coherence]
    labels = ['LSI', 'HDP', 'LDA', 'LDA_Mod']
    evaluate_bar_graph(values, labels)




.. image:: /auto_examples/howtos/images/sphx_glr_run_news_classification_002.png
    :class: sphx-glr-single-img




Customizing the topic coherence measure
---------------------------------------

Till now we only used the ``c_v`` coherence measure. There are others such as
``u_mass``\ , ``c_uci``\ , ``c_npmi``. All of these calculate coherence in a
different way. ``c_v`` is found to be most in line with human ratings but can
be much slower than ``u_mass`` since it uses a sliding window over the texts.

Making your own coherence measure
---------------------------------

Let's modify ``c_uci`` to use ``s_one_pre`` instead of ``s_one_one`` segmentation



.. code-block:: default


    from gensim.topic_coherence import (
        segmentation, probability_estimation, direct_confirmation_measure,
        indirect_confirmation_measure, aggregation
    )
    from gensim.matutils import argsort
    from collections import namedtuple

    make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')

    measure = make_pipeline(
        segmentation.s_one_one,
        probability_estimation.p_boolean_sliding_window,
        direct_confirmation_measure.log_ratio_measure,
        aggregation.arithmetic_mean,
    )







To get topics out of the topic model:



.. code-block:: default


    topics = []
    for topic in lm.state.get_lambda():
        bestn = argsort(topic, topn=10, reverse=True)
    topics.append(bestn)

    for t in topics:
        print(t)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 98  99 258 356  86 986 349 574  80 200]



**Step 1**\ : Segmentation



.. code-block:: default

    segmented_topics = measure.seg(topics)

    for t in segmented_topics:
        print(t)

    #
    # Unfortunately, the stuff below doesn't work, either :(
    #





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [(98, 99), (98, 258), (98, 356), (98, 86), (98, 986), (98, 349), (98, 574), (98, 80), (98, 200), (99, 98), (99, 258), (99, 356), (99, 86), (99, 986), (99, 349), (99, 574), (99, 80), (99, 200), (258, 98), (258, 99), (258, 356), (258, 86), (258, 986), (258, 349), (258, 574), (258, 80), (258, 200), (356, 98), (356, 99), (356, 258), (356, 86), (356, 986), (356, 349), (356, 574), (356, 80), (356, 200), (86, 98), (86, 99), (86, 258), (86, 356), (86, 986), (86, 349), (86, 574), (86, 80), (86, 200), (986, 98), (986, 99), (986, 258), (986, 356), (986, 86), (986, 349), (986, 574), (986, 80), (986, 200), (349, 98), (349, 99), (349, 258), (349, 356), (349, 86), (349, 986), (349, 574), (349, 80), (349, 200), (574, 98), (574, 99), (574, 258), (574, 356), (574, 86), (574, 986), (574, 349), (574, 80), (574, 200), (80, 98), (80, 99), (80, 258), (80, 356), (80, 86), (80, 986), (80, 349), (80, 574), (80, 200), (200, 98), (200, 99), (200, 258), (200, 356), (200, 86), (200, 986), (200, 349), (200, 574), (200, 80)]



**Step 2**\ : Probability estimation

Since this is a window-based coherence measure we will perform window based prob estimation


.. code-block:: default

    try:
        per_topic_postings, num_windows = measure.prob(
            texts=train_texts, segmented_topics=segmented_topics,
            dictionary=dictionary, window_size=2,
        )
    except Exception:
        pass







**Step 3**\ : Confirmation Measure



.. code-block:: default

    try:
        confirmed_measures = measure.conf(segmented_topics, per_topic_postings, num_windows, normalize=False)
    except Exception:
        pass







**Step 4**\ : Aggregation



.. code-block:: default


    try:
        print(measure.aggr(confirmed_measures))
    except Exception:
        pass







How this topic model can be used further
========================================

A topic model can be used as a dimensionality reduction algorithm to feed into a classifier.
A good topic model should be able to extract the signal from the noise efficiently, hence improving the performance of the classifier.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  17.335 seconds)

**Estimated memory usage:**  211 MB


.. _sphx_glr_download_auto_examples_howtos_run_news_classification.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_news_classification.py <run_news_classification.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_news_classification.ipynb <run_news_classification.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
