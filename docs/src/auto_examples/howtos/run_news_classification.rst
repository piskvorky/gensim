.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_howtos_run_news_classification.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_howtos_run_news_classification.py:


How to Classify News Articles into Topics
=========================================

Demonstrates classification of the Lee Corpus using a variety of topic models (LSI, HDP, LDA, etc).
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

All of these models are in gensim and can be used easily.  We will start by
training our models on the data, and then compare them against each other.
For LDA, we will also use the # topic coherence metrics based on `Exploring
the Space of Topic Coherence Measures
<http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>`_ by Roder
et al.

Accompanying slides can be found `here <https://speakerdeck.com/dsquareindia/pycon-delhi-lightening>`_.



.. code-block:: default


    import os
    import re
    import operator
    import matplotlib.pyplot as plt
    import warnings
    import gensim
    import numpy as np
    warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

    import nltk
    nltk.download('stopwords') # Let's make sure the 'stopword' package is downloaded & updated
    nltk.download('wordnet') # Let's also download wordnet, which will be used for lemmatization

    from pprint import pprint
    from smart_open import open

    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'







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

    with open(lee_train_file, 'rb') as f:
        for n, l in enumerate(f):
            if n < 5:
                print(l[:100])

    def build_texts(fname):
        """
        Function to build tokenized texts from file
    
        Parameters:
        ----------
        fname: File to be read
    
        Returns:
        -------
        yields preprocessed line
        """
        with open(fname, 'rb') as f:
            for line in f:
                yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)

    train_texts = list(build_texts(lee_train_file))
    print(len(train_texts))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    b'Hundreds of people have been forced to vacate their homes in the Southern Highlands of New South Wal'
    b'Indian security forces have shot dead eight suspected militants in a night-long encounter in souther'
    b'The national road toll for the Christmas-New Year holiday period stands at 45, eight fewer than for '
    b"Argentina's political and economic crisis has deepened with the resignation of its interim President"
    b'Six midwives have been suspended at Wollongong Hospital, south of Sydney, for inappropriate use of n'
    300


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

    bigram[['new', 'york', 'example']]

    from gensim.utils import lemmatize
    from nltk.corpus import stopwords

    stops = set(stopwords.words('english'))  # nltk stopwords list

    def process_texts(texts):
        """
        Function to process texts. Following are the steps we take:
    
        1. Stopword Removal.
        2. Collocation detection.
        3. Lemmatization (not stem since stemming can reduce the interpretability).
    
        Parameters:
        ----------
        texts: Tokenized texts.
    
        Returns:
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
    (1, '0.408*"says" + -0.322*"arafat" + -0.315*"palestinian" + -0.273*"israeli" + -0.192*"israel" + -0.173*"sharon" + 0.145*"australia" + 0.143*"australian" + -0.140*"west_bank" + -0.136*"hamas"')
    (2, '0.348*"says" + -0.330*"said" + -0.202*"afghanistan" + -0.191*"bin_laden" + -0.179*"taliban" + -0.169*"pakistan" + 0.162*"australia" + 0.149*"arafat" + -0.126*"tora_bora" + 0.124*"palestinian"')
    (3, '0.294*"fire" + 0.240*"sydney" + -0.214*"says" + 0.184*"firefighters" + 0.170*"south" + 0.165*"wales" + 0.165*"new_south" + 0.162*"north" + 0.160*"fires" + -0.153*"afghanistan"')
    (4, '0.220*"said" + 0.171*"test" + 0.170*"match" + -0.155*"afghanistan" + -0.149*"government" + -0.148*"says" + 0.143*"first" + -0.142*"fire" + 0.139*"australia" + -0.135*"force"')


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

    (0, '0.007*said + 0.004*taliban + 0.003*airport + 0.003*kandahar + 0.003*also + 0.003*afghanistan + 0.002*people + 0.002*opposition + 0.002*powell + 0.002*osama_bin + 0.002*forces + 0.002*killed + 0.002*city + 0.002*area + 0.002*near + 0.002*southern + 0.002*left + 0.001*bin_laden + 0.001*wounded + 0.001*says')
    (1, '0.003*says + 0.003*match + 0.002*team + 0.002*said + 0.002*government + 0.002*president + 0.002*could + 0.002*israeli + 0.002*rafter + 0.002*france + 0.002*troops + 0.001*guarantee + 0.001*media + 0.001*escude + 0.001*australia + 0.001*tennis + 0.001*afghanistan + 0.001*career + 0.001*still + 0.001*factions')
    (2, '0.004*said + 0.003*krishna + 0.003*sharon + 0.003*palestinian + 0.003*government + 0.002*benares + 0.002*ashes + 0.002*hare + 0.002*group + 0.002*ganges + 0.002*harrison + 0.002*ceremony + 0.002*dead + 0.002*human_rights + 0.002*suicide_attacks + 0.002*take + 0.001*terrorism + 0.001*river + 0.001*israeli + 0.001*arafat')
    (3, '0.003*said + 0.003*friedli + 0.002*company + 0.002*know + 0.002*people + 0.002*director + 0.002*replied + 0.002*two + 0.002*made + 0.002*adventure_world + 0.002*flood + 0.002*court + 0.002*deaths + 0.002*australian + 0.001*accident + 0.001*state + 0.001*think + 0.001*carried + 0.001*one + 0.001*head')
    (4, '0.002*says + 0.002*said + 0.002*people + 0.001*australian + 0.001*offer + 0.001*going + 0.001*ashcroft + 0.001*back + 0.001*travel + 0.001*administrator + 0.001*traveland + 0.001*mcdonald + 0.001*anaemia + 0.001*sharon + 0.001*united_states + 0.001*make + 0.001*worshipped + 0.001*arafat + 0.001*chain + 0.001*fundamentalist')
    (5, '0.004*arafat + 0.004*said + 0.003*sharon + 0.003*israeli + 0.002*palestinian + 0.002*arrest + 0.002*afp + 0.002*official + 0.002*west_bank + 0.002*attacks + 0.002*israeli_forces + 0.002*military + 0.002*end + 0.001*islamic + 0.001*war + 0.001*terrorism + 0.001*suicide_bombings + 0.001*killed + 0.001*government + 0.001*choosing')
    (6, '0.003*report + 0.003*says + 0.002*company + 0.002*storm + 0.002*brought + 0.002*trees + 0.002*said + 0.002*ses + 0.002*sydney + 0.002*storms + 0.002*welfare + 0.002*worst + 0.001*around + 0.001*hornsby + 0.001*australia + 0.001*services + 0.001*areas + 0.001*hit + 0.001*homes + 0.001*two')
    (7, '0.003*harrison + 0.002*said + 0.002*george + 0.002*memory + 0.002*died + 0.002*liverpool + 0.002*beatle + 0.001*tonight + 0.001*music + 0.001*one + 0.001*man + 0.001*come + 0.001*consensus + 0.001*family + 0.001*silence + 0.001*tree + 0.001*lord + 0.001*show + 0.001*lives + 0.001*still')
    (8, '0.002*commission + 0.002*today + 0.002*one + 0.002*collapse + 0.001*hiv + 0.001*combination + 0.001*begin + 0.001*million + 0.001*europe + 0.001*world + 0.001*barrack + 0.001*regarding + 0.001*eternal + 0.001*dickie + 0.001*epidemic + 0.001*says + 0.001*aids + 0.001*found + 0.001*former + 0.001*australian')
    (9, '0.002*says + 0.002*virgin + 0.002*launceston + 0.002*paceman + 0.002*airline + 0.002*could + 0.001*terminal + 0.001*said + 0.001*told + 0.001*possible + 0.001*owen + 0.001*justice + 0.001*melbourne + 0.001*rocket + 0.001*flight + 0.001*administrators + 0.001*ansett + 0.001*morning + 0.001*withdraw + 0.001*daily')
    (10, '0.002*says + 0.002*australia + 0.002*said + 0.002*sydney + 0.002*damage + 0.001*north + 0.001*hit + 0.001*fire + 0.001*volunteers + 0.001*wales + 0.001*state + 0.001*new_south + 0.001*storms + 0.001*almao + 0.001*west + 0.001*back + 0.001*storm + 0.001*report + 0.001*natural + 0.001*areas')
    (11, '0.002*three + 0.002*israeli + 0.002*palestinian + 0.002*near + 0.002*canyoning + 0.002*adventure_world + 0.002*one + 0.002*guides + 0.002*interlaken + 0.002*said + 0.002*killed + 0.002*militants + 0.002*two + 0.002*hamas + 0.001*west_bank + 0.001*group + 0.001*check + 0.001*yachting + 0.001*occur + 0.001*staff')
    (12, '0.003*team + 0.003*rafter + 0.002*doubles + 0.002*win + 0.002*australia + 0.002*said + 0.002*says + 0.002*good + 0.002*know + 0.002*expect + 0.001*make + 0.001*australian + 0.001*added + 0.001*john + 0.001*singles + 0.001*hewitt + 0.001*even + 0.001*davis + 0.001*enough + 0.001*part')
    (13, '0.002*dispute + 0.001*unions + 0.001*yallourn + 0.001*victorian + 0.001*energy + 0.001*government + 0.001*agreement + 0.001*matter + 0.001*electricity + 0.001*shanab + 0.001*industrial_relations + 0.001*hare + 0.001*act + 0.001*running + 0.001*malaysia + 0.001*new + 0.001*bargaining + 0.001*enterprise + 0.001*major + 0.001*compulsory')
    (14, '0.005*company + 0.002*entitlements + 0.002*staff + 0.002*austar + 0.002*says + 0.002*receive + 0.001*administrators + 0.001*million + 0.001*brands + 0.001*traveland + 0.001*payments + 0.001*albarran + 0.001*morning + 0.001*schoolboys + 0.001*operator + 0.001*network + 0.001*operating + 0.001*line + 0.001*creditors + 0.001*strangled')
    (15, '0.002*australia + 0.001*childhood + 0.001*day + 0.001*sight + 0.001*trenches + 0.001*said + 0.001*final + 0.001*ruled + 0.001*joint + 0.001*adam + 0.001*shopping + 0.001*captain + 0.001*says + 0.001*disintegration + 0.001*milder + 0.001*foisted + 0.001*set + 0.001*could + 0.001*workforce + 0.001*kabul')
    (16, '0.003*says + 0.003*india + 0.001*document + 0.001*australia + 0.001*government + 0.001*behest + 0.001*report + 0.001*laws + 0.001*said + 0.001*labour + 0.001*interrogated + 0.001*indian + 0.001*jobs + 0.001*union + 0.001*new + 0.001*sector + 0.001*economy + 0.001*hollingworth + 0.001*could + 0.001*caird')
    (17, '0.003*says + 0.003*economy + 0.002*australia + 0.002*strong + 0.001*rural + 0.001*set + 0.001*firm + 0.001*economic + 0.001*recession + 0.001*banksa + 0.001*taylor + 0.001*gaps + 0.001*australian + 0.001*sharing + 0.001*teenagers + 0.001*also + 0.001*quite + 0.001*meanwhile + 0.001*follow + 0.001*underwent')
    (18, '0.003*government + 0.002*bid + 0.002*help + 0.002*assistance + 0.002*says + 0.002*lew + 0.001*review + 0.001*trade + 0.001*would + 0.001*said + 0.001*lindsay + 0.001*break + 0.001*publicly + 0.001*qantas + 0.001*per_cent + 0.001*also + 0.001*identifiable + 0.001*time + 0.001*practices + 0.001*dominance')
    (19, '0.002*job + 0.002*per_cent + 0.002*olivier + 0.001*ever + 0.001*anz + 0.001*says + 0.001*lowest + 0.001*employment + 0.001*fateful + 0.001*since + 0.001*market + 0.001*november + 0.001*fall + 0.001*advertising + 0.001*banksa + 0.001*measured + 0.001*week + 0.001*drop + 0.001*internet + 0.001*interest_rates')


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




.. image:: /auto_examples/howtos/images/sphx_glr_run_news_classification_001.png
    :class: sphx-glr-single-img




LDA as LSI
----------

One of the problem with LDA is that if we train it on a large number of
topics, the topics get "lost" among the numbers. Let us see if we can dig out
the best topics from the best LDA model we can produce. The function below
can be used to control the quality of the LDA model we produce.



.. code-block:: default



    def ret_top_model():
        """
        Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
        quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
        evaluating the topic model until this threshold is crossed. 
    
        Returns:
        -------
        lm: Final evaluated topic model
        top_topics: ranked topics in decreasing order. List of tuples
        """
        top_topics = [(0, 0)]
        while top_topics[0][1] < 0.97:
            lm = LdaModel(corpus=corpus, id2word=dictionary)
            coherence_values = {}
            for n, topic in lm.show_topics(num_topics=-1, formatted=False):
                topic = [word for word, _ in topic]
                cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
                coherence_values[n] = cm.get_coherence()
            top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
        return lm, top_topics

    #
    # This part is broken: the confidence never reaches 0.97.
    # It also takes a prohibitively long time to run.  Disable it for now.
    # Use the regular LDA model instead, to keep the rest of this script working.
    #
    # lm, top_topics = ret_top_model()
    # print(top_topics[:5])
    lm, top_topics = ldamodel, ldatopics







Inference
---------

We can clearly see below that the first topic is about **cinema**\ , second is about **email malware**\ , third is about the land which was given back to the **Larrakia aboriginal community of Australia** in 2000. Then there's one about **Australian cricket**. LDA as LSI has worked wonderfully in finding out the best topics from within LDA.

pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])
lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]

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
    # lda_lsi_coherence = create_coherence_model(lda_lsi_topics[:10])

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

    values = [lsi_coherence, hdp_coherence, lda_coherence, lm_coherence] #, lda_lsi_coherence]
    labels = ['LSI', 'HDP', 'LDA', 'LDA_Mod'] #, 'LDA_LSI']
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

    measure = make_pipeline(segmentation.s_one_one,
                            probability_estimation.p_boolean_sliding_window,
                            direct_confirmation_measure.log_ratio_measure,
                            aggregation.arithmetic_mean)







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

    [  98   99  986  967  956 2551  258 2082  574  356]


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

    [(98, 99), (98, 986), (98, 967), (98, 956), (98, 2551), (98, 258), (98, 2082), (98, 574), (98, 356), (99, 98), (99, 986), (99, 967), (99, 956), (99, 2551), (99, 258), (99, 2082), (99, 574), (99, 356), (986, 98), (986, 99), (986, 967), (986, 956), (986, 2551), (986, 258), (986, 2082), (986, 574), (986, 356), (967, 98), (967, 99), (967, 986), (967, 956), (967, 2551), (967, 258), (967, 2082), (967, 574), (967, 356), (956, 98), (956, 99), (956, 986), (956, 967), (956, 2551), (956, 258), (956, 2082), (956, 574), (956, 356), (2551, 98), (2551, 99), (2551, 986), (2551, 967), (2551, 956), (2551, 258), (2551, 2082), (2551, 574), (2551, 356), (258, 98), (258, 99), (258, 986), (258, 967), (258, 956), (258, 2551), (258, 2082), (258, 574), (258, 356), (2082, 98), (2082, 99), (2082, 986), (2082, 967), (2082, 956), (2082, 2551), (2082, 258), (2082, 574), (2082, 356), (574, 98), (574, 99), (574, 986), (574, 967), (574, 956), (574, 2551), (574, 258), (574, 2082), (574, 356), (356, 98), (356, 99), (356, 986), (356, 967), (356, 956), (356, 2551), (356, 258), (356, 2082), (356, 574)]


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

The best topic model here can be used as a standalone for news article classification. However a topic model can also be used as a dimensionality reduction algorithm to feed into a classifier. A good topic model should be able to extract the signal from the noise efficiently, hence improving the performance of the classifier.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  19.499 seconds)

**Estimated memory usage:**  108 MB


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

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
