r"""
How to Classify News Articles into Topics
=========================================

Demonstrates classification of the Lee Corpus using a variety of topic models (LSI, HDP, LDA, etc).

"""

###############################################################################
# News article classification is performed on a huge scale by news agencies all
# over the world. We will be looking into how topic modeling can be used to
# accurately classify news articles into different categories such as sports,
# technology, politics etc.
# 
# This guide demonstrates training a topic model which can come up with topics
# that can easily be interpreted by us. On top of assigning a topic to an
# arbitrary document, this model can also discover hidden structure in the
# corpus.
# 
# We will be using the Lee corpus which is a shortened version of the `Lee
# Background Corpus
# <http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF>`_. The
# shortened version consists of 300 documents selected from the Australian
# Broadcasting Corporation's news mail service. It consists of texts of
# headline stories from around the year 2000-2001.
# 
# We will examine the following models:
# 
# - LSI (Latent Semantic Indexing)
# - HDP (Hierarchical Dirichlet Process)
# - LDA (Latent Dirichlet Allocation)
# - LDA (tweaked with topic coherence to find optimal number of topics) and
# - LDA as LSI with the help of topic coherence metrics
#
# All of these models are in gensim and can be used easily.  We will start by
# training our models on the data, and then compare them against each other.
# For LDA, we will also use the # topic coherence metrics based on `Exploring
# the Space of Topic Coherence Measures
# <http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>`_ by Roder
# et al.
# 
# Accompanying slides can be found `here <https://speakerdeck.com/dsquareindia/pycon-delhi-lightening>`_.
# 

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

###############################################################################
# Analysing our corpus.
# 
# 
#    - The first document talks about a bushfire that had occured in New South Wales.
#    - The second talks about conflict between India and Pakistan in Kashmir.
#    - The third talks about road accidents in the New South Wales area.
#    - The fourth one talks about Argentina's economic and political crisis during that time.
#    - The last one talks about the use of drugs by midwives in a Sydney hospital.
# 
# Our final topic model should be giving us keywords which we can easily
# interpret and make a small summary out of. Without this the topic model
# cannot be of much practical use.
# 
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

###############################################################################
# Preprocessing our data. Remember: Garbage In Garbage Out
# --------------------------------------------------------
# 
# This is the single most important step in setting up a good topic modeling
# system. If the preprocessing is not good, the algorithm can't do much since
# we would be feeding it a lot of noise. In this tutorial, we will be filtering
# out the noise using the following steps in this order for each line:
# 
# #. Stopword removal using NLTK's english stopwords dataset.
# #. Bigram collocation detection (frequently co-occuring tokens) using
#    gensim's `Phrases <https://radimrehurek.com/gensim/models/phrases.html>`_.
#    This is our first attempt to find some hidden structure in the corpus. You
#    can even try trigram collocation detection.
# #. Lemmatization (using :py:func:`gensim.utils.lemmatize`) to
#    only keep the nouns. Lemmatization is generally better than stemming in the
#    case of topic modeling since the words after lemmatization still remain
#    understable. However, generally stemming might be preferred if the data is
#    being fed into a vectorizer and isn't intended to be viewed.
# 
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

###############################################################################
# Finalising our dictionary and corpus
# 
from gensim.corpora import Dictionary
dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

###############################################################################
# Topic modeling with LSI
# -----------------------
# 
# This is a useful topic modeling algorithm in that it can rank topics by
# itself. Thus it outputs topics in a ranked order. However it does require a
# ``num_topics`` parameter (set to 200 by default) to determine the number of
# latent dimensions after the SVD.
# 
from gensim.models import LsiModel
lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
for t in lsimodel.show_topics(num_topics=5):
    print(t)

lsitopics = lsimodel.show_topics(formatted=False)


###############################################################################
# Topic modeling with `HDP <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_
# -----------------------------------------------------------------------------------------------------
# 
# An HDP model is fully unsupervised. It can also determine the ideal number of
# topics it needs through posterior inference.
# 
from gensim.models import HdpModel
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
for t in hdpmodel.show_topics():
    print(t)

hdptopics = hdpmodel.show_topics(formatted=False)

###############################################################################
# Topic modeling using `LDA <https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_
# ----------------------------------------------------------------------------------------------------
# 
# This is one the most popular topic modeling algorithms today. It is a
# generative model in that it assumes each document is a mixture of topics and
# in turn, each topic is a mixture of words. To understand it better you can
# watch `this <https://www.youtube.com/watch?v=DDq3OVp9dNA>`_ lecture by David
# Blei. Let's choose 10 topics to initialize this.
# 
from gensim.models import LdaModel
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldatopics = ldamodel.show_topics(formatted=False)

###############################################################################
# pyLDAvis is a great way to visualize an LDA model. To summarize in short, the
# area of the circles represent the prevelance of the topic. The length of the
# bars on the right represent the membership of a term in a particular topic.
# For more, see `LDAVis: A method for visualizing and interpreting topics
# <http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf>`_.
#
# Unfortunately, the visualization only works inside a Jupyter notebook.
# 

try:
    import pyLDAvis.gensim
    get_ipython()
except Exception:
    pass
else:
    pyLDAvis.enable_notebook()
    pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

###############################################################################
# Determining the optimal number of topics
# ----------------------------------------
# 
# **Introduction to topic coherence**\ :
# 
# .. role:: raw-html-m2r(raw)
#    :format: html
# 
# :raw-html-m2r:`<img src="https://rare-technologies.com/wp-content/uploads/2016/06/pipeline.png">`
# 
# Topic coherence in essence measures the human interpretability of a topic
# model. Traditionally `perplexity has been used
# <http://qpleple.com/perplexity-to-evaluate-topic-models/>`_ to evaluate topic
# models however this does not correlate with human annotations at times. Topic
# coherence is another way to evaluate topic models with a much higher
# guarantee on human interpretability. Thus this can be used to compare
# different topic models among many other use-cases. Here's a short blog I
# wrote explaining topic coherence:
# 
# `What is topic coherence? <https://rare-technologies.com/what-is-topic-coherence/>`_
# 
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

###############################################################################
# LDA as LSI
# ----------
#
# One of the problem with LDA is that if we train it on a large number of
# topics, the topics get "lost" among the numbers. Let us see if we can dig out
# the best topics from the best LDA model we can produce. The function below
# can be used to control the quality of the LDA model we produce.
# 


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

###############################################################################
# Inference
# ---------
# 
# We can clearly see below that the first topic is about **cinema**\ , second is about **email malware**\ , third is about the land which was given back to the **Larrakia aboriginal community of Australia** in 2000. Then there's one about **Australian cricket**. LDA as LSI has worked wonderfully in finding out the best topics from within LDA.
# 
# pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])
# lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]

###############################################################################
# Evaluating all the topic models
# -------------------------------
# 
# Any topic model which can come up with topic terms can be plugged into the
# coherence pipeline. You can even plug in an `NMF topic model
# <http://derekgreene.com/nmf-topic/>`_ created with scikit-learn.
# 

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

###############################################################################
# Customizing the topic coherence measure
# ---------------------------------------
# 
# Till now we only used the ``c_v`` coherence measure. There are others such as
# ``u_mass``\ , ``c_uci``\ , ``c_npmi``. All of these calculate coherence in a
# different way. ``c_v`` is found to be most in line with human ratings but can
# be much slower than ``u_mass`` since it uses a sliding window over the texts.
# 
# Making your own coherence measure
# ---------------------------------
# 
# Let's modify ``c_uci`` to use ``s_one_pre`` instead of ``s_one_one`` segmentation
# 

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

###############################################################################
# To get topics out of the topic model:
# 

topics = []
for topic in lm.state.get_lambda():
    bestn = argsort(topic, topn=10, reverse=True)
topics.append(bestn)

for t in topics:
    print(t)

###############################################################################
# 
# **Step 1**\ : Segmentation
# 
segmented_topics = measure.seg(topics)

for t in segmented_topics:
    print(t)

#
# Unfortunately, the stuff below doesn't work, either :(
#

###############################################################################
# **Step 2**\ : Probability estimation
# 
# Since this is a window-based coherence measure we will perform window based prob estimation
try:
    per_topic_postings, num_windows = measure.prob(
        texts=train_texts, segmented_topics=segmented_topics,
        dictionary=dictionary, window_size=2,
    )
except Exception:
    pass

###############################################################################
# **Step 3**\ : Confirmation Measure
# 
try:
    confirmed_measures = measure.conf(segmented_topics, per_topic_postings, num_windows, normalize=False)
except Exception:
    pass

###############################################################################
# **Step 4**\ : Aggregation
# 

try:
    print(measure.aggr(confirmed_measures))
except Exception:
    pass

###############################################################################
# How this topic model can be used further
# ========================================
# 
# The best topic model here can be used as a standalone for news article classification. However a topic model can also be used as a dimensionality reduction algorithm to feed into a classifier. A good topic model should be able to extract the signal from the noise efficiently, hence improving the performance of the classifier.
# 
