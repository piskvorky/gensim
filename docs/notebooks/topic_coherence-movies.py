
# coding: utf-8
"""
topic_coherence-movies.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# ## Benchmark testing of coherence pipeline on Movies dataset:
# ## How to find how well coherence measure matches your manual annotators

# __Introduction__: For the validation of any model adapted from a paper, it is of utmost importance that the results of benchmark testing on the datasets listed in the paper match between the actual implementation (palmetto) and gensim. This coherence pipeline has been implemented from the work done by Roeder et al. The paper can be found [here](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf).
#
# __Approach__ :
# 1. We will use the Movies dataset first. This dataset along with the topics on which the coherence is calculated and the gold (human) ratings on these topics can be found [here](http://139.18.2.164/mroeder/palmetto/datasets/).
# 2. We will then calculate the coherence on these topics using the pipeline implemented in gensim.
# 3. Once we have got all our coherence values on these topics we will calculate the correlation with the human ratings using pearson's r.
# 4. We will compare this final correlation value with the values listed in the paper and see if the pipeline is working as expected.

# In[9]:

import re
import os

from scipy.stats import pearsonr
from datetime import datetime

from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
# %load_ext line_profiler  # This was used for finding out which line was taking maximum time for indirect confirmation measure


# Download the dataset from the link and plug in the location here

# In[10]:

prefix = "/home/devashish/datasets/Movies/movie/"


# In[11]:

import os
if not os.path.exists(prefix):
    raise ValueError("SKIP: Please download the movie corpus.")

start = datetime.now()
texts = []
for fil in os.listdir(prefix):
    for line in open(prefix + fil):
        # lower case all words
        lowered = line.lower()
        #remove punctuation and split into seperate words
        words = re.findall(r'\w+', lowered, flags = re.UNICODE | re.LOCALE)
        texts.append(words)
end = datetime.now()
print("Time taken: %s" % (end - start))


# In[12]:

start = datetime.now()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
end = datetime.now()
print ("Time taken: %s" % (end - start))


# ### Cross validate the numbers

# According to the paper the number of documents should be 108952 with a vocabulary of 1625124. The difference is because of a difference in preprocessing. However the results obtained are still very similar.

# In[13]:

print (len(corpus))
print (dictionary)


# In[14]:

topics = []  # list of 100 topics
for l in open('/home/devashish/datasets/Movies/topicsMovie.txt'):
    topics.append([l.split()])
topics.pop(100)


# In[16]:

human_scores = []
for l in open('/home/devashish/datasets/Movies/goldMovie.txt'):
    human_scores.append(float(l.strip()))


# ### Start off with u_mass coherence measure.

# In[17]:

start = datetime.now()
u_mass = []
flags = []
for n, topic in enumerate(topics):
    try:
        cm = CoherenceModel(topics=topic, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        u_mass.append(cm.get_coherence())
    except KeyError:
        flags.append(n)
end = datetime.now()
print ("Time taken: %s" % (end - start))


# ### Start c_v coherence measure
# This is expected to take much more time since `c_v` uses a sliding window to perform probability estimation and uses the cosine similarity indirect confirmation measure.

# In[18]:

start = datetime.now()
c_v = []
for n, topic in enumerate(topics):
    try:
        cm = CoherenceModel(topics=topic, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
    except KeyError:
        pass
end = datetime.now()
print ("Time taken: %s" % (end - start))


# ### Start c_uci and c_npmi coherence measures
# They should be taking lesser time than c_v but should have a higher correlation than u_mass

# In[19]:

start = datetime.now()
c_uci = []
flags = []
for n, topic in enumerate(topics):
    try:
        cm = CoherenceModel(topics=topic, texts=texts, dictionary=dictionary, coherence='c_uci')
        c_uci.append(cm.get_coherence())
    except KeyError:
        flags.append(n)
end = datetime.now()
print ("Time taken: %s" % (end - start))


# In[20]:

start = datetime.now()
c_npmi = []
for n, topic in enumerate(topics):
    print (n)
    try:
        cm = CoherenceModel(topics=topic, texts=texts, dictionary=dictionary, coherence='c_npmi')
        c_npmi.append(cm.get_coherence())
    except KeyError:
        pass
end = datetime.now()
print ("Time taken: %s" % (end - start))


# In[21]:

final_scores = []
for n, score in enumerate(human_scores):
    if n not in flags:
        final_scores.append(score)


# One topic encountered a KeyError. This was because of a difference in preprocessing due to which one topic word wasn't found in the dictionary

# In[22]:

print (len(u_mass), len(c_v), len(c_uci), len(c_npmi), len(final_scores))
# 1 topic has word(s) that is not in the dictionary. Probably some difference
# in preprocessing


# The values in the paper were:
#
# __`u_mass` correlation__ : 0.093
#
# __`c_v` correlation__    : 0.548
#
# __`c_uci` correlation__  : 0.473
#
# __`c_npmi` correlation__ : 0.438
#
# Our values are also very similar to these values which is good. This validates the correctness of our pipeline.

# In[23]:

print (pearsonr(u_mass, final_scores)[0])
print (pearsonr(c_v, final_scores)[0])
print (pearsonr(c_uci, final_scores)[0])
print (pearsonr(c_npmi, final_scores)[0])


# ### Where do we go now?
#
# - Preprocessing can be improved for this notebook by following the exact process mentioned in [this](http://arxiv.org/pdf/1403.6397v1.pdf) paper.
# - The time required for completing all of these operations can be improved a lot by cythonising the operations.
