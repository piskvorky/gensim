
# coding: utf-8
"""
online_w2v_tutorial.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # Online word2vec tutorial
#
# So far, word2vec cannot increase the size of vocabulary after initial training. To handle unknown words, not in word2vec vocaburary, you must  retrain updated documents over again.
#
# In this tutorial, we introduce gensim new feature, online vocaburary update. This additional feature overcomes the unknown word problems. Despite after initial training, we can continuously add new vocaburary to the pre-trained word2vec model using this online feature.
#
# This implementation is still beta version at 16/09/04. You can download the beta version of online word2vec implementation in the following repository.

# In[ ]:

# get_ipython().run_cell_magic(u'bash', u'', u'git clone -b online-w2v git@github.com:isohyt/gensim.git')


# In[1]:

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence
from pprint import pprint
from copy import deepcopy
from multiprocessing import cpu_count


# ### Download wikipedia dump files
#
# We use the past and the current version of wiki dump files as online training.

# In[ ]:

# get_ipython().run_cell_magic(u'bash', u'', u'wget https://dumps.wikimedia.org/archive/2010/2010-11/enwiki/20101011/enwiki-20101011-pages-articles.xml.bz2\nwget https://dumps.wikimedia.org/enwiki/20160820/enwiki-20160820-pages-articles.xml.bz2')


# ### Convert two wikipedia dump files
# To avoid alert when convert old verision of wikipedia dump, you should download alternative wikicorpus.py in my repo.

# In[2]:

old, new = [WikiCorpus('enwiki-{0}-pages-articles.xml.bz2'.format(ymd)) for ymd in ['20101011', '20160820']]


# In[3]:

def write_wiki(wiki, name, titles = []):
    with open('{0}.wiki'.format(name), 'wb') as f:
        wiki.metadata = True
        for text, (page_id, title) in wiki.get_texts():
            if title not in titles:
                f.write(b' '.join(text)+b'\n')
                titles.append(title)
    return titles


# In[4]:

old_titles = write_wiki(old, 'old')
all_titles = write_wiki(new, 'new', old_titles)


# In[5]:

oldwiki, newwiki = [LineSentence(f+'.wiki') for f in ['old', 'new']]


# ### Initial training
# At first we train word2vec using "enwiki-20101011-pages-articles.xml.bz2". After that, we update model using "enwiki-20160820-pages-articles.xml.bz2".

# In[6]:

# get_ipython().run_cell_magic(u'time', u'', u"model = Word2Vec(oldwiki, min_count = 0, workers=cpu_count())\n# model = Word2Vec.load('oldmodel')\noldmodel = deepcopy(model)\noldmodel.save('oldmodel')")


# #### Japanese new idol group, ["Babymetal"](https://en.wikipedia.org/wiki/Babymetal), weren't known worldwide in 2010, so that the word, "babymetal", is not in oldmodel vocaburary.
# Note: In recent years, they became the famous idol group not only in Japan. They won many music awards and run world tour.

# In[7]:

try:
    print(oldmodel.most_similar('babymetal'))
except KeyError as e:
    print(e)


# ## Online update
# To use online word2vec feature, set update=True when you use build_vocab using new documents.

# In[8]:

# get_ipython().run_cell_magic(u'time', u'', u"model.build_vocab(newwiki, update=True)\nmodel.train(newwiki)\nmodel.save('newmodel')\n# model = Word2Vec.load('newmodel')")


# #### Model Comparison
# By the online training, the size of vocaburaries are increased about 3 millions.

# In[9]:

for m in ['oldmodel', 'model']:
    print('The vocabulary size of the', m, 'is', len(eval(m).vocab))


# #### After online training, the word, "babymetal", is added in model. This word is simillar with rock and metal bands.

# In[10]:

try:
    pprint(model.most_similar('babymetal'))
except KeyError as e:
    print(e)


# ## The word, "Zootopia", become disney movie through the years.
# In the past, the word, "Zootopia", was used just for an annual summer concert put on by New York top-40 radio station Z100, so that the word, "zootopia", is simillar with music festival.
#
# In 2016, Zootopia is a American 3D computer-animated comedy film released by Walt Disney Pictures. As a result, the word, "zootopia", was often used as Animation films.

# In[11]:

w = 'zootopia'
for m in ['oldmodel', 'model']:
    print('The count of the word,'+w+', is', eval(m).vocab[w].count, 'in', m)
    pprint(eval(m).most_similar(w))
    print('')
