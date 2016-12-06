
# coding: utf-8
"""
doc2vec-wikipedia.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # Doc2Vec to wikipedia articles

# We conduct the replication to **Document Embedding with Paragraph Vectors** (http://arxiv.org/abs/1507.07998).
# In this paper, they showed only DBOW results to Wikipedia data. So we replicate this experiments using not only DBOW but also DM.

# ## Basic Setup

# Let's import Doc2Vec module.

# In[1]:

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing


# ## Preparing the corpus

# First, download the dump of all Wikipedia articles from [here](http://download.wikimedia.org/enwiki/) (you want the file enwiki-latest-pages-articles.xml.bz2, or enwiki-YYYYMMDD-pages-articles.xml.bz2 for date-specific dumps).
#
# Second, convert the articles to WikiCorpus. WikiCorpus construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.
#
# For more details on WikiCorpus, you should access [Corpus from a Wikipedia dump](https://radimrehurek.com/gensim/corpora/wikicorpus.html).

# In[2]:

wiki = WikiCorpus("enwiki-latest-pages-articles.xml.bz2")
#wiki = WikiCorpus("enwiki-YYYYMMDD-pages-articles.xml.bz2")


# Define **TaggedWikiDocument** class to convert WikiCorpus into suitable form for Doc2Vec.

# In[3]:

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])


# In[4]:

documents = TaggedWikiDocument(wiki)


# ## Preprocessing
# To set the same vocabulary size with original papar. We first calculate the optimal **min_count** parameter.

# In[5]:

pre = Doc2Vec(min_count=0)
pre.scan_vocab(documents)


# In[6]:

for num in range(0, 20):
    print('min_count: {0}, size of vocab: '.format(num), pre.scale_vocab(min_count=num, dry_run=True)['memory']['vocab']/700)


# In the original paper, they set the vocabulary size 915,715. It seems similar size of vocabulary if we set min_count = 19. (size of vocab = 898,725)

# ## Training the Doc2Vec Model
# To train Doc2Vec model by several method, DBOW and DM, we define the list of models.

# In[7]:

cores = multiprocessing.cpu_count()

models = [
    # PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter =10, workers=cores),
]


# In[8]:

models[0].build_vocab(documents)
print(str(models[0]))
models[1].reset_from(models[0])
print(str(models[1]))


# Now we’re ready to train Doc2Vec of the English Wikipedia.

# In[9]:

# for model in models:
#     get_ipython().magic(u'%time model.train(documents)')


# ## Similarity interface

# After that, let's test both models! DBOW model show the simillar results with the original paper. First, calculating cosine simillarity of "Machine learning" using Paragraph Vector. Word Vector and Document Vector are separately stored. We have to add .docvecs after model name to extract Document Vector from Doc2Vec Model.

# In[10]:

for model in models:
    print(str(model))
    pprint(model.docvecs.most_similar(positive=["Machine learning"], topn=20))


# DBOW model interpret the word 'Machine Learning' as a part of Computer Science field, and DM model as Data Science related field.
#
# Second, calculating cosine simillarity of "Lady Gaga" using Paragraph Vector.

# In[11]:

for model in models:
    print(str(model))
    pprint(model.docvecs.most_similar(positive=["Lady Gaga"], topn=10))


# DBOW model reveal the similar singer in the U.S., and DM model understand that many of Lady Gaga's songs are similar with the word "Lady Gaga".
#
# Third, calculating cosine simillarity of "Lady Gaga" - "American" + "Japanese" using Document vector and Word Vectors. "American" and "Japanese" are Word Vectors, not Paragraph Vectors. Word Vectors are already converted to lowercases by WikiCorpus.

# In[12]:

for model in models:
    print(str(model))
    vec = [model.docvecs["Lady Gaga"] - model["american"] + model["japanese"]]
    pprint([m for m in model.docvecs.most_similar(vec, topn=11) if m[0] != "Lady Gaga"])


# As a result, DBOW model demonstrate the similar artists with Lady Gaga in Japan such as 'Perfume', which is the Most famous Idol in Japan. On the other hand, DM model results don't include the Japanese aritsts in top 10 simillar documents. It's almost same with no vector calculated results.
#
# This results demonstrate that DBOW employed in the original paper is outstanding for calculating the similarity between Document Vector and Word Vector.
