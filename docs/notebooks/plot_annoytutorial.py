
# coding: utf-8
"""
plot_annoytutorial.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # Similarity Queries using Annoy Tutorial

# This tutorial is about using the [Annoy(Approximate Nearest Neighbors Oh Yeah)]((https://github.com/spotify/annoy "Link to annoy repo") library for similarity queries in gensim

# ## Why use Annoy?
# The current implementation for finding k nearest neighbors in a vector space in gensim has linear complexity via brute force in the number of indexed documents, although with extremely low constant factors. The retrieved results are exact, which is an overkill in many applications: approximate results retrieved in sub-linear time may be enough. Annoy can find approximate nearest neighbors much faster.

# For the following examples, we'll use the Lee Corpus (which you already have if you've installed gensim)
#
# See the [Word2Vec tutorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb) for how to initialize and save this model.

# In[106]:

# Load the model
import gensim, os
from gensim.models.word2vec import Word2Vec

# Set file names for train and test data
test_data_dir = '{0}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'

class MyText(object):
    def __iter__(self):
        for line in open(lee_train_file):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()

sentences = MyText()

model = Word2Vec(sentences, min_count=1)

print(model)


#
# #### Comparing the traditional implementation and the Annoy
#
# N.B. Running the timing cells below more than once gives subsequent timings close to zero, as cached objects are used. To get accurate timings, always run these cells from a freshly started kernel.

# In[107]:

#Set up the model and vector that we are using in the comparison
try:
    from gensim.similarities.index import AnnoyIndexer
except ImportError:
    raise ValueError("SKIP: Please install the annoy indexer")

model.init_sims()
vector = model.syn0norm[0]
annoy_index = AnnoyIndexer(model, 500)


# In[108]:

# get_ipython().run_cell_magic(u'time', u'', u'#Traditional implementation:\nmodel.most_similar([vector], topn=5)')


# In[109]:

# get_ipython().run_cell_magic(u'time', u'', u'#Annoy implementation:\nneighbors = model.most_similar([vector], topn=5, indexer=annoy_index)\nfor neighbor in neighbors:\n    print(neighbor)')


#
# A similarity query using Annoy is significantly faster than using the traditional brute force method
# >**Note**: Initialization time for the annoy indexer was not included in the times. The optimal knn algorithm for you to use will depend on how many queries you need to make and the size of the corpus. If you are making very few similarity queries, the time taken to initialize the annoy indexer will be longer than the time it would take the brute force method to retrieve results. If you are making many queries however, the time it takes to initialize the annoy indexer will be made up for by the incredibly fast retrieval times for queries once the indexer has been initialized

# ## What is Annoy?

# Annoy is an open source library to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data. For our purpose, it is used to find similarity between words or documents in a vector space. [See the tutorial on similarity queries for more information on them](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Similarity_Queries.ipynb).

# ## Getting Started

# First thing to do is to install annoy, by running the following in the command line:
#
# `sudo pip install annoy`
#
# And then set up the logger:

# In[110]:

# import modules & set up logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ## Making a Similarity Query

# ### Creating an indexer
# An instance of `AnnoyIndexer` needs to be created in order to use Annoy in gensim. The `AnnoyIndexer` class is located in `gensim.similarities.index`
#
# `AnnoyIndexer()` takes two parameters:
#
# **`model`**: A `Word2Vec` or `Doc2Vec` model
#
# **`num_trees`**: A positive integer. `num_trees` effects the build time and the index size. **A larger value will give more accurate results, but larger indexes**. More information on what trees in Annoy do can be found [here](https://github.com/spotify/annoy#how-does-it-work). The relationship between `num_trees`, build time, and accuracy will be investigated later in the tutorial.
#

# In[111]:

from gensim.similarities.index import AnnoyIndexer
# 100 trees are being used in this example
annoy_index = AnnoyIndexer(model,100)


# Now that we are ready to make a query, lets find the top 5 most similar words to "army" in the lee corpus. To make a similarity query we call `Word2Vec.most_similar` like we would traditionally, but with an added parameter, `indexer`. The only supported indexer in gensim as of now is Annoy.

# In[112]:

# Derive the vector for the word "army" in our model
vector = model["science"]
# The instance of AnnoyIndexer we just created is passed
approximate_neighbors = model.most_similar([vector], topn=5, indexer=annoy_index)
# Neatly print the approximate_neighbors and their corresponding cosine similarity values
for neighbor in approximate_neighbors:
    print(neighbor)


# ### Analyzing the results

# The closer the cosine similarity of a vector is to 1, the more similar that word is to our query, which was the vector for "army".

# ## Persisting Indexes
# You can save and load your indexes from/to disk to prevent having to construct them each time. This will create two files on disk, _fname_ and _fname.d_. Both files are needed to correctly restore all attributes. Before loading an index, you will have to create an empty AnnoyIndexer object.

# In[113]:

fname = 'index'

# Persist index to disk
annoy_index.save(fname)

# Load index back
if os.path.exists(fname):
    annoy_index2 = AnnoyIndexer()
    annoy_index2.load(fname)
    annoy_index2.model = model


# In[114]:

# Results should be identical to above
vector = model["science"]
approximate_neighbors = model.most_similar([vector], topn=5, indexer=annoy_index2)
for neighbor in approximate_neighbors:
    print (neighbor)


# Be sure to use the same model at load that was used originally, otherwise you will get unexpected behaviors.

# ## Save memory by memory-mapping indices saved to disk

# Annoy library has a useful feature that indices can be memory-mapped from disk. It saves memory when the same index is used by several processes.
#
# Below are two snippets of code. First one has a separate index for each process. The second snipped shares the index between two processes via memory-mapping. The second example uses less total RAM as it is shared.

# In[115]:

# get_ipython().run_cell_magic(u'time', u'', u'\n# Bad example. Two processes load the Word2vec model from disk and create there own Annoy indices from that model. \n\nfrom gensim import models\nfrom gensim.similarities.index import AnnoyIndexer\nfrom multiprocessing import Process\nimport os\nimport psutil\n\nmodel.save(\'/tmp/mymodel\')\n\ndef f(process_id):\n    print \'Process Id: \', os.getpid()\n    process = psutil.Process(os.getpid())\n    new_model = models.Word2Vec.load(\'/tmp/mymodel\')\n    vector = new_model["science"]\n    annoy_index = AnnoyIndexer(new_model,100)\n    approximate_neighbors = new_model.most_similar([vector], topn=5, indexer=annoy_index)\n    for neighbor in approximate_neighbors:\n        print neighbor\n    print \'Memory used by process \'+str(os.getpid())+\'=\', process.memory_info()\n\n# Creating and running two parallel process to share the same index file.\np1 = Process(target=f, args=(\'1\',))\np1.start()\np1.join()\np2 = Process(target=f, args=(\'2\',))\np2.start()\np2.join()')


# In[116]:

# get_ipython().run_cell_magic(u'time', u'', u'\n# Good example. Two processes load both the Word2vec model and index from disk and memory-map the index\n\nfrom gensim import models\nfrom gensim.similarities.index import AnnoyIndexer\nfrom multiprocessing import Process\nimport os\nimport psutil\n\nmodel.save(\'/tmp/mymodel\')\n\ndef f(process_id):\n    print \'Process Id: \', os.getpid()\n    process = psutil.Process(os.getpid())\n    new_model = models.Word2Vec.load(\'/tmp/mymodel\')\n    vector = new_model["science"]\n    annoy_index = AnnoyIndexer()\n    annoy_index.load(\'index\')\n    annoy_index.model = new_model\n    approximate_neighbors = new_model.most_similar([vector], topn=5, indexer=annoy_index)\n    for neighbor in approximate_neighbors:\n        print neighbor\n    print \'Memory used by process \'+str(os.getpid()), process.memory_info()\n\n# Creating and running two parallel process to share the same index file.\np1 = Process(target=f, args=(\'1\',))\np1.start()\np1.join()\np2 = Process(target=f, args=(\'2\',))\np2.start()\np2.join()')


# # Relationship between num_trees and initialization time

# In[117]:

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt, time
x_cor = []
y_cor = []
for x in range(100):
    start_time = time.time()
    AnnoyIndexer(model, x)
    y_cor.append(time.time()-start_time)
    x_cor.append(x)

plt.plot(x_cor, y_cor)
plt.title("num_trees vs initalization time")
plt.ylabel("Initialization time (s)")
plt.xlabel("num_tress")
plt.show()


# Initialization time of the annoy indexer increases in a linear fashion with num_trees. Initialization time will vary from corpus to corpus, in the graph above the lee corpus was used

# # Relationship between num_trees and accuracy

# In[118]:

exact_results = [element[0] for element in model.most_similar([model.syn0norm[0]], topn=100)]
x_axis = []
y_axis = []
for x in range(1,30):
    annoy_index = AnnoyIndexer(model, x)
    approximate_results = model.most_similar([model.syn0norm[0]],topn=100, indexer=annoy_index)
    top_words = [result[0] for result in approximate_results]
    x_axis.append(x)
    y_axis.append(len(set(top_words).intersection(exact_results)))

plt.plot(x_axis, y_axis)
plt.title("num_trees vs accuracy")
plt.ylabel("% accuracy")
plt.xlabel("num_trees")
plt.show()


# This was again done with the lee corpus, a relatively small corpus. Results will vary from corpus to corpus
