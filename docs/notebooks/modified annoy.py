# -*- coding: utf-8 -*-
"""
=================================
Plotting the exponential function1
=================================

A simple example for ploting two figures of a exponential
function in order to test the autonomy of the gallery
stacking multiple images.
"""
import gensim, os
from gensim.models.word2vec import Word2Vec

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'

class MyText(object):
    def __iter__(self):
        for line in open(lee_train_file):
            yield line.lower().split()

sentences = MyText()

model = Word2Vec(sentences, min_count=1)

print(model)


try:
    from gensim.similarities.index import AnnoyIndexer
except ImportError:
    raise ValueError("SKIP: Please install the annoy indexer")

model.init_sims()
vector = model.syn0norm[0]
annoy_index = AnnoyIndexer(model, 500)



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim.similarities.index import AnnoyIndexer
annoy_index = AnnoyIndexer(model,100)


vector = model["science"]
approximate_neighbors = model.most_similar([vector], topn=5, indexer=annoy_index)
for neighbor in approximate_neighbors:
    print(neighbor)

fname = 'index'

annoy_index.save(fname)

if os.path.exists(fname):
    annoy_index2 = AnnoyIndexer()
    annoy_index2.load(fname)
    annoy_index2.model = model

vector = model["science"]
approximate_neighbors = model.most_similar([vector], topn=5, indexer=annoy_index2)
for neighbor in approximate_neighbors:
    print (neighbor)

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
