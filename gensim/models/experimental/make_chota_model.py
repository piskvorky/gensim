from drmm_tks import DRMM_TKS
from gensim.models import Word2Vec
import gensim.downloader as api

# sents = ["The world war was bad".lower().split(), "It was fought in 1996".lower().split(), "Gandhi was born in the 18th century".lower().split(),

#          "He fought for the Indian freedom movement".lower().split(),
#          "Gandhi was assasinated".lower().split()]

'''
w2v = Word2Vec(sentences= ["Hello World".split(), "I am Groot".split(), "give me food".split()], min_count=1, size=10)

# print(w2v.wv)
# print(w2v.most_similar('I'))

kv = w2v.wv

# kv = api.load('glove-wiki-gigaword-50')

queries = ["When was World War 1 fought ?".lower().split(), "When was Gandhi born ?".lower().split()]

docs = [["The world war was bad".lower().split(), "It was fought in 1996".lower().split()],
     ["Gandhi was born in the 18th century".lower().split(), "He fought for the Indian freedom movement".lower().split(),
      "Gandhi was assasinated".lower().split()]]

labels = [[0, 1], [1, 0, 0]]

# model = DRMM_TKS(queries, docs, labels, word_embedding=kv)

model = DRMM_TKS(queries=queries, docs=docs, labels=labels, word_embedding=kv, text_maxlen=10, topk=10)
model.save('drmm_tks')'''

queries = ["When was World War 1 fought ?".lower().split(), "When was Gandhi born ?".lower().split()]

docs = [["The world war was bad".lower().split(), "It was fought in 1996".lower().split()],
     ["Gandhi was born in the 18th century".lower().split(), "He fought for the Indian freedom movement".lower().split(),
      "Gandhi was assasinated".lower().split()]]

labels = [[0, 1], [1, 0, 0]]

from gensim.test.utils import datapath
mod = DRMM_TKS.load(datapath('drmm_tks'))
print(mod.model.summary())
mod.train(queries, docs, labels, text_maxlen=10)
print(mod.predict(queries, docs))