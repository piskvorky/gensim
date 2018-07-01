from drmm_tks import DRMM_TKS
import gensim.downloader as api

kv = api.load('glove-wiki-gigaword-50')

# queries = ["When was World War 1 fought ?".lower().split(), \
#                "When was Gandhi born ?".lower().split()]

# docs = [["The world war was bad".lower().split(),\
#         "It was fought in 1996".lower().split()],\
#         ["Gandhi was born in the 18th century".lower().split(),\
#          "He fought for the Indian freedom movement".lower().split(),\
#          "Gandhi was assasinated".lower().split()]]

# labels = [[0, 1],\
#              [1, 0, 0]]

# model = DRMM_TKS(queries, docs, labels, word_embedding=kv)
# model.save('temp.model')

# model.train(queries, docs, labels)

# print(model.predict(queries, docs))

# bbbbbbbbbbbbbbbbb



queries = ["When was World War 1 fought ?".lower().split(),
            "When was Gandhi born ?".lower().split()]

docs = [["The world war was bad".lower().split(),
     "It was fought in 1996".lower().split()],
     ["Gandhi was born in the 18th century".lower().split(),
      "He fought for the Indian freedom movement".lower().split(),
      "Gandhi was assasinated".lower().split()]]

labels = [[0, 1],
         [1, 0, 0]]

model = DRMM_TKS()
model.train(queries, docs, labels, kv, epochs=12, verbose=0)