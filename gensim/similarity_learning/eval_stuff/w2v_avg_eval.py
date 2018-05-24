import numpy as np
import pickle
import os
from pprint import pprint
import random

from gensim.similarity_learning import WikiQAExtractor

wikiqa = WikiQAExtractor(os.path.join("..", "data", "WikiQACorpus", "WikiQA-train.tsv"))
data = wikiqa.get_data()

# w2v = {}

# with open('glove.6B.50d.txt') as f:
#   for line in f:
#       string_array = np.array(line.split()[1:])
#       string_array = [float(i) for i in string_array]
#       w2v[line.split()[0]] = string_array

# with open('w2v.pkl', 'wb') as f:
#   pickle.dump(w2v, f)

# with open('w2v.pkl', 'rb') as f:
#     w2v = pickle.load(f)

# def sent2vec(sentence):
#     vec_sum = []

#     for word in sentence.split():
#         if word in w2v:
#             vec_sum.append(w2v[word])

#     return np.mean(np.array(vec_sum), axis=0)

# def cos_sim(vec1, vec2):
#     return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

# big_y_true = []
# big_y_pred = []

# for doc in data:
#     y_true = []
#     y_pred = []

#     for query, doc, label in doc:
#         # print(sent2vec(query))
#         # print(sent2vec(doc))
#         y_pred.append(cos_sim(sent2vec(query), sent2vec(doc)))
#         y_true.append(label)
#         # break

#     big_y_true.append(y_true)
#     big_y_pred.append(y_pred)

# big_y_pred = np.array(big_y_pred)
# big_y_true = np.array(big_y_true)


# n_correct = 0

# for y_pred, y_true in zip(big_y_pred, big_y_true):
#     if (np.argmax(y_true) == np.argmax(y_pred)):
#         n_correct += 1


# print("Accuracy : ", n_correct/len(big_y_true))
# # Accuracy :  0.2867264997638167


# with open('ys.pkl', 'wb') as f:
#     pickle.dump([big_y_true, big_y_pred], f)





def my_map(big_y_true, big_y_pred):
    
    aps = []

    for y_true, y_pred in zip(big_y_true, big_y_pred):

        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[1], reverse=True)

        avg = 0
        n_relevant = 0

        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                avg += 1./(i + 1.)
                n_relevant += 1

        if n_relevant != 0:
            ap = avg/n_relevant
            aps.append(ap)
            
    return np.mean(np.array(aps))


def my_mean_ndcg(big_y_true, big_y_pred, k=10):
    
    ndcgs = []

    for y_true, y_pred in zip(big_y_true, big_y_pred):

        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[1], reverse=True)
        true_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[0], reverse=True)

        pred_sorted = pred_sorted[:k]
        true_sorted = true_sorted[:k]

        dcg = 0

        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                dcg += 1./np.log2(i + 2)


        idcg = 0

        for i, val in enumerate(true_sorted):
            if val[0] == 1:
                idcg += 1./np.log2(i + 2)

        if idcg != 0:
            ndcgs.append(dcg/idcg)

    return np.mean(np.array(ndcgs))

with open('ys.pkl', 'rb') as f:
    big_y_true, big_y_pred = pickle.load(f)

print(my_map(big_y_true, big_y_pred))
print(my_mean_ndcg(big_y_true, big_y_pred, k=3))
