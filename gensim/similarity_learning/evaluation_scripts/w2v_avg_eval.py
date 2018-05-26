import numpy as np
import pickle
import os
from pprint import pprint
import random

from gensim.similarity_learning import WikiQAExtractor

def accuracy(Y_true, Y_pred):
    n_correct = 0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if (np.argmax(y_true) == np.argmax(y_pred)):
            n_correct += 1

    return n_correct/len(Y_true)



def my_map(Y_true, Y_pred):
    average_precisions = []

    for y_true, y_pred in zip(Y_true, Y_pred):

        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[1], reverse=True)

        avg = 0
        n_relevant = 0

        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                avg += 1./(i + 1.)
                n_relevant += 1

        if n_relevant != 0:
            ap = avg/n_relevant
            average_precisions.append(ap)
            
    return np.mean(np.array(average_precisions))


def mean_ndcg(Y_true, Y_pred, k=10):
    
    ndcgs = []

    for y_true, y_pred in zip(Y_true, Y_pred):

        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[1], reverse=True)
        true_sorted = sorted(zip(y_true, y_pred), key=lambda x:x[0], reverse=True)

        # make cutoff for @k
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

def sent2vec(w2v, sentence):
    vec_sum = []

    for word in sentence.split():
        if word in w2v:
            vec_sum.append(w2v[word])

    return np.mean(np.array(vec_sum), axis=0)

def cos_sim(vec1, vec2):
    return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


if __name__ == '__main__':


    w2v = {}

    with open('glove.6B.50d.txt') as f:
      for line in f:
          string_array = np.array(line.split()[1:])
          string_array = [float(i) for i in string_array]
          w2v[line.split()[0]] = string_array


    # if you'd like to save it and load it instead of calculating it each time
    # with open('w2v.pkl', 'wb') as f:
    #   pickle.dump(w2v, f)
    # with open('w2v.pkl', 'rb') as f:
    #     w2v = pickle.load(f)

    wikiqa = WikiQAExtractor(os.path.join("..", "data", "WikiQACorpus", "WikiQA-train.tsv"))
    data = wikiqa.get_data()

    Y_true = []
    Y_pred = []

    for query_doc_group in data:
        y_true = []
        y_pred = []

        for query, doc, label in query_doc_group:
            y_pred.append(cos_sim(sent2vec(w2v, query), sent2vec(w2v, doc)))
            y_true.append(label)

        Y_true.append(y_true)
        Y_pred.append(y_pred)

    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    # If you want to save and load it once the initial calculation is done
    # with open('ys.pkl', 'wb') as f:
    #     pickle.dump([Y_true, Y_pred], f)
    # with open('ys.pkl', 'rb') as f:
    #     Y_true, Y_pred = pickle.load(f)


    print("Accuracy : ", accuracy(Y_true, Y_pred))
    # Accuracy :  0.2867264997638167

    print("map :", my_map(Y_true, Y_pred))

    for k in [1, 3, 5, 10, 20]:
        print("ndcg@",k, " : ", mean_ndcg(Y_true, Y_pred, k=k))

# Accuracy :  0.2867264997638167
# map : 0.525039852586227
# ndcg@ 1  :  0.37844036697247707
# ndcg@ 3  :  0.5288832435950492
# ndcg@ 5  :  0.5938376161901073
# ndcg@ 10  :  0.6426858123695702
# ndcg@ 20  :  0.6578442782634695