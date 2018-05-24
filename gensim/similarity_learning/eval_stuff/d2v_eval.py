from gensim.models.doc2vec import Doc2Vec
from gensim.similarity_learning import WikiQAExtractor
import os
import numpy as np

model = Doc2Vec.load('d2v.model')

wikiqa = WikiQAExtractor(os.path.join("..", "data", "WikiQACorpus", "WikiQA-train.tsv"))
data = wikiqa.get_data()

def cos_sim(vec1, vec2):
    return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

big_y_true = []
big_y_pred = []

for doc in data:
    y_true = []
    y_pred = []

    for query, d, label in doc:
        y_pred.append(cos_sim(model.infer_vector(query), model.infer_vector(d)))
        y_true.append(label)

    big_y_true.append(y_true)
    big_y_pred.append(y_pred)

big_y_pred = np.array(big_y_pred)
big_y_true = np.array(big_y_true)

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


def mean_ndcg(big_y_true, big_y_pred, k=10):
    
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

print(my_map(big_y_true, big_y_pred))
print(mean_ndcg(big_y_true, big_y_pred, k=10))
