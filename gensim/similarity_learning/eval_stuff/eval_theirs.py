import pandas as pd
import numpy as np

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

with open('predict.test.wikiqa.txt') as f:
    df = pd.read_csv(f, sep='\t')

big_y_true = []
big_y_pred = []

for Question, Answer in df.groupby('QuestionID').apply(dict).items():

    this_y_true = []
    this_y_pred = []

    for d, l in zip(Answer['e'], Answer['g']):
        this_y_pred.append(d)
        this_y_true.append(l)

    big_y_pred.append(this_y_pred)
    big_y_true.append(this_y_true)

print("map :", my_map(big_y_true, big_y_pred))

for k in [1, 3, 5, 10, 20]:
    print("ndcg@",k, " : ", my_mean_ndcg(big_y_true, big_y_pred, k=k))
