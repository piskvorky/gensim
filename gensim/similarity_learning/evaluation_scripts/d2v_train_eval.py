from gensim.similarity_learning import WikiQAExtractor
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import os, pickle
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LabeledLineSentence(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for uid, line in enumerate(self.corpus):
            yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])

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

def cos_sim(vec1, vec2):
    return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

wikiqa_train = WikiQAExtractor(os.path.join("..", "data", "WikiQACorpus", "WikiQA-train.tsv"))
train_data = wikiqa_train.get_preprocessed_corpus()

wikiqa_test = WikiQAExtractor(os.path.join("..", "data", "WikiQACorpus", "WikiQA-test.tsv"))
test_doc_data = wikiqa_test.get_data()

lls = LabeledLineSentence(train_data)

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.025,
                min_count=1,
                dm =1)
  
model.build_vocab(lls)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(lls,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


big_y_true = []
big_y_pred = []

for doc in test_doc_data:
    y_true = []
    y_pred = []

    for query, d, label in doc:
        y_pred.append(cos_sim(model.infer_vector(query), model.infer_vector(d)))
        y_true.append(label)

    big_y_true.append(y_true)
    big_y_pred.append(y_pred)

big_y_pred = np.array(big_y_pred)
big_y_true = np.array(big_y_true)

print("map :", my_map(big_y_true, big_y_pred))

for k in [1, 3, 5, 10, 20]:
    print("ndcg@",k, " : ", mean_ndcg(big_y_true, big_y_pred, k=k))