from gensim.similarity_learning import QuoraQPExtractor
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import os, pickle
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

quoraqp = QuoraQPExtractor(os.path.join("..","data", "QuoraQP", "quora_duplicate_questions.tsv"))
corpus = quoraqp.get_preprocessed_corpus()

class LabeledLineSentence(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for uid, line in enumerate(self.corpus):
            yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])

lls = LabeledLineSentence(corpus)

max_epochs = 4
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