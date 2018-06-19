from gensim.similarity_learning import DRMM_TKS_Model
import os

queries = ["When was Abraham Lincoln born ?".split(), 
            "When was the first World War ?".split()]
docs = [
		 ["Abraham Lincoln was the president of the United States of America".split(),
		 "He was born in 1809".split()],
		 ["The first world war was bad".split(),
		 "It was fought in 1914".split(),
		 "There were over a million deaths".split()]
       ]
labels = [[0, 1],
		  [0, 1, 0]]

drmm_tks_model = DRMM_TKS_Model(queries, docs, labels, word_embedding_path=os.path.join('data', 'glove.6B.50d.txt'),
                                    epochs=100)
drmm_tks_model.predict(queries=["When was Abraham Lincoln born ?".split(), "When was Abraham Lincoln born ?".split(),"When was Abraham Lincoln born ?".split()],
                        docs=["Abraham Lincoln was the president of the United States of America".split(),
         "He was born in 1809".split(), "It was fought in 1914".split()])
