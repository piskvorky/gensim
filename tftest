import gensim,os,time 
from gensim.models.tfword2vec import TfWord2Vec
from gensim.models.word2vec import Word2Vec, Text8Corpus

#TF benchmark
start = time.time()
model = TfWord2Vec("text8", epochs_to_train=5, batch_size=100)
print "Tensorflow:\n" + str(time.time()-start)

#Gensim benchmark
corpus = Text8Corpus("text8")
start = time.time()
model = Word2Vec(corpus)
print "Gensim:\n" + str(time.time()-start)
