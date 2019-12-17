doc2vec in gensim – support pretrained word2vec
==================================

This is a forked gensim version, which edits the default doc2vec model to support pretrained word2vec during training doc2vec. It forked from gensim 3.8.

The default doc2vec model in gensim does't support pretrained word2vec model. But according to Jey Han Lau and Timothy Baldwin's paper, [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation(2016)](https://arxiv.org/abs/1607.05368), using pretrained word2vec model usually gets better results in NLP tasks. The author also released a [forked gensim verstion](https://github.com/jhlau/gensim) to perform pretrained embeddings, but it is from a very old Gensim version, which can't be used in gensim 3.8(the latest gensim version when I release this fork).




Features and notice
=============
* 1.Support pretrained word2vec when train doc2vec.
* 2.Support Python 3.
* 3.Support gensim 3.8.
* 4.The pretrainned word2vec model should be C text format.
* 5.The dimension of the pretrained word2vec and doc2vec should be the same.




Use the model
=============

1.Install the forked gensim
---------------------------

* Clone gensim to your machine
> git clone 

* install gensim
> python setup.py install


2. Train your model 
---------------------------

pretrained_emb = "word2vec_pretrained.txt" # This is a pretrained word2vec model of C text format

model = gensim.models.doc2vec.Doc2Vec(corpus_train,  # This is the document corpus to be trained which should meet gensim's format
                                      vector_size=50,
                                      min_count=1, epochs=20, 
                                      dm=0,
                                      pretrained_emb=pretrained_emb) 



Publications
=============

1.Jey Han Lau and Timothy Baldwin (2016). An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation. In Proceedings of the 1st Workshop on Representation Learning for NLP, 2016.

2.[The initial forked gensim version](https://github.com/jhlau/gensim)
