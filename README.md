doc2vec in gensim – support pretrained word2vec
==================================

The default doc2vec model in Gensim does't support pretrained word2vec model. But according to Jey Han Lau and Timothy Baldwin's paper, [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation(2016)](https://arxiv.org/abs/1607.05368), using pretrained word2vec model usually gets better results in NLP tasks. The author also released a [forked Gensim verstion](https://github.com/jhlau/gensim) to perform pretrained embeddings, but it is from a very old Gensim version, which can't be used in Gensim 3.8(which is the latest Gensim version when I release this fork).




Features
=============





Installation
=============






Publications
=============
Jey Han Lau and Timothy Baldwin (2016). An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation. In Proceedings of the 1st Workshop on Representation Learning for NLP, 2016.
