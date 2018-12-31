FastText Notes
==============

The gensim FastText implementation consists of several key classes:

The implementation is split across several submodules:

- models.fasttext
- models.keyedvectors (includes FastText-specific code, not good)
- models.word2vec (superclasses)
- models.base_any2vec (superclasses)

The implementation consists of several key classes:

1. models.fasttext.FastTextVocab: the vocabulary
2. models.keyedvectors.FastTextKeyedVectors: the vectors
3. models.fasttext.FastTextTrainables: the underlying neural network
4. models.fasttext.FastText: ties everything together

FastTextVocab
-------------

Seems to be an entirely redundant class.
Inherits from models.word2vec.Word2VecVocab, adding no new functionality.

FastTextKeyedVectors
--------------------

FastTextTrainables
------------------

[Link](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastTextTrainables)

This is a neural network that learns the vectors for the FastText embedding.
Mostly inherits from its [Word2Vec parent](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecTrainables).
Adds logic for calculating and maintaining ngram weights.

Key attributes:

- hashfxn: function for randomly initializing weights.  Defaults to the built-in hash() 
- layer1_size: The size of the inner layer of the NN.  Equal to the vector dimensionality.  Set in the Word2VecTrainables constructor.
- seed: The random generator seed used in reset_weights and update_weights
- syn1: The inner layer of the NN.  Each row corresponds to a term in the vocabulary.  Columns correspond to weights of the inner layer.  There are layer1_size such weights.  Set in the reset_weights and update_weights methods, only if hierarchical sampling is used.
- syn1neg: Similar to syn1, but only set if negative sampling is used.
- vectors_lockf: A one-dimensional array with one element for each term in the vocab.  Set in reset_weights to an array of ones.
- vectors_vocab_lockf: Similar to vectors_vocab_lockf, ones(len(model.trainables.vectors), dtype=REAL)
- vectors_ngrams_lockf = ones((self.bucket, wv.vector_size), dtype=REAL)

The lockf stuff looks like it gets used by the fast C implementation.

The inheritance hierarchy here is:

1. FastTextTrainables
2. Word2VecTrainables
3. utils.SaveLoad

FastText
--------

Inheritance hierarchy:

1. FastText
2. BaseWordEmbeddingsModel: vocabulary management plus a ton of deprecated attrs
3. BaseAny2VecModel: logging and training functionality
4. utils.SaveLoad: for loading and saving

Lots of attributes (many inherited from superclasses).

From BaseAny2VecModel:

- workers
- vector_size
- epochs
- callbacks
- batch_words
- kv
- vocabulary
- trainables

From BaseWordEmbeddingModel:

- alpha
- min_alpha
- min_alpha_yet_reached
- window
- random
- hs
- negative
- ns_exponent
- cbow_mean
- compute_loss
- running_training_loss
- corpus_count
- corpus_total_words
- neg_labels

FastText attributes:

- wv: FastTextWordVectors.  Used instead of .kv
- 

Logging
-------

The logging seems to be inheritance-based.
It may be better to refactor this using aggregation istead of inheritance in the future.
The benefits would be leaner classes with less responsibilities and better separation of concerns.
