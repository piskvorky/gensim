FastText Notes
==============

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

Inheritance hierarchy:

1. FastTextKeyedVectors
2. WordEmbeddingsKeyedVectors.  Implements word similarity e.g. cosine similarity, WMD, etc.
3. BaseKeyedVectors (abstract base class)
4. utils.SaveLoad

There are many attributes.

Inherited from BaseKeyedVectors:

- vectors: a 2D numpy array.  Flexible number of rows (0 by default).  Number of columns equals vector dimensionality.
- vocab: a dictionary.  Keys are words.  Items are Vocab instances: these are essentially namedtuples that contain an index and a count.  The former is the index of a term in the entire vocab.  The latter is the number of times the term occurs.
- vector_size (dimensionality)
- index2entity

Inherited from WordEmbeddingsKeyedVectors:

- vectors_norm
- index2word

Added by FastTextKeyedVectors:

- vectors_vocab: 2D array.  Rows are vectors.  Columns correspond to vector dimensions.  Initialized in FastTextTrainables.init_ngrams_weights.  Reset in reset_ngrams_weights.  Referred to as syn0_vocab in fasttext_inner.pyx.  These are vectors for every word in the vocabulary.
- vectors_vocab_norm: looks unused, see _clear_post_train method.
- vectors_ngrams: 2D array.  Each row is a bucket.  Columns correspond to vector dimensions.  Initialized in init_ngrams_weights function.  Initialized in _load_vectors method when reading from native FB binary.  Modified in reset_ngrams_weights method.  This is the first matrix loaded from the native binary files.
- vectors_ngrams_norm: looks unused, see _clear_post_train method.
- buckets_word: A hashmap.  Keyed by the index of a term in the vocab.  Each value is an array, where each element is an integer that corresponds to a bucket.  Initialized in init_ngrams_weights function
- hash2index: A hashmap.  Keys are hashes of ngrams.  Values are the number of ngrams (?).  Initialized in init_ngrams_weights function.
- min_n: minimum ngram length
- max_n: maximum ngram length
- num_ngram_vectors: initialized in the init_ngrams_weights function

The init_ngrams_method looks like an internal method of FastTextTrainables.
It gets called as part of the prepare_weights method, which is effectively part of the FastModel constructor.

The above attributes are initialized to None in the FastTextKeyedVectors class constructor.
Unfortunately, their real initialization happens in an entirely different module, models.fasttext - another indication of poor separation of concerns.

Some questions:

- What is the x_lockf stuff?  Why is it used only by the fast C implementation?
- How are vectors_vocab and vectors_ngrams different?

vectors_vocab contains vectors for entire vocabulary.
vectors_ngrams contains vectors for each _bucket_.


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

Logging
-------

The logging seems to be inheritance-based.
It may be better to refactor this using aggregation istead of inheritance in the future.
The benefits would be leaner classes with less responsibilities and better separation of concerns.
