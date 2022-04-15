r"""
FastText Model
==============

Introduces Gensim's fastText model and demonstrates its use on the Lee Corpus.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Here, we'll learn to work with fastText library for training word-embedding
# models, saving & loading them and performing similarity operations & vector
# lookups analogous to Word2Vec.


###############################################################################
#
# When to use fastText?
# ---------------------
#
# The main principle behind `fastText <https://github.com/facebookresearch/fastText>`_ is that the
# morphological structure of a word carries important information about the meaning of the word.
# Such structure is not taken into account by traditional word embeddings like Word2Vec, which
# train a unique word embedding for every individual word.
# This is especially significant for morphologically rich languages (German, Turkish) in which a
# single word can have a large number of morphological forms, each of which might occur rarely,
# thus making it hard to train good word embeddings.
#
#
# fastText attempts to solve this by treating each word as the aggregation of its subwords.
# For the sake of simplicity and language-independence, subwords are taken to be the character ngrams
# of the word. The vector for a word is simply taken to be the sum of all vectors of its component char-ngrams.
#
#
# According to a detailed comparison of Word2Vec and fastText in
# `this notebook <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Word2Vec_FastText_Comparison.ipynb>`__,
# fastText does significantly better on syntactic tasks as compared to the original Word2Vec,
# especially when the size of the training corpus is small. Word2Vec slightly outperforms fastText
# on semantic tasks though. The differences grow smaller as the size of the training corpus increases.
#
#
# fastText can obtain vectors even for out-of-vocabulary (OOV) words, by summing up vectors for its
# component char-ngrams, provided at least one of the char-ngrams was present in the training data.
#


###############################################################################
#
# Training models
# ---------------
#


###############################################################################
#
# For the following examples, we'll use the Lee Corpus (which you already have if you've installed Gensim) for training our model.
#
#
#
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

# Set file names for train and test data
corpus_file = datapath('lee_background.cor')

model = FastText(vector_size=100)

# build the vocabulary
model.build_vocab(corpus_file=corpus_file)

# train the model
model.train(
    corpus_file=corpus_file, epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words,
)

print(model)


###############################################################################
#
# Training hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^
#


###############################################################################
#
# Hyperparameters for training the model follow the same pattern as Word2Vec. FastText supports the following parameters from the original word2vec:
#
# - model: Training architecture. Allowed values: `cbow`, `skipgram` (Default `cbow`)
# - vector_size: Dimensionality of vector embeddings to be learnt (Default 100)
# - alpha: Initial learning rate (Default 0.025)
# - window: Context window size (Default 5)
# - min_count: Ignore words with number of occurrences below this (Default 5)
# - loss: Training objective. Allowed values: `ns`, `hs`, `softmax` (Default `ns`)
# - sample: Threshold for downsampling higher-frequency words (Default 0.001)
# - negative: Number of negative words to sample, for `ns` (Default 5)
# - epochs: Number of epochs (Default 5)
# - sorted_vocab: Sort vocab by descending frequency (Default 1)
# - threads: Number of threads to use (Default 12)
#
#
# In addition, fastText has three additional parameters:
#
# - min_n: min length of char ngrams (Default 3)
# - max_n: max length of char ngrams (Default 6)
# - bucket: number of buckets used for hashing ngrams (Default 2000000)
#
#
# Parameters ``min_n`` and ``max_n`` control the lengths of character ngrams that each word is broken down into while training and looking up embeddings. If ``max_n`` is set to 0, or to be lesser than ``min_n``\ , no character ngrams are used, and the model effectively reduces to Word2Vec.
#
#
#
# To bound the memory requirements of the model being trained, a hashing function is used that maps ngrams to integers in 1 to K. For hashing these character sequences, the `Fowler-Noll-Vo hashing function <http://www.isthe.com/chongo/tech/comp/fnv>`_ (FNV-1a variant) is employed.
#


###############################################################################
#
# **Note:** You can continue to train your model while using Gensim's native implementation of fastText.
#


###############################################################################
#
# Saving/loading models
# ---------------------
#


###############################################################################
#
# Models can be saved and loaded via the ``load`` and ``save`` methods, just like
# any other model in Gensim.
#


# Save a model trained via Gensim's fastText implementation to temp.
import tempfile
import os
with tempfile.NamedTemporaryFile(prefix='saved_model_gensim-', delete=False) as tmp:
    model.save(tmp.name, separately=[])

# Load back the same model.
loaded_model = FastText.load(tmp.name)
print(loaded_model)

os.unlink(tmp.name)  # demonstration complete, don't need the temp file anymore

###############################################################################
#
# The ``save_word2vec_format`` is also available for fastText models, but will
# cause all vectors for ngrams to be lost.
# As a result, a model loaded in this way will behave as a regular word2vec model.
#


###############################################################################
#
# Word vector lookup
# ------------------
#
#
# All information necessary for looking up fastText words (incl. OOV words) is
# contained in its ``model.wv`` attribute.
#
# If you don't need to continue training your model, you can export & save this `.wv`
# attribute and discard `model`, to save space and RAM.
#
wv = model.wv
print(wv)

#
# FastText models support vector lookups for out-of-vocabulary words by summing up character ngrams belonging to the word.
#
print('night' in wv.key_to_index)

###############################################################################
#
print('nights' in wv.key_to_index)

###############################################################################
#
print(wv['night'])

###############################################################################
#
print(wv['nights'])


###############################################################################
#
# Similarity operations
# ---------------------
#


###############################################################################
#
# Similarity operations work the same way as word2vec. **Out-of-vocabulary words can also be used, provided they have at least one character ngram present in the training data.**
#


print("nights" in wv.key_to_index)

###############################################################################
#
print("night" in wv.key_to_index)

###############################################################################
#
print(wv.similarity("night", "nights"))

###############################################################################
#
# Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.
#


###############################################################################
#
# Other similarity operations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only
print(wv.most_similar("nights"))

###############################################################################
#
print(wv.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))

###############################################################################
#
print(wv.doesnt_match("breakfast cereal dinner lunch".split()))

###############################################################################
#
print(wv.most_similar(positive=['baghdad', 'england'], negative=['london']))

###############################################################################
#
print(wv.evaluate_word_analogies(datapath('questions-words.txt')))

###############################################################################
# Word Movers distance
# ^^^^^^^^^^^^^^^^^^^^
#
# You'll need the optional ``POT`` library for this section, ``pip install POT``.
#
# Let's start with two sentences:
sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()


###############################################################################
# Remove their stopwords.
#
from gensim.parsing.preprocessing import STOPWORDS
sentence_obama = [w for w in sentence_obama if w not in STOPWORDS]
sentence_president = [w for w in sentence_president if w not in STOPWORDS]

###############################################################################
# Compute the Word Movers Distance between the two sentences.
distance = wv.wmdistance(sentence_obama, sentence_president)
print(f"Word Movers Distance is {distance} (lower means closer)")

###############################################################################
# That's all! You've made it to the end of this tutorial.
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('fasttext-logo-color-web.png')
imgplot = plt.imshow(img)
_ = plt.axis('off')
