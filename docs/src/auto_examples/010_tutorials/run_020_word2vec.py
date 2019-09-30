r"""
Word2Vec Model
==============

Introduces Gensim's Word2Vec model and demonstrates its use on the Lee Corpus.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# In case you missed the buzz, word2vec is a widely featured as a member of the
# “new wave” of machine learning algorithms based on neural networks, commonly
# referred to as "deep learning" (though word2vec itself is rather shallow).
# Using large amounts of unannotated plain text, word2vec learns relationships
# between words automatically. The output are vectors, one vector per word,
# with remarkable linear relationships that allow us to do things like:
#
# * vec("king") - vec("man") + vec("woman") =~ vec("queen")
# * vec("Montreal Canadiens") – vec("Montreal") + vec("Toronto") =~ vec("Toronto Maple Leafs").
#
# Word2vec is very useful in `automatic text tagging
# <https://github.com/RaRe-Technologies/movie-plots-by-genre>`_\ , recommender
# systems and machine translation.
#
# This tutorial:
#
# #. Introduces ``Word2Vec`` as an improvement over traditional bag-of-words
# #. Shows off a demo of ``Word2Vec`` using a pre-trained model
# #. Demonstrates training a new model from your own data
# #. Demonstrates loading and saving models
# #. Introduces several training parameters and demonstrates their effect
# #. Discusses memory requirements
# #. Visualizes Word2Vec embeddings by applying dimensionality reduction
#
# Review: Bag-of-words
# --------------------
#
# .. Note:: Feel free to skip these review sections if you're already familiar with the models.
#
# You may be familiar with the `bag-of-words model
# <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ from the
# :ref:`core_concepts_vector` section.
# This model transforms each document to a fixed-length vector of integers.
# For example, given the sentences:
#
# - ``John likes to watch movies. Mary likes movies too.``
# - ``John also likes to watch football games. Mary hates football.``
#
# The model outputs the vectors:
#
# - ``[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]``
# - ``[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]``
#
# Each vector has 10 elements, where each element counts the number of times a
# particular word occurred in the document.
# The order of elements is arbitrary.
# In the example above, the order of the elements corresponds to the words:
# ``["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]``.
#
# Bag-of-words models are surprisingly effective, but have several weaknesses.
#
# First, they lose all information about word order: "John likes Mary" and
# "Mary likes John" correspond to identical vectors. There is a solution: bag
# of `n-grams <https://en.wikipedia.org/wiki/N-gram>`__
# models consider word phrases of length n to represent documents as
# fixed-length vectors to capture local word order but suffer from data
# sparsity and high dimensionality.
#
# Second, the model does not attempt to learn the meaning of the underlying
# words, and as a consequence, the distance between vectors doesn't always
# reflect the difference in meaning.  The ``Word2Vec`` model addresses this
# second problem.
#
# Introducing: the ``Word2Vec`` Model
# -----------------------------------
#
# ``Word2Vec`` is a more recent model that embeds words in a lower-dimensional
# vector space using a shallow neural network. The result is a set of
# word-vectors where vectors close together in vector space have similar
# meanings based on context, and word-vectors distant to each other have
# differing meanings. For example, ``strong`` and ``powerful`` would be close
# together and ``strong`` and ``Paris`` would be relatively far.
#
# The are two versions of this model and :py:class:`~gensim.models.word2vec.Word2Vec`
# class implements them both:
#
# 1. Skip-grams (SG)
# 2. Continuous-bag-of-words (CBOW)
#
# .. Important::
#   Don't let the implementation details below scare you.
#   They're advanced material: if it's too much, then move on to the next section.
#
# The `Word2Vec Skip-gram <http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model>`__
# model, for example, takes in pairs (word1, word2) generated by moving a
# window across text data, and trains a 1-hidden-layer neural network based on
# the synthetic task of given an input word, giving us a predicted probability
# distribution of nearby words to the input. A virtual `one-hot
# <https://en.wikipedia.org/wiki/One-hot>`__ encoding of words
# goes through a 'projection layer' to the hidden layer; these projection
# weights are later interpreted as the word embeddings. So if the hidden layer
# has 300 neurons, this network will give us 300-dimensional word embeddings.
#
# Continuous-bag-of-words Word2vec is very similar to the skip-gram model. It
# is also a 1-hidden-layer neural network. The synthetic training task now uses
# the average of multiple input context words, rather than a single word as in
# skip-gram, to predict the center word. Again, the projection weights that
# turn one-hot words into averageable vectors, of the same width as the hidden
# layer, are interpreted as the word embeddings.
#

###############################################################################
# Word2Vec Demo
# -------------
#
# To see what ``Word2Vec`` can do, let's download a pre-trained model and play
# around with it. We will fetch the Word2Vec model trained on part of the
# Google News dataset, covering approximately 3 million words and phrases. Such
# a model can take hours to train, but since it's already available,
# downloading and loading it with Gensim takes minutes.
#
# .. Important::
#   The model is approximately 2GB, so you'll need a decent network connection
#   to proceed.  Otherwise, skip ahead to the "Training Your Own Model" section
#   below.
#
# You may also check out an `online word2vec demo
# <http://radimrehurek.com/2014/02/word2vec-tutorial/#app>`_ where you can try
# this vector algebra for yourself. That demo runs ``word2vec`` on the
# **entire** Google News dataset, of **about 100 billion words**.
#
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

###############################################################################
# We can easily obtain vectors for terms the model is familiar with:
#
vec_king = wv['king']

###############################################################################
# Unfortunately, the model is unable to infer vectors for unfamiliar words.
# This is one limitation of Word2Vec: if this limitation matters to you, check
# out the FastText model.
#
try:
    vec_weapon = wv['cameroon']
except KeyError:
    pass
else:
    raise RuntimeError('expected to trip over a KeyError')

###############################################################################
# Moving on, ``Word2Vec`` supports several word similarity tasks out of the
# box.  You can see how the similarity intuitively decreases as the words get
# less and less similar.
#
pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))

###############################################################################
# Print the 5 most similar words to "car" or "minivan"
print(wv.most_similar(positive=['car', 'minivan'], topn=5))

###############################################################################
# Which of the below does not belong in the sequence?
print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))

###############################################################################
# Training Your Own Model
# -----------------------
#
# To start, you'll need some data for training the model.  For the following
# examples, we'll use the `Lee Corpus
# <https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/lee_background.cor>`_
# (which you already have if you've installed gensim).
#
# This corpus is small enough to fit entirely in memory, but we'll implement a
# memory-friendly iterator that reads it line-by-line to demonstrate how you
# would handle a larger corpus.
#

from gensim.test.utils import datapath


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()

###############################################################################
# If we wanted to do any custom preprocessing, e.g. decode a non-standard
# encoding, lowercase, remove numbers, extract named entities... All of this can
# be done inside the ``MyCorpus`` iterator and ``word2vec`` doesn’t need to
# know. All that is required is that the input yields one sentence (list of
# utf8 words) after another.
#
# Let's go ahead and train a model on our corpus.  Don't worry about the
# training parameters much for now, we'll revisit them later.
#
import gensim.models

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

###############################################################################
# Once we have our model, we can use it in the same way as in the demo above.
#
# The main part of the model is ``model.wv``\ , where "wv" stands for "word vectors".
#
vec_king = model.wv['king']

###############################################################################
# Storing and loading models
# --------------------------
#
# You'll notice that training non-trivial models can take time.  Once you've
# trained your model and it works as expected, you can save it to disk.  That
# way, you don't have to spend time training it all over again later.
#
# You can store/load models using the standard gensim methods:
#
import tempfile

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
    #
    # To load a saved model:
    #
    new_model = gensim.models.Word2Vec.load(temporary_filepath)

###############################################################################
# which uses pickle internally, optionally ``mmap``\ ‘ing the model’s internal
# large NumPy matrices into virtual memory directly from disk files, for
# inter-process memory sharing.
#
# In addition, you can load models created by the original C tool, both using
# its text and binary formats::
#
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)
#   # using gzipped/bz2 input works too, no need to unzip
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
#


###############################################################################
# Training Parameters
# -------------------
#
# ``Word2Vec`` accepts several parameters that affect both training speed and quality.
#
# min_count
# ---------
#
# ``min_count`` is for pruning the internal dictionary. Words that appear only
# once or twice in a billion-word corpus are probably uninteresting typos and
# garbage. In addition, there’s not enough data to make any meaningful training
# on those words, so it’s best to ignore them:
#
# default value of min_count=5
model = gensim.models.Word2Vec(sentences, min_count=10)

###############################################################################
#
# size
# ----
#
# ``size`` is the number of dimensions (N) of the N-dimensional space that
# gensim Word2Vec maps the words onto.
#
# Bigger size values require more training data, but can lead to better (more
# accurate) models. Reasonable values are in the tens to hundreds.
#

# default value of size=100
model = gensim.models.Word2Vec(sentences, size=200)

###############################################################################
# workers
# -------
#
# ``workers`` , the last of the major parameters (full list `here
# <http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_)
# is for training parallelization, to speed up training:
#

# default value of workers=3 (tutorial says 1...)
model = gensim.models.Word2Vec(sentences, workers=4)

###############################################################################
# The ``workers`` parameter only has an effect if you have `Cython
# <http://cython.org/>`_ installed. Without Cython, you’ll only be able to use
# one core because of the `GIL
# <https://wiki.python.org/moin/GlobalInterpreterLock>`_ (and ``word2vec``
# training will be `miserably slow
# <http://rare-technologies.com/word2vec-in-python-part-two-optimizing/>`_\ ).
#

###############################################################################
# Memory
# ------
#
# At its core, ``word2vec`` model parameters are stored as matrices (NumPy
# arrays). Each array is **#vocabulary** (controlled by min_count parameter)
# times **#size** (size parameter) of floats (single precision aka 4 bytes).
#
# Three such matrices are held in RAM (work is underway to reduce that number
# to two, or even one). So if your input contains 100,000 unique words, and you
# asked for layer ``size=200``\ , the model will require approx.
# ``100,000*200*4*3 bytes = ~229MB``.
#
# There’s a little extra memory needed for storing the vocabulary tree (100,000 words would take a few megabytes), but unless your words are extremely loooong strings, memory footprint will be dominated by the three matrices above.
#


###############################################################################
# Evaluating
# ----------
#
# ``Word2Vec`` training is an unsupervised task, there’s no good way to
# objectively evaluate the result. Evaluation depends on your end application.
#
# Google has released their testing set of about 20,000 syntactic and semantic
# test examples, following the “A is to B as C is to D” task. It is provided in
# the 'datasets' folder.
#
# For example a syntactic analogy of comparative type is bad:worse;good:?.
# There are total of 9 types of syntactic comparisons in the dataset like
# plural nouns and nouns of opposite meaning.
#
# The semantic questions contain five types of semantic analogies, such as
# capital cities (Paris:France;Tokyo:?) or family members
# (brother:sister;dad:?).
#

###############################################################################
# Gensim supports the same evaluation set, in exactly the same format:
#
model.accuracy('./datasets/questions-words.txt')

###############################################################################
#
# This ``accuracy`` takes an `optional parameter
# <http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.accuracy>`_
# ``restrict_vocab`` which limits which test examples are to be considered.
#


###############################################################################
# In the December 2016 release of Gensim we added a better way to evaluate semantic similarity.
#
# By default it uses an academic dataset WS-353 but one can create a dataset
# specific to your business based on it. It contains word pairs together with
# human-assigned similarity judgments. It measures the relatedness or
# co-occurrence of two words. For example, 'coast' and 'shore' are very similar
# as they appear in the same context. At the same time 'clothes' and 'closet'
# are less similar because they are related but not interchangeable.
#
model.evaluate_word_pairs(datapath('wordsim353.tsv'))

###############################################################################
# Important::
#   Good performance on Google's or WS-353 test set doesn’t mean word2vec will
#   work well in your application, or vice versa. It’s always best to evaluate
#   directly on your intended task. For an example of how to use word2vec in a
#   classifier pipeline, see this `tutorial
#   <https://github.com/RaRe-Technologies/movie-plots-by-genre>`_.
#

###############################################################################
# Online training / Resuming training
# -----------------------------------
#
# Advanced users can load a model and continue training it with more sentences
# and `new vocabulary words <online_w2v_tutorial.ipynb>`_:
#
model = gensim.models.Word2Vec.load(temporary_filepath)
more_sentences = [
    ['Advanced', 'users', 'can', 'load', 'a', 'model',
     'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']
]
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

# cleaning up temporary file
import os
os.remove(temporary_filepath)

###############################################################################
# You may need to tweak the ``total_words`` parameter to ``train()``,
# depending on what learning rate decay you want to simulate.
#
# Note that it’s not possible to resume training with models generated by the C
# tool, ``KeyedVectors.load_word2vec_format()``. You can still use them for
# querying/similarity, but information vital for training (the vocab tree) is
# missing there.
#

###############################################################################
# Training Loss Computation
# -------------------------
#
# The parameter ``compute_loss`` can be used to toggle computation of loss
# while training the Word2Vec model. The computed loss is stored in the model
# attribute ``running_training_loss`` and can be retrieved using the function
# ``get_latest_training_loss`` as follows :
#

# instantiating and training the Word2Vec model
model_with_loss = gensim.models.Word2Vec(
    sentences,
    min_count=1,
    compute_loss=True,
    hs=0,
    sg=1,
    seed=42
)

# getting the training loss value
training_loss = model_with_loss.get_latest_training_loss()
print(training_loss)

###############################################################################
# Benchmarks
# ----------
#
# Let's run some benchmarks to see effect of the training loss computation code
# on training time.
#
# We'll use the following data for the benchmarks:
#
# #. Lee Background corpus: included in gensim's test data
# #. Text8 corpus.  To demonstrate the effect of corpus size, we'll look at the
#    first 1MB, 10MB, 50MB of the corpus, as well as the entire thing.
#

import io
import os
import os.path

import gensim.models.word2vec
import gensim.downloader as api
import smart_open


def head(path, size):
    with smart_open.open(path) as fin:
        return io.StringIO(fin.read(size))


def generate_input_data():
    lee_path = datapath('lee_background.cor')
    ls = gensim.models.word2vec.LineSentence(lee_path)
    ls.name = '25kB'
    yield ls

    text8_path = api.load('text8').fn
    labels = ('1MB', '10MB', '50MB', '100MB')
    sizes = (1024 ** 2, 10 * 1024 ** 2, 50 * 1024 ** 2, 100 * 1024 ** 2)
    for l, s in zip(labels, sizes):
        ls = gensim.models.word2vec.LineSentence(head(text8_path, s))
        ls.name = l
        yield ls


input_data = list(generate_input_data())

###############################################################################
# We now compare the training time taken for different combinations of input
# data and model training parameters like ``hs`` and ``sg``.
#
# For each combination, we repeat the test several times to obtain the mean and
# standard deviation of the test duration.
#

# Temporarily reduce logging verbosity
logging.basicConfig(level=logging.ERROR)

import time
import numpy as np
import pandas as pd

train_time_values = []
seed_val = 42
sg_values = [0, 1]
hs_values = [0, 1]

fast = True
if fast:
    input_data_subset = input_data[:3]
else:
    input_data_subset = input_data


for data in input_data_subset:
    for sg_val in sg_values:
        for hs_val in hs_values:
            for loss_flag in [True, False]:
                time_taken_list = []
                for i in range(3):
                    start_time = time.time()
                    w2v_model = gensim.models.Word2Vec(
                        data,
                        compute_loss=loss_flag,
                        sg=sg_val,
                        hs=hs_val,
                        seed=seed_val
                    )
                    time_taken_list.append(time.time() - start_time)

                time_taken_list = np.array(time_taken_list)
                time_mean = np.mean(time_taken_list)
                time_std = np.std(time_taken_list)
          
                d = {
                    'train_data': data.name,
                    'compute_loss': loss_flag,
                    'sg': sg_val,
                    'hs': hs_val,
                    'mean': time_mean,
                    'std': time_std
                }
                print(d)
                train_time_values.append(d)

train_times_table = pd.DataFrame(train_time_values)
train_times_table = train_times_table.sort_values(
    by=['train_data', 'sg', 'hs', 'compute_loss'],
    ascending=[False, False, True, False]
)
print(train_times_table)

###############################################################################
# Adding Word2Vec "model to dict" method to production pipeline
# -------------------------------------------------------------
#
# Suppose, we still want more performance improvement in production.
#
# One good way is to cache all the similar words in a dictionary.
#
# So that next time when we get the similar query word, we'll search it first in the dict.
#
# And if it's a hit then we will show the result directly from the dictionary.
#
# otherwise we will query the word and then cache it so that it doesn't miss next time.
#


# re-enable logging
logging.basicConfig(level=logging.INFO)

most_similars_precalc = {word : model.wv.most_similar(word) for word in model.wv.index2word}
for i, (key, value) in enumerate(most_similars_precalc.items()):
    if i == 3:
        break
    print(key, value)

###############################################################################
# Comparison with and without caching
# -----------------------------------
#
# for time being lets take 4 words randomly
#
import time
words = ['voted', 'few', 'their', 'around']

###############################################################################
# Without caching
#
start = time.time()
for word in words:
    result = model.wv.most_similar(word)
    print(result)
end = time.time()
print(end-start)

###############################################################################
# Now with caching
#
start = time.time()
for word in words:
    if 'voted' in most_similars_precalc:
        result = most_similars_precalc[word]
        print(result)
    else:
        result = model.wv.most_similar(word)
        most_similars_precalc[word] = result
        print(result)

end = time.time()
print(end-start)

###############################################################################
# Clearly you can see the improvement but this difference will be even larger
# when we take more words in the consideration.
#

###############################################################################
#
# Visualising the Word Embeddings
# -------------------------------
#
# The word embeddings made by the model can be visualised by reducing
# dimensionality of the words to 2 dimensions using tSNE.
#
# Visualisations can be used to notice semantic and syntactic trends in the data.
#
# Example:
#
# * Semantic: words like cat, dog, cow, etc. have a tendency to lie close by
# * Syntactic: words like run, running or cut, cutting lie close together.
#
# Vector relations like vKing - vMan = vQueen - vWoman can also be noticed.
#
# .. Important::
#   The model used for the visualisation is trained on a small corpus. Thus
#   some of the relations might not be so clear.
#
# .. Important::
#   Beware: This sort of dimensionality reduction comes at the cost of loss of
#   information.
#

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label some random data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)

###############################################################################
# Conclusion
# ----------
#
# In this tutorial we learned how to train word2vec models on your custom data
# and also how to evaluate it. Hope that you too will find this popular tool
# useful in your Machine Learning tasks!
#
# Links
# -----
#
# - API docs: :py:mod:`gensim.models.word2vec`
# - `Original C toolkit and word2vec papers by Google <https://code.google.com/archive/p/word2vec/>`_.
#
