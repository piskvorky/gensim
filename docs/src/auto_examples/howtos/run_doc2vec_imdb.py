r"""
How to reproduce the doc2vec 'Paragraph Vector' paper
=====================================================

Shows how to reproduce results of the "Distributed Representation of Sentences and Documents" paper by Le and Mikolov using Gensim.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Introduction
# ------------
#
# This guide shows you how to reproduce the results of the paper by `Le and
# Mikolov 2014 <https://arxiv.org/pdf/1405.4053.pdf>`_ using Gensim. While the
# entire paper is worth reading (it's only 9 pages), we will be focusing on
# Section 3.2: "Beyond One Sentence - Sentiment Analysis with the IMDB
# dataset".
#
# This guide follows the following steps:
#
# #. Load the IMDB dataset
# #. Train a variety of Doc2Vec models on the dataset
# #. Evaluate the performance of each model using a logistic regression
# #. Examine some of the results directly:
#
# When examining results, we will look for answers for the following questions:
#
# #. Are inferred vectors close to the precalculated ones?
# #. Do close documents seem more related than distant ones?
# #. Do the word vectors show useful similarities?
# #. Are the word vectors from this dataset any good at analogies?
#
# Load corpus
# -----------
#
# Our data for the tutorial will be the `IMDB archive
# <http://ai.stanford.edu/~amaas/data/sentiment/>`_.
# If you're not familiar with this dataset, then here's a brief intro: it
# contains several thousand movie reviews.
#
# Each review is a single line of text containing multiple sentences, for example:
#
# ```
# One of the best movie-dramas I have ever seen. We do a lot of acting in the
# church and this is one that can be used as a resource that highlights all the
# good things that actors can do in their work. I highly recommend this one,
# especially for those who have an interest in acting, as a "must see."
# ```
#
# These reviews will be the **documents** that we will work with in this tutorial.
# There are 100 thousand reviews in total.
#
# #. 25k reviews for training (12.5k positive, 12.5k negative)
# #. 25k reviews for testing (12.5k positive, 12.5k negative)
# #. 50k unlabeled reviews
#
# Out of 100k reviews, 50k have a label: either positive (the reviewer liked
# the movie) or negative.
# The remaining 50k are unlabeled.
#
# Our first task will be to prepare the dataset.
#
# More specifically, we will:
#
# #. Download the tar.gz file (it's only 84MB, so this shouldn't take too long)
# #. Unpack it and extract each movie review
# #. Split the reviews into training and test datasets
#
# First, let's define a convenient datatype for holding data for a single document:
#
# * words: The text of the document, as a ``list`` of words.
# * tags: Used to keep the index of the document in the entire dataset.
# * split: one of ``train``\ , ``test`` or ``extra``. Determines how the document will be used (for training, testing, etc).
# * sentiment: either 1 (positive), 0 (negative) or None (unlabeled document).
#
# This data type is helpful for later evaluation and reporting.
# In particular, the ``index`` member will help us quickly and easily retrieve the vectors for a document from a model.
#
import collections

SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags split sentiment')

###############################################################################
# We can now proceed with loading the corpus.
import io
import re
import tarfile
import os.path

import smart_open
import gensim.utils

def download_dataset(url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
    fname = url.split('/')[-1]

    if os.path.isfile(fname):
       return fname

    # Download the file to local storage first.
    with smart_open.open(url, "rb", ignore_ext=True) as fin:
        with smart_open.open(fname, 'wb', ignore_ext=True) as fout:
            while True:
                buf = fin.read(io.DEFAULT_BUFFER_SIZE)
                if not buf:
                    break
                fout.write(buf)

    return fname

def create_sentiment_document(name, text, index):
    _, split, sentiment_str, _ = name.split('/')
    sentiment = {'pos': 1.0, 'neg': 0.0, 'unsup': None}[sentiment_str]

    if sentiment is None:
        split = 'extra'

    tokens = gensim.utils.to_unicode(text).split()
    return SentimentDocument(tokens, [index], split, sentiment)

def extract_documents():
    fname = download_dataset()

    index = 0

    with tarfile.open(fname, mode='r:gz') as tar:
        for member in tar.getmembers():
            if re.match(r'aclImdb/(train|test)/(pos|neg|unsup)/\d+_\d+.txt$', member.name):
                member_bytes = tar.extractfile(member).read()
                member_text = member_bytes.decode('utf-8', errors='replace')
                assert member_text.count('\n') == 0
                yield create_sentiment_document(member.name, member_text, index)
                index += 1

alldocs = list(extract_documents())

###############################################################################
# Here's what a single document looks like.
print(alldocs[27])

###############################################################################
# Extract our documents and split into training/test sets.
train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
print(f'{len(alldocs)} docs: {len(train_docs)} train-sentiment, {len(test_docs)} test-sentiment')

###############################################################################
# Set-up Doc2Vec Training & Evaluation Models
# -------------------------------------------
#
# We approximate the experiment of Le & Mikolov `"Distributed Representations
# of Sentences and Documents"
# <http://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_ with guidance from
# Mikolov's `example go.sh
# <https://groups.google.com/g/word2vec-toolkit/c/Q49FIrNOQRo/m/J6KG8mUj45sJ>`_::
#
#     ./word2vec -train ../alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
#
# We vary the following parameter choices:
#
# * 100-dimensional vectors, as the 400-d vectors of the paper take a lot of
#   memory and, in our tests of this task, don't seem to offer much benefit
# * Similarly, frequent word subsampling seems to decrease sentiment-prediction
#   accuracy, so it's left out
# * ``cbow=0`` means skip-gram which is equivalent to the paper's 'PV-DBOW'
#   mode, matched in gensim with ``dm=0``
# * Added to that DBOW model are two DM models, one which averages context
#   vectors (\ ``dm_mean``\ ) and one which concatenates them (\ ``dm_concat``\ ,
#   resulting in a much larger, slower, more data-hungry model)
# * A ``min_count=2`` saves quite a bit of model memory, discarding only words
#   that appear in a single doc (and are thus no more expressive than the
#   unique-to-each doc vectors themselves)
#

import multiprocessing
from collections import OrderedDict

import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

from gensim.models.doc2vec import Doc2Vec

common_kwargs = dict(
    vector_size=100, epochs=20, min_count=2,
    sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
)

simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, **common_kwargs),
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, window=10, alpha=0.05, comment='alpha=0.05', **common_kwargs),
    # PV-DM w/ concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs),
]

for model in simple_models:
    model.build_vocab(alldocs)
    print(f"{model} vocabulary scanned & state initialized")

models_by_name = OrderedDict((str(model), model) for model in simple_models)

###############################################################################
# Le and Mikolov note that combining a paragraph vector from Distributed Bag of
# Words (DBOW) and Distributed Memory (DM) improves performance. We will
# follow, pairing the models together for evaluation. Here, we concatenate the
# paragraph vectors obtained from each model with the help of a thin wrapper
# class included in a gensim test module. (Note that this a separate, later
# concatenation of output-vectors than the kind of input-window-concatenation
# enabled by the ``dm_concat=1`` mode above.)
#
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])

###############################################################################
# Predictive Evaluation Methods
# -----------------------------
#
# Given a document, our ``Doc2Vec`` models output a vector representation of the document.
# How useful is a particular model?
# In case of sentiment analysis, we want the ouput vector to reflect the sentiment in the input document.
# So, in vector space, positive documents should be distant from negative documents.
#
# We train a logistic regression from the training set:
#
#   - regressors (inputs): document vectors from the Doc2Vec model
#   - target (outpus): sentiment labels
#
# So, this logistic regression will be able to predict sentiment given a document vector.
#
# Next, we test our logistic regression on the test set, and measure the rate of errors (incorrect predictions).
# If the document vectors from the Doc2Vec model reflect the actual sentiment well, the error rate will be low.
#
# Therefore, the error rate of the logistic regression is indication of *how well* the given Doc2Vec model represents documents as vectors.
# We can then compare different ``Doc2Vec`` models by looking at their error rates.
#

import numpy as np
import statsmodels.api as sm
from random import sample

def logistic_predictor_from_data(train_targets, train_regressors):
    """Fit a statsmodel logistic predictor on supplied data"""
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc.sentiment for doc in train_set]
    train_regressors = [test_model.dv[doc.tags[0]] for doc in train_set]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_regressors = [test_model.dv[doc.tags[0]] for doc in test_set]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_set])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)

###############################################################################
# Bulk Training & Per-Model Evaluation
# ------------------------------------
#
# Note that doc-vector training is occurring on *all* documents of the dataset,
# which includes all TRAIN/TEST/DEV docs.  Because the native document-order
# has similar-sentiment documents in large clumps – which is suboptimal for
# training – we work with once-shuffled copy of the training set.
#
# We evaluate each model's sentiment predictive power based on error rate, and
# the evaluation is done for each model.
#
# (On a 4-core 2.6Ghz Intel Core i7, these 20 passes training and evaluating 3
# main models takes about an hour.)
#
from collections import defaultdict
error_rates = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved

###############################################################################
#
from random import shuffle
shuffled_alldocs = alldocs[:]
shuffle(shuffled_alldocs)

for model in simple_models:
    print(f"Training {model}")
    model.train(shuffled_alldocs, total_examples=len(shuffled_alldocs), epochs=model.epochs)

    print(f"\nEvaluating {model}")
    err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    print("\n%f %s\n" % (err_rate, model))

for model in [models_by_name['dbow+dmm'], models_by_name['dbow+dmc']]:
    print(f"\nEvaluating {model}")
    err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    print(f"\n{err_rate} {model}\n")

###############################################################################
# Achieved Sentiment-Prediction Accuracy
# --------------------------------------
# Compare error rates achieved, best-to-worst
print("Err_rate Model")
for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
    print(f"{rate} {name}")

###############################################################################
# In our testing, contrary to the results of the paper, on this problem,
# PV-DBOW alone performs as good as anything else. Concatenating vectors from
# different models only sometimes offers a tiny predictive improvement – and
# stays generally close to the best-performing solo model included.
#
# The best results achieved here are just around 10% error rate, still a long
# way from the paper's reported 7.42% error rate.
#
# (Other trials not shown, with larger vectors and other changes, also don't
# come close to the paper's reported value. Others around the net have reported
# a similar inability to reproduce the paper's best numbers. The PV-DM/C mode
# improves a bit with many more training epochs – but doesn't reach parity with
# PV-DBOW.)
#

###############################################################################
# Examining Results
# -----------------
#
# Let's look for answers to the following questions:
#
# #. Are inferred vectors close to the precalculated ones?
# #. Do close documents seem more related than distant ones?
# #. Do the word vectors show useful similarities?
# #. Are the word vectors from this dataset any good at analogies?
#

###############################################################################
# Are inferred vectors close to the precalculated ones?
# -----------------------------------------------------
doc_id = np.random.randint(len(simple_models[0].dv))  # Pick random doc; re-run cell for more examples
print(f'for doc {doc_id}...')
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print(f'{model}:\n {model.dv.most_similar([inferred_docvec], topn=3)}')

###############################################################################
# (Yes, here the stored vector from 20 epochs of training is usually one of the
# closest to a freshly-inferred vector for the same words. Defaults for
# inference may benefit from tuning for each dataset or model parameters.)
#

###############################################################################
# Do close documents seem more related than distant ones?
# -------------------------------------------------------
import random

doc_id = np.random.randint(len(simple_models[0].dv))  # pick random doc, re-run cell for more examples
model = random.choice(simple_models)  # and a random model
sims = model.dv.most_similar(doc_id, topn=len(model.dv))  # get *all* similar documents
print(f'TARGET ({doc_id}): «{" ".join(alldocs[doc_id].words)}»\n')
print(f'SIMILAR/DISSIMILAR DOCS PER MODEL {model}%s:\n')
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    s = sims[index]
    i = sims[index][0]
    words = ' '.join(alldocs[i].words)
    print(f'{label} {s}: «{words}»\n')

###############################################################################
# Somewhat, in terms of reviewer tone, movie genre, etc... the MOST
# cosine-similar docs usually seem more like the TARGET than the MEDIAN or
# LEAST... especially if the MOST has a cosine-similarity > 0.5. Re-run the
# cell to try another random target document.
#

###############################################################################
# Do the word vectors show useful similarities?
# ---------------------------------------------
#
import random

word_models = simple_models[:]

def pick_random_word(model, threshold=10):
    # pick a random word with a suitable number of occurences
    while True:
        word = random.choice(model.wv.index_to_key)
        if model.wv.get_vecattr(word, "count") > threshold:
            return word

target_word = pick_random_word(word_models[0])
# or uncomment below line, to just pick a word from the relevant domain:
# target_word = 'comedy/drama'

for model in word_models:
    print(f'target_word: {repr(target_word)} model: {model} similar words:')
    for i, (word, sim) in enumerate(model.wv.most_similar(target_word, topn=10), 1):
        print(f'    {i}. {sim:.2f} {repr(word)}')
    print()

###############################################################################
# Do the DBOW words look meaningless? That's because the gensim DBOW model
# doesn't train word vectors – they remain at their random initialized values –
# unless you ask with the ``dbow_words=1`` initialization parameter. Concurrent
# word-training slows DBOW mode significantly, and offers little improvement
# (and sometimes a little worsening) of the error rate on this IMDB
# sentiment-prediction task, but may be appropriate on other tasks, or if you
# also need word-vectors.
#
# Words from DM models tend to show meaningfully similar words when there are
# many examples in the training data (as with 'plot' or 'actor'). (All DM modes
# inherently involve word-vector training concurrent with doc-vector training.)
#


###############################################################################
# Are the word vectors from this dataset any good at analogies?
# -------------------------------------------------------------

from gensim.test.utils import datapath
questions_filename = datapath('questions-words.txt')

# Note: this analysis takes many minutes
for model in word_models:
    score, sections = model.wv.evaluate_word_analogies(questions_filename)
    correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
    print(f'{model}: {float(correct*100)/(correct+incorrect):0.2f}%% correct ({correct} of {correct+incorrect}')

###############################################################################
# Even though this is a tiny, domain-specific dataset, it shows some meager
# capability on the general word analogies – at least for the DM/mean and
# DM/concat models which actually train word vectors. (The untrained
# random-initialized words of the DBOW model of course fail miserably.)
#
