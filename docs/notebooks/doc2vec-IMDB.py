# -*- coding: utf-8 -*-
"""
Gallery of Examples
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # gensim doc2vec & IMDB sentiment dataset

# TODO: section on introduction & motivation
#
# TODO: prerequisites + dependencies (statsmodels, patsy, ?)

# ## Load corpus

# Fetch and prep exactly as in Mikolov's go.sh shell script. (Note this cell tests for existence of required files, so steps won't repeat once the final summary file (`aclImdb/alldata-id.txt`) is available alongside this notebook.)

# In[2]:

import locale
import glob
import os.path
import requests
import tarfile

dirname = 'aclImdb'
filename = 'aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL, 'C')


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


if not os.path.isfile('aclImdb/alldata-id.txt'):
    if not os.path.isdir(dirname):
        if not os.path.isfile(filename):
            # Download IMDB archive
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)

        tar = tarfile.open(filename, mode='r')
        tar.extractall()
        tar.close()

    # Concat and normalize test/train data
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
    alldata = u''

    for fol in folders:
        temp = u''
        output = fol.replace('/', '-') + '.txt'

        # Is there a better pattern to use?
        txt_files = glob.glob('/'.join([dirname, fol, '*.txt']))

        for txt in txt_files:
            with open(txt, 'r', encoding='utf-8') as t:
                control_chars = [chr(0x85)]
                t_clean = t.read()

                for c in control_chars:
                    t_clean = t_clean.replace(c, ' ')

                temp += t_clean

            temp += "\n"

        temp_norm = normalize_text(temp)
        with open('/'.join([dirname, output]), 'w', encoding='utf-8') as n:
            n.write(temp_norm)

        alldata += temp_norm

    with open('/'.join([dirname, 'alldata-id.txt']), 'w', encoding='utf-8') as f:
        for idx, line in enumerate(alldata.splitlines()):
            num_line = "_*{0} {1}\n".format(idx, line)
            f.write(num_line)


# In[3]:

import os.path
assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"


# The data is small enough to be read into memory.

# In[1]:

import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # will hold all docs in original order
with open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # for reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))


# ## Set-up Doc2Vec Training & Evaluation Models

# Approximating experiment of Le & Mikolov ["Distributed Representations of Sentences and Documents"](http://cs.stanford.edu/~quocle/paragraph_vector.pdf), also with guidance from Mikolov's [example go.sh](https://groups.google.com/d/msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ):
#
# `./word2vec -train ../alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1`
#
# Parameter choices below vary:
#
# * 100-dimensional vectors, as the 400d vectors of the paper don't seem to offer much benefit on this task
# * similarly, frequent word subsampling seems to decrease sentiment-prediction accuracy, so it's left out
# * `cbow=0` means skip-gram which is equivalent to the paper's 'PV-DBOW' mode, matched in gensim with `dm=0`
# * added to that DBOW model are two DM models, one which averages context vectors (`dm_mean`) and one which concatenates them (`dm_concat`, resulting in a much larger, slower, more data-hungry model)
# * a `min_count=2` saves quite a bit of model memory, discarding only words that appear in a single doc (and are thus no more expressive than the unique-to-each doc vectors themselves)

# In[2]:

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)


# Following the paper, we also evaluate models in pairs. These wrappers return the concatenation of the vectors from each model. (Only the singular models are trained.)

# In[5]:

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


# ## Predictive Evaluation Methods

# Helper methods for evaluating error rate.

# In[8]:

import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    #print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


# ## Bulk Training

# Using explicit multiple-pass, alpha-reduction approach as sketched in [gensim doc2vec blog post](http://radimrehurek.com/2014/12/doc2vec-tutorial/) – with added shuffling of corpus on each pass.
#
# Note that vector training is occurring on *all* documents of the dataset, which includes all TRAIN/TEST/DEV docs.
#
# Evaluation of each model's sentiment-predictive power is repeated after each pass, as an error rate (lower is better), to see the rates-of-relative-improvement. The base numbers reuse the TRAIN and TEST vectors stored in the models for the logistic regression, while the _inferred_ results use newly-inferred TEST vectors.
#
# (On a 4-core 2.6Ghz Intel Core i7, these 20 passes training and evaluating 3 main models takes about an hour.)

# In[9]:

from collections import defaultdict
best_error = defaultdict(lambda :1.0)  # to selectively-print only best errors achieved


# In[10]:

from random import shuffle
import datetime

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results

    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(doc_list)
            duration = '%.1f' % elapsed()

        # evaluate
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs)
        eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*'
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

        if ((epoch + 1) % 5) == 0 or epoch == 0:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if infer_err < best_error[name + '_inferred']:
                best_error[name + '_inferred'] = infer_err
                best_indicator = '*'
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))


# ## Achieved Sentiment-Prediction Accuracy

# In[12]:

# print best error rates achieved
for rate, name in sorted((rate, name) for name, rate in best_error.items()):
    print("%f %s" % (rate, name))


# In my testing, unlike the paper's report, DBOW performs best. Concatenating vectors from different models only offers a small predictive improvement. The best results I've seen are still just under 10% error rate, still a ways from the paper's 7.42%.
#

# ## Examining Results

# ### Are inferred vectors close to the precalculated ones?

# In[13]:

doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc; re-run cell for more examples
print('for doc %d...' % doc_id)
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))


# (Yes, here the stored vector from 20 epochs of training is usually one of the closest to a freshly-inferred vector for the same words. Note the defaults for inference are very abbreviated – just 3 steps starting at a high alpha – and likely need tuning for other applications.)

# ### Do close documents seem more related than distant ones?

# In[14]:

import random

doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
model = random.choice(simple_models)  # and a random model
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))


# (Somewhat, in terms of reviewer tone, movie genre, etc... the MOST cosine-similar docs usually seem more like the TARGET than the MEDIAN or LEAST.)

# ### Do the word vectors show useful similarities?

# In[15]:

word_models = simple_models[:]


# In[17]:

import random
from IPython.display import HTML
# pick a random word with a suitable number of occurences
while True:
    word = random.choice(word_models[0].index2word)
    if word_models[0].vocab[word].count > 10:
        break
# or uncomment below line, to just pick a word from the relevant domain:
#word = 'comedy/drama'
similars_per_model = [str(model.most_similar(word, topn=20)).replace('), ','),<br>\n') for model in word_models]
similar_table = ("<table><tr><th>" +
    "</th><th>".join([str(model) for model in word_models]) +
    "</th></tr><tr><td>" +
    "</td><td>".join(similars_per_model) +
    "</td></tr></table>")
print("most similar words for '%s' (%d occurences)" % (word, simple_models[0].vocab[word].count))
HTML(similar_table)


# Do the DBOW words look meaningless? That's because the gensim DBOW model doesn't train word vectors – they remain at their random initialized values – unless you ask with the `dbow_words=1` initialization parameter. Concurrent word-training slows DBOW mode significantly, and offers little improvement (and sometimes a little worsening) of the error rate on this IMDB sentiment-prediction task.
#
# Words from DM models tend to show meaningfully similar words when there are many examples in the training data (as with 'plot' or 'actor'). (All DM modes inherently involve word vector training concurrent with doc vector training.)

# ### Are the word vectors from this dataset any good at analogies?

# In[26]:

# assuming something like
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
# is in local directory
# note: this takes many minutes
for model in word_models:
    sections = model.accuracy('questions-words.txt')
    correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
    print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))


# Even though this is a tiny, domain-specific dataset, it shows some meager capability on the general word analogies – at least for the DM/concat and DM/mean models which actually train word vectors. (The untrained random-initialized words of the DBOW model of course fail miserably.)

# ## Slop

# In[ ]:

# This cell left intentionally erroneous.


# To mix the Google dataset (if locally available) into the word tests...

# In[ ]:

from gensim.models import Word2Vec
w2v_g100b = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_g100b.compact_name = 'w2v_g100b'
word_models.append(w2v_g100b)


# To get copious logging output from above steps...

# In[ ]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


# To auto-reload python code while developing...

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
