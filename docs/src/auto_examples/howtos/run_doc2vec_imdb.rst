.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_howtos_run_doc2vec_imdb.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_howtos_run_doc2vec_imdb.py:


How to reproduce the doc2vec 'Paragraph Vector' paper
=====================================================

Shows how to reproduce results of the "Distributed Representation of Sentences and Documents" paper by Le and Mikolov using Gensim.



.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








Introduction
------------

This guide shows you how to reproduce the results of the paper by `Le and
Mikolov 2014 <https://arxiv.org/pdf/1405.4053.pdf>`_ using Gensim. While the
entire paper is worth reading (it's only 9 pages), we will be focusing on
Section 3.2: "Beyond One Sentence - Sentiment Analysis with the IMDB
dataset".

This guide follows the following steps:

#. Load the IMDB dataset
#. Train a variety of Doc2Vec models on the dataset
#. Evaluate the performance of each model using a logistic regression
#. Examine some of the results directly:

When examining results, we will look for answers for the following questions:

#. Are inferred vectors close to the precalculated ones?
#. Do close documents seem more related than distant ones?
#. Do the word vectors show useful similarities?
#. Are the word vectors from this dataset any good at analogies?

Load corpus
-----------

Our data for the tutorial will be the `IMDB archive
<http://ai.stanford.edu/~amaas/data/sentiment/>`_.
If you're not familiar with this dataset, then here's a brief intro: it
contains several thousand movie reviews.

Each review is a single line of text containing multiple sentences, for example:

```
One of the best movie-dramas I have ever seen. We do a lot of acting in the
church and this is one that can be used as a resource that highlights all the
good things that actors can do in their work. I highly recommend this one,
especially for those who have an interest in acting, as a "must see."
```

These reviews will be the **documents** that we will work with in this tutorial.
There are 100 thousand reviews in total.

#. 25k reviews for training (12.5k positive, 12.5k negative)
#. 25k reviews for testing (12.5k positive, 12.5k negative)
#. 50k unlabeled reviews

Out of 100k reviews, 50k have a label: either positive (the reviewer liked
the movie) or negative.
The remaining 50k are unlabeled.

Our first task will be to prepare the dataset.

More specifically, we will:

#. Download the tar.gz file (it's only 84MB, so this shouldn't take too long)
#. Unpack it and extract each movie review
#. Split the reviews into training and test datasets

First, let's define a convenient datatype for holding data for a single document:

* words: The text of the document, as a ``list`` of words.
* tags: Used to keep the index of the document in the entire dataset.
* split: one of ``train``\ , ``test`` or ``extra``. Determines how the document will be used (for training, testing, etc).
* sentiment: either 1 (positive), 0 (negative) or None (unlabeled document).

This data type is helpful for later evaluation and reporting.
In particular, the ``index`` member will help us quickly and easily retrieve the vectors for a document from a model.



.. code-block:: default

    import collections

    SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags split sentiment')








We can now proceed with loading the corpus.


.. code-block:: default

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








Here's what a single document looks like.


.. code-block:: default

    print(alldocs[27])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SentimentDocument(words=['I', 'was', 'looking', 'forward', 'to', 'this', 'movie.', 'Trustworthy', 'actors,', 'interesting', 'plot.', 'Great', 'atmosphere', 'then', '?????', 'IF', 'you', 'are', 'going', 'to', 'attempt', 'something', 'that', 'is', 'meant', 'to', 'encapsulate', 'the', 'meaning', 'of', 'life.', 'First.', 'Know', 'it.', 'OK', 'I', 'did', 'not', 'expect', 'the', 'directors', 'or', 'writers', 'to', 'actually', 'know', 'the', 'meaning', 'but', 'I', 'thought', 'they', 'may', 'have', 'offered', 'crumbs', 'to', 'peck', 'at', 'and', 'treats', 'to', 'add', 'fuel', 'to', 'the', 'fire-Which!', 'they', 'almost', 'did.', 'Things', 'I', "didn't", 'get.', 'A', 'woman', 'wandering', 'around', 'in', 'dark', 'places', 'and', 'lonely', 'car', 'parks', 'alone-oblivious', 'to', 'the', 'consequences.', 'Great', 'riddles', 'that', 'fell', 'by', 'the', 'wayside.', 'The', 'promise', 'of', 'the', 'knowledge', 'therein', 'contained', 'by', 'the', 'original', 'so-called', 'criminal.', 'I', 'had', 'no', 'problem', 'with', 'the', 'budget', 'and', 'enjoyed', 'the', 'suspense.', 'I', 'understood', 'and', 'can', 'wax', 'lyrical', 'about', 'the', 'fool', 'and', 'found', 'Adrian', 'Pauls', 'role', 'crucial', 'and', 'penetrating', 'and', 'then', '?????', 'Basically', 'the', 'story', 'line', 'and', 'the', 'script', 'where', 'good', 'up', 'to', 'a', 'point', 'and', 'that', 'point', 'was', 'the', 'last', '10', 'minutes', 'or', 'so.', 'What?', 'Run', 'out', 'of', 'ideas!', 'Such', 'a', 'pity', 'that', 'this', 'movie', 'had', 'to', 'let', 'us', 'down', 'so', 'badly.', 'It', 'may', 'not', 'comprehend', 'the', 'meaning', 'and', 'I', 'really', 'did', 'not', 'expect', 'the', 'writers', 'to', 'understand', 'it', 'but', 'I', 'was', 'hoping', 'for', 'an', 'intellectual,', 'if', 'not', 'spiritual', 'ride', 'and', 'got', 'a', 'bump', 'in', 'the', 'road'], tags=[27], split='test', sentiment=0.0)




Extract our documents and split into training/test sets.


.. code-block:: default

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print(f'{len(alldocs)} docs: {len(train_docs)} train-sentiment, {len(test_docs)} test-sentiment')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    100000 docs: 25000 train-sentiment, 25000 test-sentiment




Set-up Doc2Vec Training & Evaluation Models
-------------------------------------------

We approximate the experiment of Le & Mikolov `"Distributed Representations
of Sentences and Documents"
<http://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_ with guidance from
Mikolov's `example go.sh
<https://groups.google.com/d/msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ>`_::

    ./word2vec -train ../alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1

We vary the following parameter choices:

* 100-dimensional vectors, as the 400-d vectors of the paper take a lot of
  memory and, in our tests of this task, don't seem to offer much benefit
* Similarly, frequent word subsampling seems to decrease sentiment-prediction
  accuracy, so it's left out
* ``cbow=0`` means skip-gram which is equivalent to the paper's 'PV-DBOW'
  mode, matched in gensim with ``dm=0``
* Added to that DBOW model are two DM models, one which averages context
  vectors (\ ``dm_mean``\ ) and one which concatenates them (\ ``dm_concat``\ ,
  resulting in a much larger, slower, more data-hungry model)
* A ``min_count=2`` saves quite a bit of model memory, discarding only words
  that appear in a single doc (and are thus no more expressive than the
  unique-to-each doc vectors themselves)



.. code-block:: default


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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    2020-09-30 22:37:03,794 : INFO : using concatenative 1100-dimensional layer1
    2020-09-30 22:37:03,797 : INFO : collecting all words and their counts
    2020-09-30 22:37:03,797 : INFO : PROGRESS: at example #0, processed 0 words (0/s), 0 word types, 0 tags
    2020-09-30 22:37:04,523 : INFO : PROGRESS: at example #10000, processed 2292381 words (3159292/s), 150816 word types, 0 tags
    2020-09-30 22:37:05,236 : INFO : PROGRESS: at example #20000, processed 4573645 words (3201510/s), 238497 word types, 0 tags
    2020-09-30 22:37:05,974 : INFO : PROGRESS: at example #30000, processed 6865575 words (3106937/s), 312348 word types, 0 tags
    2020-09-30 22:37:06,731 : INFO : PROGRESS: at example #40000, processed 9190019 words (3071781/s), 377231 word types, 0 tags
    2020-09-30 22:37:07,465 : INFO : PROGRESS: at example #50000, processed 11557847 words (3227463/s), 438729 word types, 0 tags
    2020-09-30 22:37:08,233 : INFO : PROGRESS: at example #60000, processed 13899883 words (3046947/s), 493913 word types, 0 tags
    2020-09-30 22:37:09,009 : INFO : PROGRESS: at example #70000, processed 16270094 words (3056103/s), 548474 word types, 0 tags
    2020-09-30 22:37:09,777 : INFO : PROGRESS: at example #80000, processed 18598876 words (3035248/s), 598272 word types, 0 tags
    2020-09-30 22:37:10,563 : INFO : PROGRESS: at example #90000, processed 20916044 words (2945690/s), 646082 word types, 0 tags
    2020-09-30 22:37:11,354 : INFO : collected 693922 word types and 100000 unique tags from a corpus of 100000 examples and 23279529 words
    2020-09-30 22:37:11,354 : INFO : Loading a fresh vocabulary
    2020-09-30 22:37:13,167 : INFO : effective_min_count=2 retains 265408 unique words (38% of original 693922, drops 428514)
    2020-09-30 22:37:13,167 : INFO : effective_min_count=2 leaves 22851015 word corpus (98% of original 23279529, drops 428514)
    2020-09-30 22:37:15,301 : INFO : deleting the raw counts dictionary of 693922 items
    2020-09-30 22:37:15,317 : INFO : sample=0 downsamples 0 most-common words
    2020-09-30 22:37:15,317 : INFO : downsampling leaves estimated 22851015 word corpus (100.0% of prior 22851015)
    2020-09-30 22:37:18,663 : INFO : estimated required memory for 265408 words and 100 dimensions: 405030400 bytes
    2020-09-30 22:37:18,663 : INFO : resetting layer weights
    Doc2Vec(dbow,d100,n5,mc2,t8) vocabulary scanned & state initialized
    2020-09-30 22:37:37,706 : INFO : collecting all words and their counts
    2020-09-30 22:37:37,706 : INFO : PROGRESS: at example #0, processed 0 words (0/s), 0 word types, 0 tags
    2020-09-30 22:37:38,202 : INFO : PROGRESS: at example #10000, processed 2292381 words (4629335/s), 150816 word types, 0 tags
    2020-09-30 22:37:38,761 : INFO : PROGRESS: at example #20000, processed 4573645 words (4082741/s), 238497 word types, 0 tags
    2020-09-30 22:37:39,283 : INFO : PROGRESS: at example #30000, processed 6865575 words (4388176/s), 312348 word types, 0 tags
    2020-09-30 22:37:39,843 : INFO : PROGRESS: at example #40000, processed 9190019 words (4151942/s), 377231 word types, 0 tags
    2020-09-30 22:37:40,396 : INFO : PROGRESS: at example #50000, processed 11557847 words (4287824/s), 438729 word types, 0 tags
    2020-09-30 22:37:40,939 : INFO : PROGRESS: at example #60000, processed 13899883 words (4311204/s), 493913 word types, 0 tags
    2020-09-30 22:37:41,498 : INFO : PROGRESS: at example #70000, processed 16270094 words (4242545/s), 548474 word types, 0 tags
    2020-09-30 22:37:42,032 : INFO : PROGRESS: at example #80000, processed 18598876 words (4360341/s), 598272 word types, 0 tags
    2020-09-30 22:37:42,582 : INFO : PROGRESS: at example #90000, processed 20916044 words (4217682/s), 646082 word types, 0 tags
    2020-09-30 22:37:43,152 : INFO : collected 693922 word types and 100000 unique tags from a corpus of 100000 examples and 23279529 words
    2020-09-30 22:37:43,152 : INFO : Loading a fresh vocabulary
    2020-09-30 22:37:44,948 : INFO : effective_min_count=2 retains 265408 unique words (38% of original 693922, drops 428514)
    2020-09-30 22:37:44,948 : INFO : effective_min_count=2 leaves 22851015 word corpus (98% of original 23279529, drops 428514)
    2020-09-30 22:37:46,994 : INFO : deleting the raw counts dictionary of 693922 items
    2020-09-30 22:37:47,008 : INFO : sample=0 downsamples 0 most-common words
    2020-09-30 22:37:47,009 : INFO : downsampling leaves estimated 22851015 word corpus (100.0% of prior 22851015)
    2020-09-30 22:37:50,336 : INFO : estimated required memory for 265408 words and 100 dimensions: 405030400 bytes
    2020-09-30 22:37:50,336 : INFO : resetting layer weights
    Doc2Vec(dm/m,d100,n5,w10,mc2,t8) vocabulary scanned & state initialized
    2020-09-30 22:38:09,628 : INFO : collecting all words and their counts
    2020-09-30 22:38:09,628 : INFO : PROGRESS: at example #0, processed 0 words (0/s), 0 word types, 0 tags
    2020-09-30 22:38:10,145 : INFO : PROGRESS: at example #10000, processed 2292381 words (4432122/s), 150816 word types, 0 tags
    2020-09-30 22:38:10,697 : INFO : PROGRESS: at example #20000, processed 4573645 words (4131700/s), 238497 word types, 0 tags
    2020-09-30 22:38:11,250 : INFO : PROGRESS: at example #30000, processed 6865575 words (4147845/s), 312348 word types, 0 tags
    2020-09-30 22:38:11,826 : INFO : PROGRESS: at example #40000, processed 9190019 words (4040528/s), 377231 word types, 0 tags
    2020-09-30 22:38:12,399 : INFO : PROGRESS: at example #50000, processed 11557847 words (4129801/s), 438729 word types, 0 tags
    2020-09-30 22:38:12,975 : INFO : PROGRESS: at example #60000, processed 13899883 words (4069527/s), 493913 word types, 0 tags
    2020-09-30 22:38:13,556 : INFO : PROGRESS: at example #70000, processed 16270094 words (4079870/s), 548474 word types, 0 tags
    2020-09-30 22:38:14,145 : INFO : PROGRESS: at example #80000, processed 18598876 words (3956736/s), 598272 word types, 0 tags
    2020-09-30 22:38:14,726 : INFO : PROGRESS: at example #90000, processed 20916044 words (3986449/s), 646082 word types, 0 tags
    2020-09-30 22:38:15,292 : INFO : collected 693922 word types and 100000 unique tags from a corpus of 100000 examples and 23279529 words
    2020-09-30 22:38:15,293 : INFO : Loading a fresh vocabulary
    2020-09-30 22:38:17,117 : INFO : effective_min_count=2 retains 265408 unique words (38% of original 693922, drops 428514)
    2020-09-30 22:38:17,117 : INFO : effective_min_count=2 leaves 22851015 word corpus (98% of original 23279529, drops 428514)
    2020-09-30 22:38:19,256 : INFO : deleting the raw counts dictionary of 693922 items
    2020-09-30 22:38:19,270 : INFO : sample=0 downsamples 0 most-common words
    2020-09-30 22:38:19,270 : INFO : downsampling leaves estimated 22851015 word corpus (100.0% of prior 22851015)
    2020-09-30 22:38:22,593 : INFO : estimated required memory for 265408 words and 100 dimensions: 1466662400 bytes
    2020-09-30 22:38:22,593 : INFO : resetting layer weights
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8) vocabulary scanned & state initialized




Le and Mikolov note that combining a paragraph vector from Distributed Bag of
Words (DBOW) and Distributed Memory (DM) improves performance. We will
follow, pairing the models together for evaluation. Here, we concatenate the
paragraph vectors obtained from each model with the help of a thin wrapper
class included in a gensim test module. (Note that this a separate, later
concatenation of output-vectors than the kind of input-window-concatenation
enabled by the ``dm_concat=1`` mode above.)



.. code-block:: default

    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    2020-09-30 22:38:42,021 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2020-09-30 22:38:42,022 : INFO : built Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...) from 9 documents (total 29 corpus positions)




Predictive Evaluation Methods
-----------------------------

Given a document, our ``Doc2Vec`` models output a vector representation of the document.
How useful is a particular model?
In case of sentiment analysis, we want the ouput vector to reflect the sentiment in the input document.
So, in vector space, positive documents should be distant from negative documents.

We train a logistic regression from the training set:

  - regressors (inputs): document vectors from the Doc2Vec model
  - target (outpus): sentiment labels

So, this logistic regression will be able to predict sentiment given a document vector.

Next, we test our logistic regression on the test set, and measure the rate of errors (incorrect predictions).
If the document vectors from the Doc2Vec model reflect the actual sentiment well, the error rate will be low.

Therefore, the error rate of the logistic regression is indication of *how well* the given Doc2Vec model represents documents as vectors.
We can then compare different ``Doc2Vec`` models by looking at their error rates.



.. code-block:: default


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








Bulk Training & Per-Model Evaluation
------------------------------------

Note that doc-vector training is occurring on *all* documents of the dataset,
which includes all TRAIN/TEST/DEV docs.  Because the native document-order
has similar-sentiment documents in large clumps – which is suboptimal for
training – we work with once-shuffled copy of the training set.

We evaluate each model's sentiment predictive power based on error rate, and
the evaluation is done for each model.

(On a 4-core 2.6Ghz Intel Core i7, these 20 passes training and evaluating 3
main models takes about an hour.)



.. code-block:: default

    from collections import defaultdict
    error_rates = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved









.. code-block:: default

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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Training Doc2Vec(dbow,d100,n5,mc2,t8)
    2020-09-30 22:38:43,643 : INFO : training model with 8 workers on 265408 vocabulary and 100 features, using sg=1 hs=0 sample=0 negative=5 window=5
    2020-09-30 22:38:44,654 : INFO : EPOCH 1 - PROGRESS: at 3.50% examples, 810634 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:38:45,665 : INFO : EPOCH 1 - PROGRESS: at 8.11% examples, 932060 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:46,674 : INFO : EPOCH 1 - PROGRESS: at 13.11% examples, 1003981 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:47,682 : INFO : EPOCH 1 - PROGRESS: at 18.36% examples, 1056038 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:48,690 : INFO : EPOCH 1 - PROGRESS: at 23.57% examples, 1080969 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:49,705 : INFO : EPOCH 1 - PROGRESS: at 28.88% examples, 1102341 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:38:50,708 : INFO : EPOCH 1 - PROGRESS: at 34.36% examples, 1120959 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:51,715 : INFO : EPOCH 1 - PROGRESS: at 39.89% examples, 1138346 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:52,729 : INFO : EPOCH 1 - PROGRESS: at 45.30% examples, 1148552 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:53,729 : INFO : EPOCH 1 - PROGRESS: at 50.84% examples, 1158541 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:38:54,738 : INFO : EPOCH 1 - PROGRESS: at 56.30% examples, 1166666 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:55,740 : INFO : EPOCH 1 - PROGRESS: at 61.75% examples, 1174829 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:56,740 : INFO : EPOCH 1 - PROGRESS: at 67.20% examples, 1179766 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:57,740 : INFO : EPOCH 1 - PROGRESS: at 72.73% examples, 1186870 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:38:58,756 : INFO : EPOCH 1 - PROGRESS: at 78.14% examples, 1189792 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:38:59,760 : INFO : EPOCH 1 - PROGRESS: at 83.70% examples, 1194989 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:00,763 : INFO : EPOCH 1 - PROGRESS: at 89.28% examples, 1199066 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:01,764 : INFO : EPOCH 1 - PROGRESS: at 94.91% examples, 1202983 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:02,608 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:39:02,614 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:39:02,615 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:39:02,616 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:39:02,617 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:39:02,622 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:39:02,629 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:39:02,631 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:39:02,631 : INFO : EPOCH - 1 : training on 23279529 raw words (22951015 effective words) took 19.0s, 1209291 effective words/s
    2020-09-30 22:39:03,641 : INFO : EPOCH 2 - PROGRESS: at 5.81% examples, 1334892 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:04,642 : INFO : EPOCH 2 - PROGRESS: at 11.52% examples, 1332668 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:05,650 : INFO : EPOCH 2 - PROGRESS: at 17.41% examples, 1339624 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:06,654 : INFO : EPOCH 2 - PROGRESS: at 23.22% examples, 1335308 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:07,656 : INFO : EPOCH 2 - PROGRESS: at 28.97% examples, 1332332 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:08,673 : INFO : EPOCH 2 - PROGRESS: at 34.48% examples, 1314497 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:09,676 : INFO : EPOCH 2 - PROGRESS: at 40.24% examples, 1314296 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:10,698 : INFO : EPOCH 2 - PROGRESS: at 45.66% examples, 1302303 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:11,700 : INFO : EPOCH 2 - PROGRESS: at 51.23% examples, 1297227 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:12,702 : INFO : EPOCH 2 - PROGRESS: at 56.84% examples, 1297181 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:13,702 : INFO : EPOCH 2 - PROGRESS: at 62.60% examples, 1300534 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:14,707 : INFO : EPOCH 2 - PROGRESS: at 68.34% examples, 1300550 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:39:15,712 : INFO : EPOCH 2 - PROGRESS: at 73.81% examples, 1298506 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:16,720 : INFO : EPOCH 2 - PROGRESS: at 79.37% examples, 1295658 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:17,724 : INFO : EPOCH 2 - PROGRESS: at 85.01% examples, 1294762 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:18,744 : INFO : EPOCH 2 - PROGRESS: at 90.64% examples, 1292767 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:19,760 : INFO : EPOCH 2 - PROGRESS: at 96.32% examples, 1290914 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:20,357 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:39:20,358 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:39:20,361 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:39:20,365 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:39:20,373 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:39:20,374 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:39:20,379 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:39:20,381 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:39:20,381 : INFO : EPOCH - 2 : training on 23279529 raw words (22951015 effective words) took 17.7s, 1293152 effective words/s
    2020-09-30 22:39:21,385 : INFO : EPOCH 3 - PROGRESS: at 5.45% examples, 1266660 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:22,387 : INFO : EPOCH 3 - PROGRESS: at 11.09% examples, 1288270 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:39:23,387 : INFO : EPOCH 3 - PROGRESS: at 16.66% examples, 1287842 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:24,389 : INFO : EPOCH 3 - PROGRESS: at 22.22% examples, 1285359 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:25,397 : INFO : EPOCH 3 - PROGRESS: at 27.87% examples, 1283208 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:26,405 : INFO : EPOCH 3 - PROGRESS: at 33.33% examples, 1275169 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:27,408 : INFO : EPOCH 3 - PROGRESS: at 39.04% examples, 1279197 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:28,414 : INFO : EPOCH 3 - PROGRESS: at 44.64% examples, 1279209 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:29,416 : INFO : EPOCH 3 - PROGRESS: at 50.46% examples, 1283072 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:30,417 : INFO : EPOCH 3 - PROGRESS: at 56.09% examples, 1284428 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:31,418 : INFO : EPOCH 3 - PROGRESS: at 61.67% examples, 1285268 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:32,435 : INFO : EPOCH 3 - PROGRESS: at 66.94% examples, 1276443 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:33,444 : INFO : EPOCH 3 - PROGRESS: at 72.56% examples, 1277342 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:34,447 : INFO : EPOCH 3 - PROGRESS: at 77.98% examples, 1275080 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:35,452 : INFO : EPOCH 3 - PROGRESS: at 83.31% examples, 1271646 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:36,467 : INFO : EPOCH 3 - PROGRESS: at 89.05% examples, 1272677 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:37,469 : INFO : EPOCH 3 - PROGRESS: at 94.11% examples, 1265033 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:38,473 : INFO : EPOCH 3 - PROGRESS: at 99.28% examples, 1259790 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:38,547 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:39:38,551 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:39:38,553 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:39:38,555 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:39:38,559 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:39:38,560 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:39:38,567 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:39:38,570 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:39:38,570 : INFO : EPOCH - 3 : training on 23279529 raw words (22951015 effective words) took 18.2s, 1262025 effective words/s
    2020-09-30 22:39:39,578 : INFO : EPOCH 4 - PROGRESS: at 5.45% examples, 1260893 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:40,581 : INFO : EPOCH 4 - PROGRESS: at 11.13% examples, 1289836 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:41,582 : INFO : EPOCH 4 - PROGRESS: at 16.91% examples, 1304703 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:42,582 : INFO : EPOCH 4 - PROGRESS: at 22.61% examples, 1305466 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:43,585 : INFO : EPOCH 4 - PROGRESS: at 28.35% examples, 1306310 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:44,588 : INFO : EPOCH 4 - PROGRESS: at 33.85% examples, 1295655 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:45,609 : INFO : EPOCH 4 - PROGRESS: at 39.63% examples, 1296177 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:46,614 : INFO : EPOCH 4 - PROGRESS: at 44.97% examples, 1286945 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:47,617 : INFO : EPOCH 4 - PROGRESS: at 50.58% examples, 1284547 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:48,631 : INFO : EPOCH 4 - PROGRESS: at 55.88% examples, 1276301 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:49,634 : INFO : EPOCH 4 - PROGRESS: at 61.46% examples, 1277819 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:39:50,635 : INFO : EPOCH 4 - PROGRESS: at 67.16% examples, 1279326 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:51,638 : INFO : EPOCH 4 - PROGRESS: at 72.88% examples, 1282749 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:39:52,653 : INFO : EPOCH 4 - PROGRESS: at 78.43% examples, 1281163 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:53,665 : INFO : EPOCH 4 - PROGRESS: at 84.31% examples, 1284425 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:54,669 : INFO : EPOCH 4 - PROGRESS: at 90.11% examples, 1286674 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:55,675 : INFO : EPOCH 4 - PROGRESS: at 96.00% examples, 1288231 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:56,310 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:39:56,315 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:39:56,318 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:39:56,319 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:39:56,327 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:39:56,329 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:39:56,332 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:39:56,335 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:39:56,335 : INFO : EPOCH - 4 : training on 23279529 raw words (22951015 effective words) took 17.8s, 1292154 effective words/s
    2020-09-30 22:39:57,340 : INFO : EPOCH 5 - PROGRESS: at 5.58% examples, 1294342 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:39:58,348 : INFO : EPOCH 5 - PROGRESS: at 10.96% examples, 1268816 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:39:59,351 : INFO : EPOCH 5 - PROGRESS: at 16.21% examples, 1244571 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:00,364 : INFO : EPOCH 5 - PROGRESS: at 21.88% examples, 1259315 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:01,369 : INFO : EPOCH 5 - PROGRESS: at 27.41% examples, 1257507 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:02,371 : INFO : EPOCH 5 - PROGRESS: at 32.96% examples, 1259700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:03,385 : INFO : EPOCH 5 - PROGRESS: at 38.66% examples, 1262599 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:04,394 : INFO : EPOCH 5 - PROGRESS: at 44.34% examples, 1265326 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:05,401 : INFO : EPOCH 5 - PROGRESS: at 49.65% examples, 1259209 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:06,418 : INFO : EPOCH 5 - PROGRESS: at 55.33% examples, 1260120 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:07,431 : INFO : EPOCH 5 - PROGRESS: at 60.91% examples, 1263647 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:08,434 : INFO : EPOCH 5 - PROGRESS: at 66.57% examples, 1264442 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:09,439 : INFO : EPOCH 5 - PROGRESS: at 71.86% examples, 1260773 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:10,450 : INFO : EPOCH 5 - PROGRESS: at 77.20% examples, 1258923 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:11,450 : INFO : EPOCH 5 - PROGRESS: at 82.67% examples, 1257666 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:40:12,455 : INFO : EPOCH 5 - PROGRESS: at 88.22% examples, 1257992 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:13,456 : INFO : EPOCH 5 - PROGRESS: at 93.77% examples, 1258003 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:14,468 : INFO : EPOCH 5 - PROGRESS: at 99.32% examples, 1257390 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:14,547 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:40:14,554 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:40:14,557 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:40:14,558 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:40:14,559 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:40:14,562 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:40:14,564 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:40:14,565 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:40:14,565 : INFO : EPOCH - 5 : training on 23279529 raw words (22951015 effective words) took 18.2s, 1259124 effective words/s
    2020-09-30 22:40:15,572 : INFO : EPOCH 6 - PROGRESS: at 4.99% examples, 1157587 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:16,586 : INFO : EPOCH 6 - PROGRESS: at 10.29% examples, 1187334 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:17,587 : INFO : EPOCH 6 - PROGRESS: at 15.62% examples, 1197426 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:18,590 : INFO : EPOCH 6 - PROGRESS: at 20.71% examples, 1195503 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:19,604 : INFO : EPOCH 6 - PROGRESS: at 26.33% examples, 1208202 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:20,608 : INFO : EPOCH 6 - PROGRESS: at 31.77% examples, 1213675 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:21,612 : INFO : EPOCH 6 - PROGRESS: at 37.07% examples, 1210678 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:22,618 : INFO : EPOCH 6 - PROGRESS: at 42.66% examples, 1219476 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:23,623 : INFO : EPOCH 6 - PROGRESS: at 48.39% examples, 1228249 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:24,647 : INFO : EPOCH 6 - PROGRESS: at 54.05% examples, 1231281 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:25,651 : INFO : EPOCH 6 - PROGRESS: at 59.74% examples, 1240284 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:26,662 : INFO : EPOCH 6 - PROGRESS: at 65.22% examples, 1239158 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:27,675 : INFO : EPOCH 6 - PROGRESS: at 70.73% examples, 1240296 words/s, in_qsize 15, out_qsize 2
    2020-09-30 22:40:28,688 : INFO : EPOCH 6 - PROGRESS: at 76.10% examples, 1240448 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:29,693 : INFO : EPOCH 6 - PROGRESS: at 81.53% examples, 1239411 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:30,696 : INFO : EPOCH 6 - PROGRESS: at 86.95% examples, 1239166 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:31,697 : INFO : EPOCH 6 - PROGRESS: at 92.42% examples, 1239150 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:32,711 : INFO : EPOCH 6 - PROGRESS: at 98.00% examples, 1240013 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:33,022 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:40:33,022 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:40:33,024 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:40:33,026 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:40:33,030 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:40:33,031 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:40:33,034 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:40:33,040 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:40:33,040 : INFO : EPOCH - 6 : training on 23279529 raw words (22951015 effective words) took 18.5s, 1242469 effective words/s
    2020-09-30 22:40:34,056 : INFO : EPOCH 7 - PROGRESS: at 5.58% examples, 1280763 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:35,057 : INFO : EPOCH 7 - PROGRESS: at 10.85% examples, 1257239 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:36,060 : INFO : EPOCH 7 - PROGRESS: at 16.51% examples, 1268944 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:37,070 : INFO : EPOCH 7 - PROGRESS: at 22.10% examples, 1270911 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:38,081 : INFO : EPOCH 7 - PROGRESS: at 27.65% examples, 1267114 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:39,092 : INFO : EPOCH 7 - PROGRESS: at 33.05% examples, 1259463 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:40,093 : INFO : EPOCH 7 - PROGRESS: at 38.45% examples, 1255238 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:41,097 : INFO : EPOCH 7 - PROGRESS: at 44.21% examples, 1262013 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:40:42,101 : INFO : EPOCH 7 - PROGRESS: at 49.84% examples, 1264196 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:43,114 : INFO : EPOCH 7 - PROGRESS: at 55.52% examples, 1266020 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:44,122 : INFO : EPOCH 7 - PROGRESS: at 60.90% examples, 1265325 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:45,127 : INFO : EPOCH 7 - PROGRESS: at 65.80% examples, 1251392 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:40:46,144 : INFO : EPOCH 7 - PROGRESS: at 70.73% examples, 1240850 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:47,148 : INFO : EPOCH 7 - PROGRESS: at 75.98% examples, 1239689 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:48,149 : INFO : EPOCH 7 - PROGRESS: at 81.30% examples, 1237700 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:40:49,155 : INFO : EPOCH 7 - PROGRESS: at 86.92% examples, 1239847 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:50,155 : INFO : EPOCH 7 - PROGRESS: at 92.27% examples, 1238659 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:51,159 : INFO : EPOCH 7 - PROGRESS: at 97.87% examples, 1240246 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:51,492 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:40:51,493 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:40:51,497 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:40:51,498 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:40:51,504 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:40:51,506 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:40:51,511 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:40:51,513 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:40:51,513 : INFO : EPOCH - 7 : training on 23279529 raw words (22951015 effective words) took 18.5s, 1242577 effective words/s
    2020-09-30 22:40:52,518 : INFO : EPOCH 8 - PROGRESS: at 5.31% examples, 1237109 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:53,523 : INFO : EPOCH 8 - PROGRESS: at 10.77% examples, 1251883 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:54,533 : INFO : EPOCH 8 - PROGRESS: at 16.24% examples, 1246617 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:55,544 : INFO : EPOCH 8 - PROGRESS: at 21.06% examples, 1212978 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:56,550 : INFO : EPOCH 8 - PROGRESS: at 25.68% examples, 1177978 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:57,551 : INFO : EPOCH 8 - PROGRESS: at 30.77% examples, 1177925 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:40:58,554 : INFO : EPOCH 8 - PROGRESS: at 36.09% examples, 1180227 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:40:59,557 : INFO : EPOCH 8 - PROGRESS: at 41.39% examples, 1184834 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:00,560 : INFO : EPOCH 8 - PROGRESS: at 46.70% examples, 1186988 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:01,570 : INFO : EPOCH 8 - PROGRESS: at 52.31% examples, 1194996 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:02,573 : INFO : EPOCH 8 - PROGRESS: at 57.66% examples, 1198839 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:03,584 : INFO : EPOCH 8 - PROGRESS: at 63.25% examples, 1204897 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:04,598 : INFO : EPOCH 8 - PROGRESS: at 68.51% examples, 1203348 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:41:05,626 : INFO : EPOCH 8 - PROGRESS: at 73.81% examples, 1203637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:06,635 : INFO : EPOCH 8 - PROGRESS: at 78.48% examples, 1193759 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:07,635 : INFO : EPOCH 8 - PROGRESS: at 83.40% examples, 1189957 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:08,639 : INFO : EPOCH 8 - PROGRESS: at 87.64% examples, 1176213 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:09,640 : INFO : EPOCH 8 - PROGRESS: at 92.10% examples, 1167420 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:10,641 : INFO : EPOCH 8 - PROGRESS: at 96.91% examples, 1163172 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:11,370 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:41:11,371 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:41:11,373 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:41:11,386 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:41:11,389 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:41:11,392 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:41:11,395 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:41:11,400 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:41:11,400 : INFO : EPOCH - 8 : training on 23279529 raw words (22951015 effective words) took 19.9s, 1154250 effective words/s
    2020-09-30 22:41:12,408 : INFO : EPOCH 9 - PROGRESS: at 4.06% examples, 934517 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:41:13,415 : INFO : EPOCH 9 - PROGRESS: at 7.93% examples, 913759 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:14,421 : INFO : EPOCH 9 - PROGRESS: at 11.52% examples, 887053 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:15,437 : INFO : EPOCH 9 - PROGRESS: at 15.84% examples, 908247 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:16,449 : INFO : EPOCH 9 - PROGRESS: at 19.68% examples, 904688 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:17,449 : INFO : EPOCH 9 - PROGRESS: at 24.45% examples, 933015 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:18,477 : INFO : EPOCH 9 - PROGRESS: at 29.50% examples, 963829 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:19,482 : INFO : EPOCH 9 - PROGRESS: at 34.48% examples, 982678 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:20,494 : INFO : EPOCH 9 - PROGRESS: at 39.30% examples, 994877 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:21,516 : INFO : EPOCH 9 - PROGRESS: at 43.82% examples, 996557 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:22,520 : INFO : EPOCH 9 - PROGRESS: at 47.98% examples, 992704 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:23,527 : INFO : EPOCH 9 - PROGRESS: at 52.95% examples, 1002986 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:24,532 : INFO : EPOCH 9 - PROGRESS: at 58.20% examples, 1019197 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:25,537 : INFO : EPOCH 9 - PROGRESS: at 62.72% examples, 1020615 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:26,541 : INFO : EPOCH 9 - PROGRESS: at 67.70% examples, 1027724 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:27,549 : INFO : EPOCH 9 - PROGRESS: at 72.18% examples, 1027853 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:28,557 : INFO : EPOCH 9 - PROGRESS: at 77.03% examples, 1033423 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:29,567 : INFO : EPOCH 9 - PROGRESS: at 82.33% examples, 1042148 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:30,572 : INFO : EPOCH 9 - PROGRESS: at 87.64% examples, 1050648 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:31,575 : INFO : EPOCH 9 - PROGRESS: at 92.92% examples, 1057974 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:32,578 : INFO : EPOCH 9 - PROGRESS: at 98.21% examples, 1064734 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:32,865 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:41:32,868 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:41:32,871 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:41:32,871 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:41:32,872 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:41:32,872 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:41:32,882 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:41:32,886 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:41:32,886 : INFO : EPOCH - 9 : training on 23279529 raw words (22951015 effective words) took 21.5s, 1068336 effective words/s
    2020-09-30 22:41:33,891 : INFO : EPOCH 10 - PROGRESS: at 4.99% examples, 1160494 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:34,902 : INFO : EPOCH 10 - PROGRESS: at 10.09% examples, 1166681 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:35,903 : INFO : EPOCH 10 - PROGRESS: at 15.45% examples, 1186580 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:36,913 : INFO : EPOCH 10 - PROGRESS: at 20.71% examples, 1194965 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:37,914 : INFO : EPOCH 10 - PROGRESS: at 26.33% examples, 1211006 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:41:38,918 : INFO : EPOCH 10 - PROGRESS: at 31.52% examples, 1206396 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:39,925 : INFO : EPOCH 10 - PROGRESS: at 37.07% examples, 1212076 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:40,930 : INFO : EPOCH 10 - PROGRESS: at 42.57% examples, 1218468 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:41,937 : INFO : EPOCH 10 - PROGRESS: at 48.25% examples, 1226103 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:42,938 : INFO : EPOCH 10 - PROGRESS: at 53.63% examples, 1225374 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:43,941 : INFO : EPOCH 10 - PROGRESS: at 59.06% examples, 1229025 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:44,943 : INFO : EPOCH 10 - PROGRESS: at 64.33% examples, 1226391 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:45,946 : INFO : EPOCH 10 - PROGRESS: at 69.81% examples, 1228690 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:41:46,967 : INFO : EPOCH 10 - PROGRESS: at 75.14% examples, 1228313 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:47,967 : INFO : EPOCH 10 - PROGRESS: at 80.50% examples, 1228427 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:48,978 : INFO : EPOCH 10 - PROGRESS: at 85.68% examples, 1224206 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:49,982 : INFO : EPOCH 10 - PROGRESS: at 90.99% examples, 1223108 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:50,995 : INFO : EPOCH 10 - PROGRESS: at 96.32% examples, 1221126 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:51,610 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:41:51,611 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:41:51,613 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:41:51,615 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:41:51,621 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:41:51,622 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:41:51,626 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:41:51,633 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:41:51,633 : INFO : EPOCH - 10 : training on 23279529 raw words (22951015 effective words) took 18.7s, 1224422 effective words/s
    2020-09-30 22:41:52,647 : INFO : EPOCH 11 - PROGRESS: at 5.28% examples, 1216993 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:41:53,651 : INFO : EPOCH 11 - PROGRESS: at 10.85% examples, 1256436 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:54,667 : INFO : EPOCH 11 - PROGRESS: at 16.29% examples, 1244085 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:55,676 : INFO : EPOCH 11 - PROGRESS: at 21.67% examples, 1243136 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:56,680 : INFO : EPOCH 11 - PROGRESS: at 26.98% examples, 1235377 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:57,686 : INFO : EPOCH 11 - PROGRESS: at 32.37% examples, 1234038 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:58,686 : INFO : EPOCH 11 - PROGRESS: at 37.28% examples, 1216537 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:41:59,692 : INFO : EPOCH 11 - PROGRESS: at 42.58% examples, 1216230 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:00,723 : INFO : EPOCH 11 - PROGRESS: at 48.02% examples, 1215516 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:01,726 : INFO : EPOCH 11 - PROGRESS: at 53.51% examples, 1217515 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:02,737 : INFO : EPOCH 11 - PROGRESS: at 58.73% examples, 1216743 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:03,737 : INFO : EPOCH 11 - PROGRESS: at 63.84% examples, 1212750 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:04,741 : INFO : EPOCH 11 - PROGRESS: at 68.82% examples, 1206454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:05,742 : INFO : EPOCH 11 - PROGRESS: at 73.84% examples, 1204651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:06,755 : INFO : EPOCH 11 - PROGRESS: at 79.19% examples, 1204610 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:07,763 : INFO : EPOCH 11 - PROGRESS: at 84.66% examples, 1206784 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:08,767 : INFO : EPOCH 11 - PROGRESS: at 89.70% examples, 1203347 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:09,771 : INFO : EPOCH 11 - PROGRESS: at 94.95% examples, 1201992 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:10,677 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:42:10,682 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:42:10,685 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:42:10,687 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:42:10,690 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:42:10,692 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:42:10,700 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:42:10,701 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:42:10,701 : INFO : EPOCH - 11 : training on 23279529 raw words (22951015 effective words) took 19.1s, 1203849 effective words/s
    2020-09-30 22:42:11,710 : INFO : EPOCH 12 - PROGRESS: at 5.24% examples, 1213409 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:12,723 : INFO : EPOCH 12 - PROGRESS: at 10.65% examples, 1230132 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:13,730 : INFO : EPOCH 12 - PROGRESS: at 16.24% examples, 1242649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:14,731 : INFO : EPOCH 12 - PROGRESS: at 21.47% examples, 1234854 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:15,737 : INFO : EPOCH 12 - PROGRESS: at 26.72% examples, 1226460 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:16,752 : INFO : EPOCH 12 - PROGRESS: at 31.94% examples, 1218649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:17,752 : INFO : EPOCH 12 - PROGRESS: at 37.10% examples, 1211470 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:18,758 : INFO : EPOCH 12 - PROGRESS: at 42.17% examples, 1204578 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:19,765 : INFO : EPOCH 12 - PROGRESS: at 47.60% examples, 1208369 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:20,769 : INFO : EPOCH 12 - PROGRESS: at 52.86% examples, 1206189 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:21,769 : INFO : EPOCH 12 - PROGRESS: at 57.94% examples, 1204070 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:22,774 : INFO : EPOCH 12 - PROGRESS: at 63.24% examples, 1204773 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:23,776 : INFO : EPOCH 12 - PROGRESS: at 68.69% examples, 1207237 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:24,782 : INFO : EPOCH 12 - PROGRESS: at 73.77% examples, 1205684 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:25,786 : INFO : EPOCH 12 - PROGRESS: at 78.85% examples, 1202468 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:26,803 : INFO : EPOCH 12 - PROGRESS: at 83.61% examples, 1194484 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:27,803 : INFO : EPOCH 12 - PROGRESS: at 88.89% examples, 1194814 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:28,821 : INFO : EPOCH 12 - PROGRESS: at 93.62% examples, 1186590 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:29,825 : INFO : EPOCH 12 - PROGRESS: at 98.55% examples, 1183182 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:30,062 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:42:30,063 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:42:30,065 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:42:30,068 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:42:30,071 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:42:30,072 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:42:30,082 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:42:30,084 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:42:30,084 : INFO : EPOCH - 12 : training on 23279529 raw words (22951015 effective words) took 19.4s, 1184266 effective words/s
    2020-09-30 22:42:31,091 : INFO : EPOCH 13 - PROGRESS: at 5.08% examples, 1176516 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:32,097 : INFO : EPOCH 13 - PROGRESS: at 9.41% examples, 1086949 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:33,107 : INFO : EPOCH 13 - PROGRESS: at 13.73% examples, 1052705 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:34,112 : INFO : EPOCH 13 - PROGRESS: at 18.60% examples, 1071677 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:35,115 : INFO : EPOCH 13 - PROGRESS: at 22.74% examples, 1046719 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:36,118 : INFO : EPOCH 13 - PROGRESS: at 26.81% examples, 1026636 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:37,127 : INFO : EPOCH 13 - PROGRESS: at 31.60% examples, 1035861 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:38,133 : INFO : EPOCH 13 - PROGRESS: at 36.57% examples, 1045551 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:39,136 : INFO : EPOCH 13 - PROGRESS: at 41.69% examples, 1060274 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:40,141 : INFO : EPOCH 13 - PROGRESS: at 46.95% examples, 1073474 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:41,145 : INFO : EPOCH 13 - PROGRESS: at 51.87% examples, 1076896 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:42,159 : INFO : EPOCH 13 - PROGRESS: at 56.21% examples, 1069865 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:43,162 : INFO : EPOCH 13 - PROGRESS: at 60.91% examples, 1072153 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:44,168 : INFO : EPOCH 13 - PROGRESS: at 65.56% examples, 1069832 words/s, in_qsize 16, out_qsize 2
    2020-09-30 22:42:45,183 : INFO : EPOCH 13 - PROGRESS: at 70.86% examples, 1078767 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:46,187 : INFO : EPOCH 13 - PROGRESS: at 76.10% examples, 1087893 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:47,191 : INFO : EPOCH 13 - PROGRESS: at 81.61% examples, 1097075 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:48,205 : INFO : EPOCH 13 - PROGRESS: at 87.21% examples, 1106285 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:49,216 : INFO : EPOCH 13 - PROGRESS: at 92.80% examples, 1114172 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:50,217 : INFO : EPOCH 13 - PROGRESS: at 98.03% examples, 1118058 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:50,547 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:42:50,553 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:42:50,553 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:42:50,559 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:42:50,561 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:42:50,563 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:42:50,570 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:42:50,573 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:42:50,573 : INFO : EPOCH - 13 : training on 23279529 raw words (22951015 effective words) took 20.5s, 1120326 effective words/s
    2020-09-30 22:42:51,583 : INFO : EPOCH 14 - PROGRESS: at 4.71% examples, 1086772 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:52,586 : INFO : EPOCH 14 - PROGRESS: at 9.53% examples, 1101042 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:53,595 : INFO : EPOCH 14 - PROGRESS: at 14.79% examples, 1133043 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:42:54,606 : INFO : EPOCH 14 - PROGRESS: at 20.02% examples, 1154453 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:55,610 : INFO : EPOCH 14 - PROGRESS: at 25.47% examples, 1168383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:56,613 : INFO : EPOCH 14 - PROGRESS: at 30.61% examples, 1171078 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:42:57,614 : INFO : EPOCH 14 - PROGRESS: at 35.72% examples, 1167782 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:58,625 : INFO : EPOCH 14 - PROGRESS: at 40.83% examples, 1167887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:42:59,641 : INFO : EPOCH 14 - PROGRESS: at 46.38% examples, 1175771 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:00,641 : INFO : EPOCH 14 - PROGRESS: at 51.61% examples, 1177368 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:01,643 : INFO : EPOCH 14 - PROGRESS: at 57.00% examples, 1183653 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:02,652 : INFO : EPOCH 14 - PROGRESS: at 62.39% examples, 1188089 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:03,661 : INFO : EPOCH 14 - PROGRESS: at 67.46% examples, 1184433 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:04,675 : INFO : EPOCH 14 - PROGRESS: at 72.47% examples, 1181853 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:05,677 : INFO : EPOCH 14 - PROGRESS: at 77.61% examples, 1182258 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:06,689 : INFO : EPOCH 14 - PROGRESS: at 82.51% examples, 1177176 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:07,689 : INFO : EPOCH 14 - PROGRESS: at 87.59% examples, 1176299 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:08,690 : INFO : EPOCH 14 - PROGRESS: at 92.88% examples, 1177654 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:09,694 : INFO : EPOCH 14 - PROGRESS: at 98.21% examples, 1179264 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:09,974 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:43:09,979 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:43:09,981 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:43:09,981 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:43:09,983 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:43:09,984 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:43:09,998 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:43:10,000 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:43:10,000 : INFO : EPOCH - 14 : training on 23279529 raw words (22951015 effective words) took 19.4s, 1181581 effective words/s
    2020-09-30 22:43:11,006 : INFO : EPOCH 15 - PROGRESS: at 5.04% examples, 1169059 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:12,020 : INFO : EPOCH 15 - PROGRESS: at 10.33% examples, 1192786 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:13,025 : INFO : EPOCH 15 - PROGRESS: at 15.66% examples, 1199598 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:43:14,040 : INFO : EPOCH 15 - PROGRESS: at 21.02% examples, 1207847 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:15,045 : INFO : EPOCH 15 - PROGRESS: at 26.36% examples, 1208733 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:16,057 : INFO : EPOCH 15 - PROGRESS: at 31.37% examples, 1196431 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:17,059 : INFO : EPOCH 15 - PROGRESS: at 36.56% examples, 1192360 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:18,059 : INFO : EPOCH 15 - PROGRESS: at 41.57% examples, 1187426 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:19,065 : INFO : EPOCH 15 - PROGRESS: at 46.38% examples, 1176196 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:20,070 : INFO : EPOCH 15 - PROGRESS: at 51.31% examples, 1170374 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:21,070 : INFO : EPOCH 15 - PROGRESS: at 56.60% examples, 1174909 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:22,073 : INFO : EPOCH 15 - PROGRESS: at 61.88% examples, 1178982 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:23,082 : INFO : EPOCH 15 - PROGRESS: at 67.37% examples, 1183558 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:24,086 : INFO : EPOCH 15 - PROGRESS: at 72.81% examples, 1188717 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:25,089 : INFO : EPOCH 15 - PROGRESS: at 78.19% examples, 1191843 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:26,093 : INFO : EPOCH 15 - PROGRESS: at 83.23% examples, 1189713 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:27,101 : INFO : EPOCH 15 - PROGRESS: at 87.72% examples, 1179068 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:28,105 : INFO : EPOCH 15 - PROGRESS: at 91.76% examples, 1164566 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:29,107 : INFO : EPOCH 15 - PROGRESS: at 96.70% examples, 1161944 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:29,701 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:43:29,704 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:43:29,708 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:43:29,710 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:43:29,711 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:43:29,712 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:43:29,719 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:43:29,723 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:43:29,723 : INFO : EPOCH - 15 : training on 23279529 raw words (22951015 effective words) took 19.7s, 1163839 effective words/s
    2020-09-30 22:43:30,747 : INFO : EPOCH 16 - PROGRESS: at 5.03% examples, 1148006 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:31,753 : INFO : EPOCH 16 - PROGRESS: at 10.37% examples, 1191820 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:32,766 : INFO : EPOCH 16 - PROGRESS: at 15.66% examples, 1192330 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:33,783 : INFO : EPOCH 16 - PROGRESS: at 20.37% examples, 1165833 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:34,797 : INFO : EPOCH 16 - PROGRESS: at 24.71% examples, 1123999 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:35,805 : INFO : EPOCH 16 - PROGRESS: at 29.73% examples, 1129538 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:43:36,813 : INFO : EPOCH 16 - PROGRESS: at 34.72% examples, 1128446 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:37,819 : INFO : EPOCH 16 - PROGRESS: at 39.98% examples, 1136526 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:38,822 : INFO : EPOCH 16 - PROGRESS: at 44.84% examples, 1134569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:39,836 : INFO : EPOCH 16 - PROGRESS: at 49.74% examples, 1130700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:40,845 : INFO : EPOCH 16 - PROGRESS: at 54.25% examples, 1120524 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:43:41,848 : INFO : EPOCH 16 - PROGRESS: at 58.77% examples, 1114984 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:43:42,849 : INFO : EPOCH 16 - PROGRESS: at 63.77% examples, 1116806 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:43,864 : INFO : EPOCH 16 - PROGRESS: at 68.73% examples, 1116856 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:44,866 : INFO : EPOCH 16 - PROGRESS: at 73.77% examples, 1121043 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:45,877 : INFO : EPOCH 16 - PROGRESS: at 79.11% examples, 1126476 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:46,880 : INFO : EPOCH 16 - PROGRESS: at 84.36% examples, 1130656 words/s, in_qsize 15, out_qsize 1
    2020-09-30 22:43:47,886 : INFO : EPOCH 16 - PROGRESS: at 89.85% examples, 1137280 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:43:48,888 : INFO : EPOCH 16 - PROGRESS: at 94.71% examples, 1134564 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:43:49,828 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:43:49,834 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:43:49,835 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:43:49,839 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:43:49,842 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:43:49,843 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:43:49,850 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:43:49,853 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:43:49,853 : INFO : EPOCH - 16 : training on 23279529 raw words (22951015 effective words) took 20.1s, 1140322 effective words/s
    2020-09-30 22:43:50,867 : INFO : EPOCH 17 - PROGRESS: at 4.99% examples, 1148756 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:51,871 : INFO : EPOCH 17 - PROGRESS: at 10.02% examples, 1155615 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:52,879 : INFO : EPOCH 17 - PROGRESS: at 15.21% examples, 1163883 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:53,879 : INFO : EPOCH 17 - PROGRESS: at 20.50% examples, 1182766 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:54,887 : INFO : EPOCH 17 - PROGRESS: at 25.64% examples, 1176517 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:55,891 : INFO : EPOCH 17 - PROGRESS: at 30.94% examples, 1184077 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:56,895 : INFO : EPOCH 17 - PROGRESS: at 35.97% examples, 1175834 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:43:57,906 : INFO : EPOCH 17 - PROGRESS: at 40.83% examples, 1167651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:58,918 : INFO : EPOCH 17 - PROGRESS: at 45.66% examples, 1158895 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:43:59,934 : INFO : EPOCH 17 - PROGRESS: at 50.88% examples, 1159432 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:00,946 : INFO : EPOCH 17 - PROGRESS: at 55.88% examples, 1157545 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:01,950 : INFO : EPOCH 17 - PROGRESS: at 60.82% examples, 1157464 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:02,961 : INFO : EPOCH 17 - PROGRESS: at 66.11% examples, 1159071 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:03,965 : INFO : EPOCH 17 - PROGRESS: at 71.42% examples, 1163791 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:04,967 : INFO : EPOCH 17 - PROGRESS: at 76.67% examples, 1167344 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:44:05,967 : INFO : EPOCH 17 - PROGRESS: at 82.11% examples, 1171839 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:06,972 : INFO : EPOCH 17 - PROGRESS: at 87.38% examples, 1173229 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:07,977 : INFO : EPOCH 17 - PROGRESS: at 92.76% examples, 1175590 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:08,979 : INFO : EPOCH 17 - PROGRESS: at 97.83% examples, 1174413 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:09,329 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:44:09,329 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:44:09,331 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:44:09,334 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:44:09,338 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:44:09,339 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:44:09,351 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:44:09,352 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:44:09,352 : INFO : EPOCH - 17 : training on 23279529 raw words (22951015 effective words) took 19.5s, 1177191 effective words/s
    2020-09-30 22:44:10,358 : INFO : EPOCH 18 - PROGRESS: at 5.08% examples, 1177268 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:44:11,374 : INFO : EPOCH 18 - PROGRESS: at 10.09% examples, 1162965 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:12,375 : INFO : EPOCH 18 - PROGRESS: at 15.21% examples, 1164978 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:13,385 : INFO : EPOCH 18 - PROGRESS: at 20.37% examples, 1173518 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:44:14,392 : INFO : EPOCH 18 - PROGRESS: at 25.83% examples, 1184972 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:15,394 : INFO : EPOCH 18 - PROGRESS: at 31.18% examples, 1191451 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:16,395 : INFO : EPOCH 18 - PROGRESS: at 36.52% examples, 1193605 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:17,397 : INFO : EPOCH 18 - PROGRESS: at 41.87% examples, 1197806 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:18,401 : INFO : EPOCH 18 - PROGRESS: at 47.12% examples, 1197465 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:19,401 : INFO : EPOCH 18 - PROGRESS: at 52.54% examples, 1200688 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:44:20,405 : INFO : EPOCH 18 - PROGRESS: at 57.82% examples, 1203074 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:21,415 : INFO : EPOCH 18 - PROGRESS: at 63.11% examples, 1203290 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:22,447 : INFO : EPOCH 18 - PROGRESS: at 68.38% examples, 1200122 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:23,465 : INFO : EPOCH 18 - PROGRESS: at 73.77% examples, 1202919 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:44:24,470 : INFO : EPOCH 18 - PROGRESS: at 79.11% examples, 1203659 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:25,470 : INFO : EPOCH 18 - PROGRESS: at 84.13% examples, 1200456 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:26,479 : INFO : EPOCH 18 - PROGRESS: at 89.28% examples, 1198131 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:27,487 : INFO : EPOCH 18 - PROGRESS: at 94.56% examples, 1197356 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:28,462 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:44:28,467 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:44:28,469 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:44:28,470 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:44:28,473 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:44:28,475 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:44:28,479 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:44:28,489 : INFO : EPOCH 18 - PROGRESS: at 100.00% examples, 1199470 words/s, in_qsize 0, out_qsize 1
    2020-09-30 22:44:28,489 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:44:28,489 : INFO : EPOCH - 18 : training on 23279529 raw words (22951015 effective words) took 19.1s, 1199451 effective words/s
    2020-09-30 22:44:29,494 : INFO : EPOCH 19 - PROGRESS: at 5.20% examples, 1207824 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:30,502 : INFO : EPOCH 19 - PROGRESS: at 10.45% examples, 1211193 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:44:31,509 : INFO : EPOCH 19 - PROGRESS: at 15.99% examples, 1227394 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:32,515 : INFO : EPOCH 19 - PROGRESS: at 21.34% examples, 1228856 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:33,519 : INFO : EPOCH 19 - PROGRESS: at 26.54% examples, 1220169 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:34,531 : INFO : EPOCH 19 - PROGRESS: at 31.39% examples, 1199499 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:35,556 : INFO : EPOCH 19 - PROGRESS: at 36.74% examples, 1196410 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:44:36,566 : INFO : EPOCH 19 - PROGRESS: at 41.87% examples, 1193103 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:37,570 : INFO : EPOCH 19 - PROGRESS: at 46.75% examples, 1183500 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:38,578 : INFO : EPOCH 19 - PROGRESS: at 51.48% examples, 1172054 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:39,591 : INFO : EPOCH 19 - PROGRESS: at 56.17% examples, 1162790 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:40,600 : INFO : EPOCH 19 - PROGRESS: at 60.91% examples, 1157797 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:44:41,603 : INFO : EPOCH 19 - PROGRESS: at 66.11% examples, 1158603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:42,610 : INFO : EPOCH 19 - PROGRESS: at 71.33% examples, 1161750 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:44:43,627 : INFO : EPOCH 19 - PROGRESS: at 76.86% examples, 1168766 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:44,631 : INFO : EPOCH 19 - PROGRESS: at 82.47% examples, 1174692 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:45,636 : INFO : EPOCH 19 - PROGRESS: at 88.22% examples, 1182700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:46,637 : INFO : EPOCH 19 - PROGRESS: at 93.86% examples, 1187942 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:47,654 : INFO : EPOCH 19 - PROGRESS: at 99.61% examples, 1193267 words/s, in_qsize 9, out_qsize 0
    2020-09-30 22:44:47,682 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:44:47,682 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:44:47,685 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:44:47,686 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:44:47,690 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:44:47,692 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:44:47,699 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:44:47,703 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:44:47,703 : INFO : EPOCH - 19 : training on 23279529 raw words (22951015 effective words) took 19.2s, 1194667 effective words/s
    2020-09-30 22:44:48,709 : INFO : EPOCH 20 - PROGRESS: at 5.63% examples, 1303461 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:49,711 : INFO : EPOCH 20 - PROGRESS: at 10.49% examples, 1219230 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:50,723 : INFO : EPOCH 20 - PROGRESS: at 15.87% examples, 1217482 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:51,736 : INFO : EPOCH 20 - PROGRESS: at 21.06% examples, 1212412 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:52,753 : INFO : EPOCH 20 - PROGRESS: at 26.01% examples, 1190365 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:53,754 : INFO : EPOCH 20 - PROGRESS: at 31.22% examples, 1191444 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:54,771 : INFO : EPOCH 20 - PROGRESS: at 36.52% examples, 1189415 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:55,772 : INFO : EPOCH 20 - PROGRESS: at 41.78% examples, 1191921 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:56,786 : INFO : EPOCH 20 - PROGRESS: at 47.07% examples, 1191888 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:57,796 : INFO : EPOCH 20 - PROGRESS: at 52.62% examples, 1197368 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:44:58,799 : INFO : EPOCH 20 - PROGRESS: at 57.70% examples, 1195866 words/s, in_qsize 15, out_qsize 2
    2020-09-30 22:44:59,800 : INFO : EPOCH 20 - PROGRESS: at 63.25% examples, 1202264 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:00,804 : INFO : EPOCH 20 - PROGRESS: at 68.52% examples, 1201864 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:01,807 : INFO : EPOCH 20 - PROGRESS: at 73.81% examples, 1204338 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:02,815 : INFO : EPOCH 20 - PROGRESS: at 79.19% examples, 1205375 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:03,817 : INFO : EPOCH 20 - PROGRESS: at 84.36% examples, 1203806 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:04,818 : INFO : EPOCH 20 - PROGRESS: at 89.45% examples, 1201252 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:05,819 : INFO : EPOCH 20 - PROGRESS: at 94.32% examples, 1195924 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:06,820 : INFO : EPOCH 20 - PROGRESS: at 99.32% examples, 1192737 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:06,894 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:45:06,904 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:45:06,906 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:45:06,910 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:45:06,920 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:45:06,921 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:45:06,923 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:45:06,925 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:45:06,925 : INFO : EPOCH - 20 : training on 23279529 raw words (22951015 effective words) took 19.2s, 1194194 effective words/s
    2020-09-30 22:45:06,925 : INFO : training on a 465590580 raw words (459020300 effective words) took 383.3s, 1197567 effective words/s

    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)

    0.105600 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec(dm/m,d100,n5,w10,mc2,t8)
    2020-09-30 22:45:07,659 : INFO : training model with 8 workers on 265408 vocabulary and 100 features, using sg=0 hs=0 sample=0 negative=5 window=10
    2020-09-30 22:45:08,666 : INFO : EPOCH 1 - PROGRESS: at 2.35% examples, 548427 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:09,682 : INFO : EPOCH 1 - PROGRESS: at 5.28% examples, 608661 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:10,688 : INFO : EPOCH 1 - PROGRESS: at 8.10% examples, 619824 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:11,697 : INFO : EPOCH 1 - PROGRESS: at 10.96% examples, 632160 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:12,725 : INFO : EPOCH 1 - PROGRESS: at 13.74% examples, 628037 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:13,730 : INFO : EPOCH 1 - PROGRESS: at 16.67% examples, 637295 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:14,743 : INFO : EPOCH 1 - PROGRESS: at 19.53% examples, 639311 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:15,743 : INFO : EPOCH 1 - PROGRESS: at 22.41% examples, 641702 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:16,745 : INFO : EPOCH 1 - PROGRESS: at 25.51% examples, 648623 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:17,761 : INFO : EPOCH 1 - PROGRESS: at 28.76% examples, 657878 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:18,773 : INFO : EPOCH 1 - PROGRESS: at 31.99% examples, 664091 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:19,775 : INFO : EPOCH 1 - PROGRESS: at 35.27% examples, 670543 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:20,779 : INFO : EPOCH 1 - PROGRESS: at 38.49% examples, 675356 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:21,790 : INFO : EPOCH 1 - PROGRESS: at 41.69% examples, 679084 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:22,808 : INFO : EPOCH 1 - PROGRESS: at 44.92% examples, 682633 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:23,809 : INFO : EPOCH 1 - PROGRESS: at 48.35% examples, 688218 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:24,810 : INFO : EPOCH 1 - PROGRESS: at 51.71% examples, 692167 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:25,824 : INFO : EPOCH 1 - PROGRESS: at 55.11% examples, 696745 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:26,828 : INFO : EPOCH 1 - PROGRESS: at 58.39% examples, 700651 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:27,834 : INFO : EPOCH 1 - PROGRESS: at 61.67% examples, 703037 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:28,844 : INFO : EPOCH 1 - PROGRESS: at 65.14% examples, 706601 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:29,849 : INFO : EPOCH 1 - PROGRESS: at 68.47% examples, 709055 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:30,852 : INFO : EPOCH 1 - PROGRESS: at 71.73% examples, 711014 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:31,861 : INFO : EPOCH 1 - PROGRESS: at 75.10% examples, 714131 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:32,873 : INFO : EPOCH 1 - PROGRESS: at 78.19% examples, 713170 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:33,880 : INFO : EPOCH 1 - PROGRESS: at 81.53% examples, 714986 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:34,901 : INFO : EPOCH 1 - PROGRESS: at 84.70% examples, 714837 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:35,927 : INFO : EPOCH 1 - PROGRESS: at 88.05% examples, 715983 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:36,928 : INFO : EPOCH 1 - PROGRESS: at 91.21% examples, 715976 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:37,947 : INFO : EPOCH 1 - PROGRESS: at 94.37% examples, 715603 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:38,947 : INFO : EPOCH 1 - PROGRESS: at 97.26% examples, 713526 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:39,725 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:45:39,731 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:45:39,732 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:45:39,739 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:45:39,741 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:45:39,742 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:45:39,756 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:45:39,757 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:45:39,757 : INFO : EPOCH - 1 : training on 23279529 raw words (22951015 effective words) took 32.1s, 715091 effective words/s
    2020-09-30 22:45:40,766 : INFO : EPOCH 2 - PROGRESS: at 3.00% examples, 692022 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:41,774 : INFO : EPOCH 2 - PROGRESS: at 6.05% examples, 696651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:42,786 : INFO : EPOCH 2 - PROGRESS: at 9.16% examples, 702760 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:43,797 : INFO : EPOCH 2 - PROGRESS: at 12.24% examples, 703596 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:44,804 : INFO : EPOCH 2 - PROGRESS: at 15.13% examples, 693553 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:45,806 : INFO : EPOCH 2 - PROGRESS: at 18.05% examples, 692473 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:46,811 : INFO : EPOCH 2 - PROGRESS: at 21.02% examples, 691519 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:47,819 : INFO : EPOCH 2 - PROGRESS: at 23.57% examples, 675702 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:48,836 : INFO : EPOCH 2 - PROGRESS: at 26.58% examples, 676807 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:49,841 : INFO : EPOCH 2 - PROGRESS: at 29.55% examples, 677251 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:50,863 : INFO : EPOCH 2 - PROGRESS: at 32.66% examples, 678347 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:51,882 : INFO : EPOCH 2 - PROGRESS: at 35.76% examples, 678809 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:52,900 : INFO : EPOCH 2 - PROGRESS: at 38.62% examples, 676355 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:53,901 : INFO : EPOCH 2 - PROGRESS: at 41.26% examples, 671628 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:45:54,918 : INFO : EPOCH 2 - PROGRESS: at 44.29% examples, 671793 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:55,926 : INFO : EPOCH 2 - PROGRESS: at 47.32% examples, 673054 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:56,952 : INFO : EPOCH 2 - PROGRESS: at 49.74% examples, 664932 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:57,952 : INFO : EPOCH 2 - PROGRESS: at 52.27% examples, 659843 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:58,987 : INFO : EPOCH 2 - PROGRESS: at 54.72% examples, 653579 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:45:59,995 : INFO : EPOCH 2 - PROGRESS: at 56.84% examples, 645398 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:01,038 : INFO : EPOCH 2 - PROGRESS: at 59.15% examples, 639229 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:02,067 : INFO : EPOCH 2 - PROGRESS: at 61.91% examples, 638337 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:03,074 : INFO : EPOCH 2 - PROGRESS: at 65.14% examples, 641954 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:04,093 : INFO : EPOCH 2 - PROGRESS: at 68.04% examples, 642498 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:05,125 : INFO : EPOCH 2 - PROGRESS: at 70.85% examples, 642015 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:06,142 : INFO : EPOCH 2 - PROGRESS: at 73.81% examples, 643690 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:07,143 : INFO : EPOCH 2 - PROGRESS: at 76.82% examples, 645613 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:08,148 : INFO : EPOCH 2 - PROGRESS: at 79.42% examples, 643238 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:09,156 : INFO : EPOCH 2 - PROGRESS: at 81.99% examples, 641282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:10,161 : INFO : EPOCH 2 - PROGRESS: at 84.87% examples, 641734 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:11,174 : INFO : EPOCH 2 - PROGRESS: at 87.88% examples, 642963 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:12,189 : INFO : EPOCH 2 - PROGRESS: at 90.11% examples, 638621 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:13,206 : INFO : EPOCH 2 - PROGRESS: at 92.92% examples, 638083 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:14,215 : INFO : EPOCH 2 - PROGRESS: at 95.45% examples, 635756 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:15,238 : INFO : EPOCH 2 - PROGRESS: at 98.07% examples, 634640 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:15,916 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:46:15,924 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:46:15,937 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:46:15,941 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:46:15,945 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:46:15,946 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:46:15,960 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:46:15,963 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:46:15,963 : INFO : EPOCH - 2 : training on 23279529 raw words (22951015 effective words) took 36.2s, 633941 effective words/s
    2020-09-30 22:46:17,015 : INFO : EPOCH 3 - PROGRESS: at 1.36% examples, 304582 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:18,015 : INFO : EPOCH 3 - PROGRESS: at 3.61% examples, 410865 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:46:19,051 : INFO : EPOCH 3 - PROGRESS: at 5.80% examples, 436241 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:20,055 : INFO : EPOCH 3 - PROGRESS: at 7.90% examples, 447092 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:21,056 : INFO : EPOCH 3 - PROGRESS: at 9.93% examples, 453686 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:22,058 : INFO : EPOCH 3 - PROGRESS: at 12.02% examples, 458457 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:23,064 : INFO : EPOCH 3 - PROGRESS: at 14.33% examples, 467007 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:24,065 : INFO : EPOCH 3 - PROGRESS: at 16.55% examples, 473911 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:25,083 : INFO : EPOCH 3 - PROGRESS: at 18.98% examples, 482773 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:26,097 : INFO : EPOCH 3 - PROGRESS: at 21.06% examples, 482215 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:46:27,109 : INFO : EPOCH 3 - PROGRESS: at 23.52% examples, 487862 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:28,114 : INFO : EPOCH 3 - PROGRESS: at 26.09% examples, 496102 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:29,114 : INFO : EPOCH 3 - PROGRESS: at 28.43% examples, 499456 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:30,116 : INFO : EPOCH 3 - PROGRESS: at 30.65% examples, 500280 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:31,123 : INFO : EPOCH 3 - PROGRESS: at 33.37% examples, 507158 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:32,126 : INFO : EPOCH 3 - PROGRESS: at 36.04% examples, 513351 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:33,132 : INFO : EPOCH 3 - PROGRESS: at 38.70% examples, 518888 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:34,139 : INFO : EPOCH 3 - PROGRESS: at 41.39% examples, 524222 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:35,153 : INFO : EPOCH 3 - PROGRESS: at 44.48% examples, 533314 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:36,192 : INFO : EPOCH 3 - PROGRESS: at 47.37% examples, 538454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:37,202 : INFO : EPOCH 3 - PROGRESS: at 50.41% examples, 545218 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:38,204 : INFO : EPOCH 3 - PROGRESS: at 53.08% examples, 548113 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:39,214 : INFO : EPOCH 3 - PROGRESS: at 55.72% examples, 550546 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:40,216 : INFO : EPOCH 3 - PROGRESS: at 58.35% examples, 553380 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:41,222 : INFO : EPOCH 3 - PROGRESS: at 61.05% examples, 556199 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:42,235 : INFO : EPOCH 3 - PROGRESS: at 63.25% examples, 553534 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:43,253 : INFO : EPOCH 3 - PROGRESS: at 65.93% examples, 555252 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:44,274 : INFO : EPOCH 3 - PROGRESS: at 68.73% examples, 557805 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:45,282 : INFO : EPOCH 3 - PROGRESS: at 71.38% examples, 559776 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:46,292 : INFO : EPOCH 3 - PROGRESS: at 74.05% examples, 561890 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:47,333 : INFO : EPOCH 3 - PROGRESS: at 76.82% examples, 563625 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:48,338 : INFO : EPOCH 3 - PROGRESS: at 79.57% examples, 565292 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:49,345 : INFO : EPOCH 3 - PROGRESS: at 82.43% examples, 567673 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:50,363 : INFO : EPOCH 3 - PROGRESS: at 85.01% examples, 568055 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:46:51,386 : INFO : EPOCH 3 - PROGRESS: at 87.88% examples, 570250 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:52,420 : INFO : EPOCH 3 - PROGRESS: at 90.74% examples, 571887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:53,440 : INFO : EPOCH 3 - PROGRESS: at 93.86% examples, 575202 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:54,443 : INFO : EPOCH 3 - PROGRESS: at 96.91% examples, 578150 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:55,333 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:46:55,347 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:46:55,349 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:46:55,352 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:46:55,363 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:46:55,365 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:46:55,370 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:46:55,377 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:46:55,377 : INFO : EPOCH - 3 : training on 23279529 raw words (22951015 effective words) took 39.4s, 582343 effective words/s
    2020-09-30 22:46:56,383 : INFO : EPOCH 4 - PROGRESS: at 2.87% examples, 666594 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:57,388 : INFO : EPOCH 4 - PROGRESS: at 5.76% examples, 665539 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:58,391 : INFO : EPOCH 4 - PROGRESS: at 8.21% examples, 632843 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:46:59,402 : INFO : EPOCH 4 - PROGRESS: at 10.85% examples, 629651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:00,418 : INFO : EPOCH 4 - PROGRESS: at 13.74% examples, 631361 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:01,433 : INFO : EPOCH 4 - PROGRESS: at 16.55% examples, 634195 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:02,439 : INFO : EPOCH 4 - PROGRESS: at 19.42% examples, 637339 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:03,439 : INFO : EPOCH 4 - PROGRESS: at 22.54% examples, 647170 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:04,445 : INFO : EPOCH 4 - PROGRESS: at 25.64% examples, 653128 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:05,446 : INFO : EPOCH 4 - PROGRESS: at 28.56% examples, 655333 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:06,455 : INFO : EPOCH 4 - PROGRESS: at 31.60% examples, 658491 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:07,460 : INFO : EPOCH 4 - PROGRESS: at 34.68% examples, 661245 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:08,470 : INFO : EPOCH 4 - PROGRESS: at 37.63% examples, 661227 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:09,489 : INFO : EPOCH 4 - PROGRESS: at 40.68% examples, 663539 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:10,491 : INFO : EPOCH 4 - PROGRESS: at 43.86% examples, 667583 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:11,521 : INFO : EPOCH 4 - PROGRESS: at 47.03% examples, 669945 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:12,526 : INFO : EPOCH 4 - PROGRESS: at 50.27% examples, 673579 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:13,535 : INFO : EPOCH 4 - PROGRESS: at 53.44% examples, 675662 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:14,536 : INFO : EPOCH 4 - PROGRESS: at 56.56% examples, 678314 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:15,551 : INFO : EPOCH 4 - PROGRESS: at 59.62% examples, 680066 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:47:16,567 : INFO : EPOCH 4 - PROGRESS: at 62.76% examples, 681323 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:17,582 : INFO : EPOCH 4 - PROGRESS: at 66.15% examples, 684631 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:18,582 : INFO : EPOCH 4 - PROGRESS: at 69.36% examples, 687279 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:19,600 : INFO : EPOCH 4 - PROGRESS: at 72.81% examples, 691197 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:20,617 : INFO : EPOCH 4 - PROGRESS: at 76.10% examples, 694067 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:21,617 : INFO : EPOCH 4 - PROGRESS: at 79.46% examples, 696387 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:22,626 : INFO : EPOCH 4 - PROGRESS: at 82.51% examples, 696184 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:23,629 : INFO : EPOCH 4 - PROGRESS: at 85.38% examples, 694826 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:24,634 : INFO : EPOCH 4 - PROGRESS: at 88.40% examples, 694455 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:25,661 : INFO : EPOCH 4 - PROGRESS: at 91.39% examples, 693306 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:26,677 : INFO : EPOCH 4 - PROGRESS: at 94.56% examples, 693748 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:27,689 : INFO : EPOCH 4 - PROGRESS: at 97.83% examples, 695156 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:28,306 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:47:28,312 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:47:28,321 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:47:28,322 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:47:28,330 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:47:28,330 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:47:28,330 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:47:28,349 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:47:28,349 : INFO : EPOCH - 4 : training on 23279529 raw words (22951015 effective words) took 33.0s, 696165 effective words/s
    2020-09-30 22:47:29,359 : INFO : EPOCH 5 - PROGRESS: at 2.83% examples, 653079 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:30,367 : INFO : EPOCH 5 - PROGRESS: at 5.80% examples, 667789 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:31,369 : INFO : EPOCH 5 - PROGRESS: at 8.76% examples, 673078 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:32,380 : INFO : EPOCH 5 - PROGRESS: at 11.72% examples, 676480 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:33,382 : INFO : EPOCH 5 - PROGRESS: at 14.91% examples, 685780 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:34,394 : INFO : EPOCH 5 - PROGRESS: at 18.01% examples, 691332 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:35,406 : INFO : EPOCH 5 - PROGRESS: at 21.06% examples, 692591 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:36,442 : INFO : EPOCH 5 - PROGRESS: at 24.06% examples, 686427 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:37,445 : INFO : EPOCH 5 - PROGRESS: at 27.07% examples, 687277 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:38,476 : INFO : EPOCH 5 - PROGRESS: at 30.19% examples, 688773 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:39,490 : INFO : EPOCH 5 - PROGRESS: at 33.37% examples, 690084 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:40,497 : INFO : EPOCH 5 - PROGRESS: at 36.56% examples, 692620 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:41,500 : INFO : EPOCH 5 - PROGRESS: at 39.79% examples, 696630 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:42,544 : INFO : EPOCH 5 - PROGRESS: at 42.88% examples, 695120 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:43,617 : INFO : EPOCH 5 - PROGRESS: at 46.03% examples, 693068 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:44,627 : INFO : EPOCH 5 - PROGRESS: at 49.13% examples, 694104 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:45,639 : INFO : EPOCH 5 - PROGRESS: at 52.23% examples, 693819 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:46,644 : INFO : EPOCH 5 - PROGRESS: at 54.67% examples, 686473 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:47,651 : INFO : EPOCH 5 - PROGRESS: at 57.66% examples, 686838 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:48,665 : INFO : EPOCH 5 - PROGRESS: at 60.62% examples, 686745 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:49,703 : INFO : EPOCH 5 - PROGRESS: at 63.11% examples, 679672 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:50,721 : INFO : EPOCH 5 - PROGRESS: at 65.84% examples, 676445 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:51,728 : INFO : EPOCH 5 - PROGRESS: at 68.69% examples, 675073 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:52,730 : INFO : EPOCH 5 - PROGRESS: at 71.28% examples, 672366 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:47:53,732 : INFO : EPOCH 5 - PROGRESS: at 73.88% examples, 669889 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:54,739 : INFO : EPOCH 5 - PROGRESS: at 76.82% examples, 669991 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:55,748 : INFO : EPOCH 5 - PROGRESS: at 79.84% examples, 670398 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:56,782 : INFO : EPOCH 5 - PROGRESS: at 82.64% examples, 668184 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:57,812 : INFO : EPOCH 5 - PROGRESS: at 85.72% examples, 668881 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:58,826 : INFO : EPOCH 5 - PROGRESS: at 88.75% examples, 669158 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:47:59,849 : INFO : EPOCH 5 - PROGRESS: at 91.72% examples, 668980 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:00,855 : INFO : EPOCH 5 - PROGRESS: at 94.56% examples, 667955 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:01,859 : INFO : EPOCH 5 - PROGRESS: at 97.55% examples, 668233 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:02,598 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:48:02,606 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:48:02,610 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:48:02,611 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:48:02,616 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:48:02,620 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:48:02,636 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:48:02,637 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:48:02,637 : INFO : EPOCH - 5 : training on 23279529 raw words (22951015 effective words) took 34.3s, 669411 effective words/s
    2020-09-30 22:48:03,647 : INFO : EPOCH 6 - PROGRESS: at 2.79% examples, 643214 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:04,679 : INFO : EPOCH 6 - PROGRESS: at 5.76% examples, 655007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:05,696 : INFO : EPOCH 6 - PROGRESS: at 8.55% examples, 648482 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:06,721 : INFO : EPOCH 6 - PROGRESS: at 11.26% examples, 641626 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:07,730 : INFO : EPOCH 6 - PROGRESS: at 13.95% examples, 634009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:08,748 : INFO : EPOCH 6 - PROGRESS: at 16.67% examples, 633106 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:09,771 : INFO : EPOCH 6 - PROGRESS: at 19.45% examples, 632048 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:10,771 : INFO : EPOCH 6 - PROGRESS: at 22.41% examples, 637689 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:11,774 : INFO : EPOCH 6 - PROGRESS: at 25.35% examples, 640747 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:12,784 : INFO : EPOCH 6 - PROGRESS: at 28.35% examples, 645383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:13,784 : INFO : EPOCH 6 - PROGRESS: at 31.26% examples, 647326 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:14,790 : INFO : EPOCH 6 - PROGRESS: at 34.32% examples, 650187 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:15,799 : INFO : EPOCH 6 - PROGRESS: at 37.41% examples, 653956 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:16,805 : INFO : EPOCH 6 - PROGRESS: at 40.36% examples, 655418 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:17,808 : INFO : EPOCH 6 - PROGRESS: at 43.43% examples, 658635 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:18,839 : INFO : EPOCH 6 - PROGRESS: at 46.38% examples, 657886 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:19,863 : INFO : EPOCH 6 - PROGRESS: at 49.38% examples, 659267 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:20,880 : INFO : EPOCH 6 - PROGRESS: at 52.36% examples, 659172 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:21,885 : INFO : EPOCH 6 - PROGRESS: at 55.33% examples, 660003 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:22,892 : INFO : EPOCH 6 - PROGRESS: at 58.24% examples, 661180 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:23,897 : INFO : EPOCH 6 - PROGRESS: at 61.05% examples, 660803 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:24,924 : INFO : EPOCH 6 - PROGRESS: at 64.14% examples, 661602 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:25,942 : INFO : EPOCH 6 - PROGRESS: at 67.12% examples, 661790 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:26,954 : INFO : EPOCH 6 - PROGRESS: at 70.03% examples, 661798 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:27,962 : INFO : EPOCH 6 - PROGRESS: at 72.77% examples, 660678 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:28,969 : INFO : EPOCH 6 - PROGRESS: at 75.14% examples, 656712 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:48:29,972 : INFO : EPOCH 6 - PROGRESS: at 77.57% examples, 652820 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:30,983 : INFO : EPOCH 6 - PROGRESS: at 80.25% examples, 651404 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:31,994 : INFO : EPOCH 6 - PROGRESS: at 83.26% examples, 652429 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:33,018 : INFO : EPOCH 6 - PROGRESS: at 86.47% examples, 654048 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:34,022 : INFO : EPOCH 6 - PROGRESS: at 89.63% examples, 656227 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:35,026 : INFO : EPOCH 6 - PROGRESS: at 92.72% examples, 657465 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:36,031 : INFO : EPOCH 6 - PROGRESS: at 95.86% examples, 658918 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:37,048 : INFO : EPOCH 6 - PROGRESS: at 98.92% examples, 659995 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:37,303 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:48:37,307 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:48:37,311 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:48:37,318 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:48:37,322 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:48:37,324 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:48:37,344 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:48:37,346 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:48:37,346 : INFO : EPOCH - 6 : training on 23279529 raw words (22951015 effective words) took 34.7s, 661277 effective words/s
    2020-09-30 22:48:38,357 : INFO : EPOCH 7 - PROGRESS: at 2.92% examples, 671649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:39,367 : INFO : EPOCH 7 - PROGRESS: at 6.01% examples, 690448 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:40,385 : INFO : EPOCH 7 - PROGRESS: at 9.07% examples, 694103 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:41,404 : INFO : EPOCH 7 - PROGRESS: at 12.37% examples, 707772 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:42,408 : INFO : EPOCH 7 - PROGRESS: at 15.45% examples, 706833 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:43,422 : INFO : EPOCH 7 - PROGRESS: at 18.69% examples, 713393 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:44,437 : INFO : EPOCH 7 - PROGRESS: at 21.80% examples, 712436 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:45,469 : INFO : EPOCH 7 - PROGRESS: at 25.02% examples, 711421 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:46,472 : INFO : EPOCH 7 - PROGRESS: at 27.95% examples, 707251 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:47,473 : INFO : EPOCH 7 - PROGRESS: at 31.22% examples, 711629 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:48,488 : INFO : EPOCH 7 - PROGRESS: at 34.36% examples, 710048 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:49,491 : INFO : EPOCH 7 - PROGRESS: at 37.63% examples, 712774 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:50,493 : INFO : EPOCH 7 - PROGRESS: at 40.64% examples, 711449 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:51,506 : INFO : EPOCH 7 - PROGRESS: at 43.90% examples, 713189 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:52,506 : INFO : EPOCH 7 - PROGRESS: at 46.87% examples, 710743 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:48:53,508 : INFO : EPOCH 7 - PROGRESS: at 50.02% examples, 711029 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:54,531 : INFO : EPOCH 7 - PROGRESS: at 53.08% examples, 709366 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:55,548 : INFO : EPOCH 7 - PROGRESS: at 56.34% examples, 711216 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:56,579 : INFO : EPOCH 7 - PROGRESS: at 59.50% examples, 711786 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:57,589 : INFO : EPOCH 7 - PROGRESS: at 62.72% examples, 712657 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:58,592 : INFO : EPOCH 7 - PROGRESS: at 65.89% examples, 712725 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:48:59,619 : INFO : EPOCH 7 - PROGRESS: at 69.06% examples, 712514 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:00,620 : INFO : EPOCH 7 - PROGRESS: at 72.18% examples, 713126 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:01,620 : INFO : EPOCH 7 - PROGRESS: at 75.26% examples, 713614 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:02,631 : INFO : EPOCH 7 - PROGRESS: at 78.43% examples, 713454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:03,668 : INFO : EPOCH 7 - PROGRESS: at 81.61% examples, 712930 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:04,690 : INFO : EPOCH 7 - PROGRESS: at 84.70% examples, 712156 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:05,693 : INFO : EPOCH 7 - PROGRESS: at 87.97% examples, 713284 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:06,704 : INFO : EPOCH 7 - PROGRESS: at 91.08% examples, 712810 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:07,717 : INFO : EPOCH 7 - PROGRESS: at 94.32% examples, 713291 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:08,731 : INFO : EPOCH 7 - PROGRESS: at 97.34% examples, 711930 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:09,493 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:49:09,501 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:49:09,505 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:49:09,507 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:49:09,510 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:49:09,514 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:49:09,527 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:49:09,529 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:49:09,529 : INFO : EPOCH - 7 : training on 23279529 raw words (22951015 effective words) took 32.2s, 713188 effective words/s
    2020-09-30 22:49:10,535 : INFO : EPOCH 8 - PROGRESS: at 2.87% examples, 665635 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:11,539 : INFO : EPOCH 8 - PROGRESS: at 5.98% examples, 689634 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:12,549 : INFO : EPOCH 8 - PROGRESS: at 9.03% examples, 695436 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:13,550 : INFO : EPOCH 8 - PROGRESS: at 12.29% examples, 709441 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:14,573 : INFO : EPOCH 8 - PROGRESS: at 15.42% examples, 707581 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:15,589 : INFO : EPOCH 8 - PROGRESS: at 18.64% examples, 713731 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:16,592 : INFO : EPOCH 8 - PROGRESS: at 21.70% examples, 712612 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:17,611 : INFO : EPOCH 8 - PROGRESS: at 25.02% examples, 714942 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:18,615 : INFO : EPOCH 8 - PROGRESS: at 28.11% examples, 714544 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:19,625 : INFO : EPOCH 8 - PROGRESS: at 31.18% examples, 712909 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:20,634 : INFO : EPOCH 8 - PROGRESS: at 34.36% examples, 712468 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:21,651 : INFO : EPOCH 8 - PROGRESS: at 37.62% examples, 714104 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:22,662 : INFO : EPOCH 8 - PROGRESS: at 40.71% examples, 713678 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:23,676 : INFO : EPOCH 8 - PROGRESS: at 43.94% examples, 714489 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:24,690 : INFO : EPOCH 8 - PROGRESS: at 47.08% examples, 713963 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:49:25,692 : INFO : EPOCH 8 - PROGRESS: at 50.36% examples, 715845 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:26,712 : INFO : EPOCH 8 - PROGRESS: at 53.62% examples, 716728 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:27,720 : INFO : EPOCH 8 - PROGRESS: at 56.84% examples, 718051 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:28,723 : INFO : EPOCH 8 - PROGRESS: at 59.82% examples, 717243 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:29,742 : INFO : EPOCH 8 - PROGRESS: at 63.12% examples, 718031 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:30,768 : INFO : EPOCH 8 - PROGRESS: at 66.23% examples, 716591 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:31,776 : INFO : EPOCH 8 - PROGRESS: at 69.36% examples, 716839 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:32,778 : INFO : EPOCH 8 - PROGRESS: at 72.56% examples, 717637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:33,811 : INFO : EPOCH 8 - PROGRESS: at 75.81% examples, 718575 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:34,823 : INFO : EPOCH 8 - PROGRESS: at 79.07% examples, 718954 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:35,862 : INFO : EPOCH 8 - PROGRESS: at 82.30% examples, 718542 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:36,880 : INFO : EPOCH 8 - PROGRESS: at 85.38% examples, 717667 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:37,915 : INFO : EPOCH 8 - PROGRESS: at 88.40% examples, 715722 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:38,925 : INFO : EPOCH 8 - PROGRESS: at 91.54% examples, 715524 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:39,939 : INFO : EPOCH 8 - PROGRESS: at 94.56% examples, 713995 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:40,942 : INFO : EPOCH 8 - PROGRESS: at 97.83% examples, 715001 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:41,557 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:49:41,558 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:49:41,564 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:49:41,569 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:49:41,570 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:49:41,571 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:49:41,577 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:49:41,582 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:49:41,582 : INFO : EPOCH - 8 : training on 23279529 raw words (22951015 effective words) took 32.1s, 716082 effective words/s
    2020-09-30 22:49:42,599 : INFO : EPOCH 9 - PROGRESS: at 3.05% examples, 697016 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:43,600 : INFO : EPOCH 9 - PROGRESS: at 6.27% examples, 720746 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:44,618 : INFO : EPOCH 9 - PROGRESS: at 9.40% examples, 720241 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:45,644 : INFO : EPOCH 9 - PROGRESS: at 12.71% examples, 726004 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:46,659 : INFO : EPOCH 9 - PROGRESS: at 15.91% examples, 725848 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:47,683 : INFO : EPOCH 9 - PROGRESS: at 19.12% examples, 726525 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:49:48,700 : INFO : EPOCH 9 - PROGRESS: at 22.41% examples, 728896 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:49,728 : INFO : EPOCH 9 - PROGRESS: at 25.68% examples, 728237 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:50,730 : INFO : EPOCH 9 - PROGRESS: at 28.88% examples, 729664 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:51,739 : INFO : EPOCH 9 - PROGRESS: at 32.15% examples, 730401 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:49:52,758 : INFO : EPOCH 9 - PROGRESS: at 35.39% examples, 729524 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:53,785 : INFO : EPOCH 9 - PROGRESS: at 38.66% examples, 729307 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:54,796 : INFO : EPOCH 9 - PROGRESS: at 41.87% examples, 729150 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:55,817 : INFO : EPOCH 9 - PROGRESS: at 45.11% examples, 729228 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:56,825 : INFO : EPOCH 9 - PROGRESS: at 48.39% examples, 729806 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:57,832 : INFO : EPOCH 9 - PROGRESS: at 51.57% examples, 728729 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:58,841 : INFO : EPOCH 9 - PROGRESS: at 54.81% examples, 729387 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:49:59,843 : INFO : EPOCH 9 - PROGRESS: at 57.94% examples, 729703 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:00,851 : INFO : EPOCH 9 - PROGRESS: at 61.18% examples, 730641 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:01,851 : INFO : EPOCH 9 - PROGRESS: at 64.28% examples, 728955 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:02,869 : INFO : EPOCH 9 - PROGRESS: at 67.62% examples, 730020 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:03,879 : INFO : EPOCH 9 - PROGRESS: at 70.73% examples, 729156 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:04,880 : INFO : EPOCH 9 - PROGRESS: at 73.84% examples, 729426 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:05,888 : INFO : EPOCH 9 - PROGRESS: at 77.03% examples, 729418 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:06,907 : INFO : EPOCH 9 - PROGRESS: at 80.24% examples, 729133 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:07,940 : INFO : EPOCH 9 - PROGRESS: at 83.62% examples, 729637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:08,942 : INFO : EPOCH 9 - PROGRESS: at 86.83% examples, 729490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:09,947 : INFO : EPOCH 9 - PROGRESS: at 90.02% examples, 729523 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:10,953 : INFO : EPOCH 9 - PROGRESS: at 93.36% examples, 729997 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:11,962 : INFO : EPOCH 9 - PROGRESS: at 96.58% examples, 729761 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:12,950 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:50:12,956 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:50:12,961 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:50:12,962 : INFO : EPOCH 9 - PROGRESS: at 99.81% examples, 730237 words/s, in_qsize 4, out_qsize 1
    2020-09-30 22:50:12,962 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:50:12,966 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:50:12,972 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:50:12,977 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:50:12,979 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:50:12,979 : INFO : EPOCH - 9 : training on 23279529 raw words (22951015 effective words) took 31.4s, 731056 effective words/s
    2020-09-30 22:50:14,009 : INFO : EPOCH 10 - PROGRESS: at 2.95% examples, 668769 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:15,012 : INFO : EPOCH 10 - PROGRESS: at 6.05% examples, 691135 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:16,028 : INFO : EPOCH 10 - PROGRESS: at 9.07% examples, 691904 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:17,031 : INFO : EPOCH 10 - PROGRESS: at 12.33% examples, 706421 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:18,045 : INFO : EPOCH 10 - PROGRESS: at 15.49% examples, 708224 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:19,069 : INFO : EPOCH 10 - PROGRESS: at 18.69% examples, 711852 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:20,077 : INFO : EPOCH 10 - PROGRESS: at 21.70% examples, 709062 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:21,108 : INFO : EPOCH 10 - PROGRESS: at 25.02% examples, 710803 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:22,115 : INFO : EPOCH 10 - PROGRESS: at 28.20% examples, 712801 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:23,118 : INFO : EPOCH 10 - PROGRESS: at 31.56% examples, 718415 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:24,136 : INFO : EPOCH 10 - PROGRESS: at 34.72% examples, 716960 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:25,149 : INFO : EPOCH 10 - PROGRESS: at 38.08% examples, 720074 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:26,154 : INFO : EPOCH 10 - PROGRESS: at 41.17% examples, 719558 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:27,159 : INFO : EPOCH 10 - PROGRESS: at 44.48% examples, 721740 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:28,212 : INFO : EPOCH 10 - PROGRESS: at 47.69% examples, 720163 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:29,216 : INFO : EPOCH 10 - PROGRESS: at 50.92% examples, 720350 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:30,217 : INFO : EPOCH 10 - PROGRESS: at 54.01% examples, 719488 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:31,220 : INFO : EPOCH 10 - PROGRESS: at 57.22% examples, 720901 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:32,224 : INFO : EPOCH 10 - PROGRESS: at 60.28% examples, 720898 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:33,235 : INFO : EPOCH 10 - PROGRESS: at 63.60% examples, 721729 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:34,243 : INFO : EPOCH 10 - PROGRESS: at 66.82% examples, 722101 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:35,276 : INFO : EPOCH 10 - PROGRESS: at 70.07% examples, 722196 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:36,305 : INFO : EPOCH 10 - PROGRESS: at 73.16% examples, 721471 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:37,319 : INFO : EPOCH 10 - PROGRESS: at 76.28% examples, 721273 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:38,366 : INFO : EPOCH 10 - PROGRESS: at 79.23% examples, 717852 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:39,367 : INFO : EPOCH 10 - PROGRESS: at 82.16% examples, 715942 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:40,367 : INFO : EPOCH 10 - PROGRESS: at 85.10% examples, 714184 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:41,374 : INFO : EPOCH 10 - PROGRESS: at 88.27% examples, 714483 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:42,389 : INFO : EPOCH 10 - PROGRESS: at 91.39% examples, 713874 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:43,393 : INFO : EPOCH 10 - PROGRESS: at 94.60% examples, 714219 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:44,398 : INFO : EPOCH 10 - PROGRESS: at 97.88% examples, 715161 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:44,985 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:50:44,992 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:50:44,995 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:50:44,996 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:50:45,000 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:50:45,001 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:50:45,016 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:50:45,020 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:50:45,020 : INFO : EPOCH - 10 : training on 23279529 raw words (22951015 effective words) took 32.0s, 716354 effective words/s
    2020-09-30 22:50:46,039 : INFO : EPOCH 11 - PROGRESS: at 2.91% examples, 666098 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:47,069 : INFO : EPOCH 11 - PROGRESS: at 6.05% examples, 685636 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:48,070 : INFO : EPOCH 11 - PROGRESS: at 9.07% examples, 691604 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:49,071 : INFO : EPOCH 11 - PROGRESS: at 12.15% examples, 696932 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:50,074 : INFO : EPOCH 11 - PROGRESS: at 15.17% examples, 694565 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:51,091 : INFO : EPOCH 11 - PROGRESS: at 18.36% examples, 701278 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:52,103 : INFO : EPOCH 11 - PROGRESS: at 21.58% examples, 706333 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:53,107 : INFO : EPOCH 11 - PROGRESS: at 24.87% examples, 709850 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:54,117 : INFO : EPOCH 11 - PROGRESS: at 27.87% examples, 707297 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:50:55,135 : INFO : EPOCH 11 - PROGRESS: at 31.03% examples, 708640 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:56,162 : INFO : EPOCH 11 - PROGRESS: at 34.28% examples, 708296 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:57,164 : INFO : EPOCH 11 - PROGRESS: at 37.41% examples, 708820 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:50:58,175 : INFO : EPOCH 11 - PROGRESS: at 40.60% examples, 710242 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:50:59,188 : INFO : EPOCH 11 - PROGRESS: at 43.86% examples, 712081 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:00,192 : INFO : EPOCH 11 - PROGRESS: at 47.03% examples, 712782 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:01,222 : INFO : EPOCH 11 - PROGRESS: at 50.36% examples, 714106 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:02,233 : INFO : EPOCH 11 - PROGRESS: at 53.43% examples, 712696 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:03,254 : INFO : EPOCH 11 - PROGRESS: at 56.60% examples, 713177 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:04,276 : INFO : EPOCH 11 - PROGRESS: at 59.60% examples, 712418 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:05,321 : INFO : EPOCH 11 - PROGRESS: at 62.76% examples, 711110 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:06,331 : INFO : EPOCH 11 - PROGRESS: at 65.93% examples, 711018 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:07,354 : INFO : EPOCH 11 - PROGRESS: at 69.06% examples, 710543 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:08,372 : INFO : EPOCH 11 - PROGRESS: at 72.18% examples, 710742 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:51:09,382 : INFO : EPOCH 11 - PROGRESS: at 75.31% examples, 711452 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:10,386 : INFO : EPOCH 11 - PROGRESS: at 78.52% examples, 711934 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:11,392 : INFO : EPOCH 11 - PROGRESS: at 81.78% examples, 713080 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:12,393 : INFO : EPOCH 11 - PROGRESS: at 84.87% examples, 712802 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:13,444 : INFO : EPOCH 11 - PROGRESS: at 88.05% examples, 712044 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:51:14,448 : INFO : EPOCH 11 - PROGRESS: at 91.26% examples, 712443 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:15,452 : INFO : EPOCH 11 - PROGRESS: at 94.37% examples, 712194 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:16,452 : INFO : EPOCH 11 - PROGRESS: at 97.47% examples, 711794 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:17,188 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:51:17,197 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:51:17,198 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:51:17,200 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:51:17,202 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:51:17,207 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:51:17,218 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:51:17,219 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:51:17,219 : INFO : EPOCH - 11 : training on 23279529 raw words (22951015 effective words) took 32.2s, 712836 effective words/s
    2020-09-30 22:51:18,223 : INFO : EPOCH 12 - PROGRESS: at 2.95% examples, 686168 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:51:19,224 : INFO : EPOCH 12 - PROGRESS: at 6.05% examples, 700997 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:51:20,266 : INFO : EPOCH 12 - PROGRESS: at 9.37% examples, 714515 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:21,272 : INFO : EPOCH 12 - PROGRESS: at 12.54% examples, 718077 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:22,284 : INFO : EPOCH 12 - PROGRESS: at 15.62% examples, 714103 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:23,334 : INFO : EPOCH 12 - PROGRESS: at 18.76% examples, 712133 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:24,343 : INFO : EPOCH 12 - PROGRESS: at 21.97% examples, 714635 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:25,349 : INFO : EPOCH 12 - PROGRESS: at 25.02% examples, 710774 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:26,357 : INFO : EPOCH 12 - PROGRESS: at 28.15% examples, 711565 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:27,366 : INFO : EPOCH 12 - PROGRESS: at 31.26% examples, 711249 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:28,391 : INFO : EPOCH 12 - PROGRESS: at 34.40% examples, 709010 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:29,412 : INFO : EPOCH 12 - PROGRESS: at 37.63% examples, 709941 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:30,428 : INFO : EPOCH 12 - PROGRESS: at 40.71% examples, 709561 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:51:31,433 : INFO : EPOCH 12 - PROGRESS: at 43.94% examples, 711113 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:32,441 : INFO : EPOCH 12 - PROGRESS: at 47.08% examples, 711067 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:33,465 : INFO : EPOCH 12 - PROGRESS: at 50.44% examples, 713361 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:34,467 : INFO : EPOCH 12 - PROGRESS: at 53.59% examples, 713490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:35,482 : INFO : EPOCH 12 - PROGRESS: at 56.51% examples, 710984 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:36,493 : INFO : EPOCH 12 - PROGRESS: at 59.53% examples, 710779 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:37,501 : INFO : EPOCH 12 - PROGRESS: at 62.72% examples, 711302 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:38,518 : INFO : EPOCH 12 - PROGRESS: at 65.81% examples, 710086 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:39,522 : INFO : EPOCH 12 - PROGRESS: at 68.86% examples, 709389 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:40,525 : INFO : EPOCH 12 - PROGRESS: at 71.95% examples, 709628 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:41,527 : INFO : EPOCH 12 - PROGRESS: at 75.02% examples, 710212 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:42,537 : INFO : EPOCH 12 - PROGRESS: at 78.14% examples, 709861 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:43,569 : INFO : EPOCH 12 - PROGRESS: at 81.26% examples, 709262 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:51:44,584 : INFO : EPOCH 12 - PROGRESS: at 84.54% examples, 710225 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:45,589 : INFO : EPOCH 12 - PROGRESS: at 87.72% examples, 710637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:46,605 : INFO : EPOCH 12 - PROGRESS: at 90.99% examples, 711478 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:47,625 : INFO : EPOCH 12 - PROGRESS: at 94.11% examples, 710887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:48,628 : INFO : EPOCH 12 - PROGRESS: at 97.26% examples, 710768 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:49,470 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:51:49,473 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:51:49,477 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:51:49,477 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:51:49,482 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:51:49,485 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:51:49,502 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:51:49,502 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:51:49,502 : INFO : EPOCH - 12 : training on 23279529 raw words (22951015 effective words) took 32.3s, 710980 effective words/s
    2020-09-30 22:51:50,513 : INFO : EPOCH 13 - PROGRESS: at 3.00% examples, 690974 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:51,515 : INFO : EPOCH 13 - PROGRESS: at 6.01% examples, 693466 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:52,529 : INFO : EPOCH 13 - PROGRESS: at 9.03% examples, 693700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:53,532 : INFO : EPOCH 13 - PROGRESS: at 12.11% examples, 698318 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:54,557 : INFO : EPOCH 13 - PROGRESS: at 15.25% examples, 698352 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:55,563 : INFO : EPOCH 13 - PROGRESS: at 18.36% examples, 702485 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:56,580 : INFO : EPOCH 13 - PROGRESS: at 21.42% examples, 701455 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:51:57,597 : INFO : EPOCH 13 - PROGRESS: at 24.66% examples, 703133 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:58,610 : INFO : EPOCH 13 - PROGRESS: at 27.69% examples, 702249 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:51:59,625 : INFO : EPOCH 13 - PROGRESS: at 30.81% examples, 703266 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:00,651 : INFO : EPOCH 13 - PROGRESS: at 33.99% examples, 701851 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:01,654 : INFO : EPOCH 13 - PROGRESS: at 37.07% examples, 701969 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:02,667 : INFO : EPOCH 13 - PROGRESS: at 40.24% examples, 703241 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:03,674 : INFO : EPOCH 13 - PROGRESS: at 43.31% examples, 703059 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:04,674 : INFO : EPOCH 13 - PROGRESS: at 46.42% examples, 703201 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:05,684 : INFO : EPOCH 13 - PROGRESS: at 49.47% examples, 703007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:06,685 : INFO : EPOCH 13 - PROGRESS: at 52.62% examples, 703234 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:07,704 : INFO : EPOCH 13 - PROGRESS: at 55.67% examples, 702742 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:08,717 : INFO : EPOCH 13 - PROGRESS: at 58.73% examples, 702985 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:09,739 : INFO : EPOCH 13 - PROGRESS: at 61.88% examples, 703282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:10,740 : INFO : EPOCH 13 - PROGRESS: at 65.09% examples, 704365 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:11,747 : INFO : EPOCH 13 - PROGRESS: at 68.03% examples, 702941 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:12,754 : INFO : EPOCH 13 - PROGRESS: at 71.17% examples, 703762 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:13,779 : INFO : EPOCH 13 - PROGRESS: at 74.14% examples, 702789 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:14,784 : INFO : EPOCH 13 - PROGRESS: at 77.33% examples, 703937 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:15,791 : INFO : EPOCH 13 - PROGRESS: at 80.25% examples, 702395 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:16,794 : INFO : EPOCH 13 - PROGRESS: at 83.53% examples, 703934 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:17,796 : INFO : EPOCH 13 - PROGRESS: at 86.59% examples, 703336 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:18,811 : INFO : EPOCH 13 - PROGRESS: at 89.70% examples, 703382 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:19,816 : INFO : EPOCH 13 - PROGRESS: at 92.76% examples, 702807 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:20,820 : INFO : EPOCH 13 - PROGRESS: at 95.91% examples, 702922 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:21,860 : INFO : EPOCH 13 - PROGRESS: at 98.97% examples, 702191 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:22,084 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:52:22,091 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:52:22,095 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:52:22,096 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:52:22,105 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:52:22,106 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:52:22,117 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:52:22,119 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:52:22,119 : INFO : EPOCH - 13 : training on 23279529 raw words (22951015 effective words) took 32.6s, 703714 effective words/s
    2020-09-30 22:52:23,134 : INFO : EPOCH 14 - PROGRESS: at 3.00% examples, 687998 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:24,138 : INFO : EPOCH 14 - PROGRESS: at 6.01% examples, 691254 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:25,150 : INFO : EPOCH 14 - PROGRESS: at 9.07% examples, 696072 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:26,160 : INFO : EPOCH 14 - PROGRESS: at 12.15% examples, 698753 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:27,160 : INFO : EPOCH 14 - PROGRESS: at 15.34% examples, 704003 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:28,208 : INFO : EPOCH 14 - PROGRESS: at 18.45% examples, 702261 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:29,248 : INFO : EPOCH 14 - PROGRESS: at 21.67% examples, 704569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:30,264 : INFO : EPOCH 14 - PROGRESS: at 24.71% examples, 699976 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:31,266 : INFO : EPOCH 14 - PROGRESS: at 27.73% examples, 700235 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:32,278 : INFO : EPOCH 14 - PROGRESS: at 30.81% examples, 700797 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:33,288 : INFO : EPOCH 14 - PROGRESS: at 33.99% examples, 700523 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:34,317 : INFO : EPOCH 14 - PROGRESS: at 37.07% examples, 699314 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:35,317 : INFO : EPOCH 14 - PROGRESS: at 40.24% examples, 701489 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:36,324 : INFO : EPOCH 14 - PROGRESS: at 43.26% examples, 700715 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:37,331 : INFO : EPOCH 14 - PROGRESS: at 46.38% examples, 700724 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:38,337 : INFO : EPOCH 14 - PROGRESS: at 49.42% examples, 700846 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:39,351 : INFO : EPOCH 14 - PROGRESS: at 52.54% examples, 700071 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:40,353 : INFO : EPOCH 14 - PROGRESS: at 55.52% examples, 699379 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:41,358 : INFO : EPOCH 14 - PROGRESS: at 58.60% examples, 700625 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:42,366 : INFO : EPOCH 14 - PROGRESS: at 61.67% examples, 700513 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:43,387 : INFO : EPOCH 14 - PROGRESS: at 64.84% examples, 700637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:44,388 : INFO : EPOCH 14 - PROGRESS: at 67.91% examples, 700850 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:45,402 : INFO : EPOCH 14 - PROGRESS: at 71.01% examples, 701189 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:46,422 : INFO : EPOCH 14 - PROGRESS: at 73.84% examples, 699248 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:47,423 : INFO : EPOCH 14 - PROGRESS: at 76.86% examples, 699128 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:48,424 : INFO : EPOCH 14 - PROGRESS: at 79.96% examples, 699371 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:49,428 : INFO : EPOCH 14 - PROGRESS: at 83.09% examples, 699933 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:50,433 : INFO : EPOCH 14 - PROGRESS: at 86.12% examples, 699076 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:51,437 : INFO : EPOCH 14 - PROGRESS: at 89.28% examples, 699865 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:52,448 : INFO : EPOCH 14 - PROGRESS: at 92.42% examples, 699893 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:52:53,458 : INFO : EPOCH 14 - PROGRESS: at 95.56% examples, 699964 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:54,480 : INFO : EPOCH 14 - PROGRESS: at 98.58% examples, 699418 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:54,882 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:52:54,891 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:52:54,898 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:52:54,900 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:52:54,901 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:52:54,902 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:52:54,916 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:52:54,922 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:52:54,923 : INFO : EPOCH - 14 : training on 23279529 raw words (22951015 effective words) took 32.8s, 699694 effective words/s
    2020-09-30 22:52:55,931 : INFO : EPOCH 15 - PROGRESS: at 2.70% examples, 624746 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:56,943 : INFO : EPOCH 15 - PROGRESS: at 5.45% examples, 628334 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:57,967 : INFO : EPOCH 15 - PROGRESS: at 8.55% examples, 651732 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:58,978 : INFO : EPOCH 15 - PROGRESS: at 11.64% examples, 667759 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:52:59,985 : INFO : EPOCH 15 - PROGRESS: at 14.74% examples, 674127 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:00,990 : INFO : EPOCH 15 - PROGRESS: at 17.80% examples, 680707 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:01,993 : INFO : EPOCH 15 - PROGRESS: at 20.98% examples, 688603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:02,996 : INFO : EPOCH 15 - PROGRESS: at 24.06% examples, 688036 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:04,001 : INFO : EPOCH 15 - PROGRESS: at 27.16% examples, 690733 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:05,020 : INFO : EPOCH 15 - PROGRESS: at 30.19% examples, 690764 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:06,030 : INFO : EPOCH 15 - PROGRESS: at 33.33% examples, 691330 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:07,032 : INFO : EPOCH 15 - PROGRESS: at 36.48% examples, 693240 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:08,036 : INFO : EPOCH 15 - PROGRESS: at 39.59% examples, 694884 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:09,062 : INFO : EPOCH 15 - PROGRESS: at 42.53% examples, 692406 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:10,066 : INFO : EPOCH 15 - PROGRESS: at 45.61% examples, 693083 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:11,095 : INFO : EPOCH 15 - PROGRESS: at 48.71% examples, 692625 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:12,107 : INFO : EPOCH 15 - PROGRESS: at 51.83% examples, 692500 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:13,108 : INFO : EPOCH 15 - PROGRESS: at 54.89% examples, 693309 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:14,115 : INFO : EPOCH 15 - PROGRESS: at 57.94% examples, 694263 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:15,126 : INFO : EPOCH 15 - PROGRESS: at 61.05% examples, 695396 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:16,133 : INFO : EPOCH 15 - PROGRESS: at 64.14% examples, 695209 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:17,138 : INFO : EPOCH 15 - PROGRESS: at 67.29% examples, 695978 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:18,169 : INFO : EPOCH 15 - PROGRESS: at 70.40% examples, 696027 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:53:19,170 : INFO : EPOCH 15 - PROGRESS: at 73.48% examples, 697257 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:20,178 : INFO : EPOCH 15 - PROGRESS: at 76.53% examples, 697432 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:21,179 : INFO : EPOCH 15 - PROGRESS: at 79.69% examples, 698130 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:22,180 : INFO : EPOCH 15 - PROGRESS: at 82.72% examples, 697694 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:23,191 : INFO : EPOCH 15 - PROGRESS: at 85.84% examples, 698154 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:24,216 : INFO : EPOCH 15 - PROGRESS: at 88.74% examples, 696161 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:25,216 : INFO : EPOCH 15 - PROGRESS: at 91.72% examples, 695616 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:26,222 : INFO : EPOCH 15 - PROGRESS: at 94.87% examples, 695870 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:27,231 : INFO : EPOCH 15 - PROGRESS: at 98.00% examples, 696383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:27,812 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:53:27,816 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:53:27,817 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:53:27,822 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:53:27,825 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:53:27,826 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:53:27,844 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:53:27,846 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:53:27,846 : INFO : EPOCH - 15 : training on 23279529 raw words (22951015 effective words) took 32.9s, 697158 effective words/s
    2020-09-30 22:53:28,860 : INFO : EPOCH 16 - PROGRESS: at 3.00% examples, 688568 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:29,893 : INFO : EPOCH 16 - PROGRESS: at 6.05% examples, 686386 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:30,904 : INFO : EPOCH 16 - PROGRESS: at 9.20% examples, 699415 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:31,916 : INFO : EPOCH 16 - PROGRESS: at 12.28% examples, 700809 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:32,919 : INFO : EPOCH 16 - PROGRESS: at 15.41% examples, 703469 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:33,951 : INFO : EPOCH 16 - PROGRESS: at 18.45% examples, 700564 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:34,951 : INFO : EPOCH 16 - PROGRESS: at 21.39% examples, 697439 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:35,973 : INFO : EPOCH 16 - PROGRESS: at 24.40% examples, 693131 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:36,975 : INFO : EPOCH 16 - PROGRESS: at 27.52% examples, 696324 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:37,989 : INFO : EPOCH 16 - PROGRESS: at 30.57% examples, 696226 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:39,005 : INFO : EPOCH 16 - PROGRESS: at 33.66% examples, 695067 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:40,013 : INFO : EPOCH 16 - PROGRESS: at 36.74% examples, 694697 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:41,015 : INFO : EPOCH 16 - PROGRESS: at 39.89% examples, 697111 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:42,023 : INFO : EPOCH 16 - PROGRESS: at 42.88% examples, 696028 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:43,031 : INFO : EPOCH 16 - PROGRESS: at 46.03% examples, 696839 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:44,039 : INFO : EPOCH 16 - PROGRESS: at 49.09% examples, 697138 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:45,040 : INFO : EPOCH 16 - PROGRESS: at 52.15% examples, 696579 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:46,051 : INFO : EPOCH 16 - PROGRESS: at 55.20% examples, 696208 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:47,053 : INFO : EPOCH 16 - PROGRESS: at 58.28% examples, 697749 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:48,056 : INFO : EPOCH 16 - PROGRESS: at 61.32% examples, 698001 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:49,088 : INFO : EPOCH 16 - PROGRESS: at 64.50% examples, 697831 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:50,089 : INFO : EPOCH 16 - PROGRESS: at 67.54% examples, 697717 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:51,092 : INFO : EPOCH 16 - PROGRESS: at 70.61% examples, 698119 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:52,094 : INFO : EPOCH 16 - PROGRESS: at 73.57% examples, 698009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:53,096 : INFO : EPOCH 16 - PROGRESS: at 76.74% examples, 699459 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:54,115 : INFO : EPOCH 16 - PROGRESS: at 79.76% examples, 698536 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:53:55,129 : INFO : EPOCH 16 - PROGRESS: at 82.80% examples, 697744 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:56,133 : INFO : EPOCH 16 - PROGRESS: at 85.89% examples, 698027 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:57,177 : INFO : EPOCH 16 - PROGRESS: at 89.05% examples, 697897 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:53:58,201 : INFO : EPOCH 16 - PROGRESS: at 92.27% examples, 698340 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:53:59,207 : INFO : EPOCH 16 - PROGRESS: at 95.49% examples, 698856 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:00,228 : INFO : EPOCH 16 - PROGRESS: at 98.58% examples, 698977 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:00,602 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:54:00,609 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:54:00,610 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:54:00,611 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:54:00,618 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:54:00,620 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:54:00,633 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:54:00,636 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:54:00,637 : INFO : EPOCH - 16 : training on 23279529 raw words (22951015 effective words) took 32.8s, 699972 effective words/s
    2020-09-30 22:54:01,641 : INFO : EPOCH 17 - PROGRESS: at 2.87% examples, 666568 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:02,647 : INFO : EPOCH 17 - PROGRESS: at 6.01% examples, 694213 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:03,661 : INFO : EPOCH 17 - PROGRESS: at 9.03% examples, 694199 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:04,688 : INFO : EPOCH 17 - PROGRESS: at 12.02% examples, 689782 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:05,703 : INFO : EPOCH 17 - PROGRESS: at 15.09% examples, 689026 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:06,726 : INFO : EPOCH 17 - PROGRESS: at 18.09% examples, 689490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:07,728 : INFO : EPOCH 17 - PROGRESS: at 21.19% examples, 693268 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:08,734 : INFO : EPOCH 17 - PROGRESS: at 24.32% examples, 693258 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:09,747 : INFO : EPOCH 17 - PROGRESS: at 27.41% examples, 694675 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:10,762 : INFO : EPOCH 17 - PROGRESS: at 30.57% examples, 697416 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:11,772 : INFO : EPOCH 17 - PROGRESS: at 33.66% examples, 696529 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:12,775 : INFO : EPOCH 17 - PROGRESS: at 36.78% examples, 697202 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:13,782 : INFO : EPOCH 17 - PROGRESS: at 39.72% examples, 695407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:14,806 : INFO : EPOCH 17 - PROGRESS: at 42.88% examples, 696381 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:15,832 : INFO : EPOCH 17 - PROGRESS: at 46.03% examples, 696381 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:16,834 : INFO : EPOCH 17 - PROGRESS: at 49.13% examples, 697554 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:17,851 : INFO : EPOCH 17 - PROGRESS: at 52.27% examples, 697422 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:18,857 : INFO : EPOCH 17 - PROGRESS: at 55.45% examples, 698825 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:19,879 : INFO : EPOCH 17 - PROGRESS: at 58.39% examples, 697953 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:20,886 : INFO : EPOCH 17 - PROGRESS: at 61.41% examples, 697618 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:21,928 : INFO : EPOCH 17 - PROGRESS: at 64.50% examples, 696238 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:22,957 : INFO : EPOCH 17 - PROGRESS: at 67.70% examples, 697065 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:23,992 : INFO : EPOCH 17 - PROGRESS: at 70.85% examples, 697374 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:24,998 : INFO : EPOCH 17 - PROGRESS: at 73.92% examples, 698364 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:26,003 : INFO : EPOCH 17 - PROGRESS: at 77.07% examples, 699300 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:27,009 : INFO : EPOCH 17 - PROGRESS: at 80.17% examples, 699424 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:28,023 : INFO : EPOCH 17 - PROGRESS: at 83.26% examples, 699399 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:29,056 : INFO : EPOCH 17 - PROGRESS: at 86.38% examples, 698549 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:30,073 : INFO : EPOCH 17 - PROGRESS: at 89.41% examples, 698041 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:31,083 : INFO : EPOCH 17 - PROGRESS: at 92.60% examples, 698495 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:32,095 : INFO : EPOCH 17 - PROGRESS: at 95.70% examples, 698237 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:33,111 : INFO : EPOCH 17 - PROGRESS: at 98.92% examples, 699364 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:33,380 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:54:33,390 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:54:33,399 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:54:33,400 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:54:33,404 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:54:33,404 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:54:33,420 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:54:33,423 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:54:33,423 : INFO : EPOCH - 17 : training on 23279529 raw words (22951015 effective words) took 32.8s, 700071 effective words/s
    2020-09-30 22:54:34,439 : INFO : EPOCH 18 - PROGRESS: at 2.79% examples, 639168 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:35,450 : INFO : EPOCH 18 - PROGRESS: at 5.92% examples, 679215 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:36,473 : INFO : EPOCH 18 - PROGRESS: at 8.89% examples, 675879 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:37,487 : INFO : EPOCH 18 - PROGRESS: at 11.68% examples, 668776 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:38,539 : INFO : EPOCH 18 - PROGRESS: at 14.41% examples, 652013 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:39,561 : INFO : EPOCH 18 - PROGRESS: at 17.11% examples, 647596 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:40,576 : INFO : EPOCH 18 - PROGRESS: at 19.80% examples, 642547 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:54:41,589 : INFO : EPOCH 18 - PROGRESS: at 22.66% examples, 642339 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:42,611 : INFO : EPOCH 18 - PROGRESS: at 25.68% examples, 645557 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:43,616 : INFO : EPOCH 18 - PROGRESS: at 28.73% examples, 650996 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:44,631 : INFO : EPOCH 18 - PROGRESS: at 31.77% examples, 654206 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:45,634 : INFO : EPOCH 18 - PROGRESS: at 34.89% examples, 658246 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:46,639 : INFO : EPOCH 18 - PROGRESS: at 37.98% examples, 661594 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:47,660 : INFO : EPOCH 18 - PROGRESS: at 41.00% examples, 663091 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:54:48,670 : INFO : EPOCH 18 - PROGRESS: at 43.64% examples, 658543 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:49,677 : INFO : EPOCH 18 - PROGRESS: at 46.38% examples, 655791 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:50,693 : INFO : EPOCH 18 - PROGRESS: at 49.05% examples, 653096 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:51,705 : INFO : EPOCH 18 - PROGRESS: at 51.71% examples, 649329 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:52,714 : INFO : EPOCH 18 - PROGRESS: at 54.25% examples, 645947 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:53,718 : INFO : EPOCH 18 - PROGRESS: at 56.84% examples, 643619 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:54,719 : INFO : EPOCH 18 - PROGRESS: at 59.53% examples, 643289 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:54:55,720 : INFO : EPOCH 18 - PROGRESS: at 62.18% examples, 641354 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:56,725 : INFO : EPOCH 18 - PROGRESS: at 64.80% examples, 639067 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:57,739 : INFO : EPOCH 18 - PROGRESS: at 67.62% examples, 639048 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:58,743 : INFO : EPOCH 18 - PROGRESS: at 70.69% examples, 641699 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:54:59,750 : INFO : EPOCH 18 - PROGRESS: at 73.53% examples, 642519 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:00,763 : INFO : EPOCH 18 - PROGRESS: at 76.74% examples, 645988 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:01,805 : INFO : EPOCH 18 - PROGRESS: at 79.84% examples, 647184 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:02,814 : INFO : EPOCH 18 - PROGRESS: at 82.83% examples, 648039 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:03,814 : INFO : EPOCH 18 - PROGRESS: at 85.38% examples, 645866 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:04,825 : INFO : EPOCH 18 - PROGRESS: at 88.18% examples, 645442 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:05,830 : INFO : EPOCH 18 - PROGRESS: at 90.64% examples, 642736 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:06,864 : INFO : EPOCH 18 - PROGRESS: at 93.53% examples, 642287 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:07,895 : INFO : EPOCH 18 - PROGRESS: at 96.58% examples, 643113 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:08,902 : INFO : EPOCH 18 - PROGRESS: at 99.65% examples, 644799 words/s, in_qsize 8, out_qsize 0
    2020-09-30 22:55:08,921 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:55:08,926 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:55:08,927 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:55:08,927 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:55:08,941 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:55:08,942 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:55:08,950 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:55:08,955 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:55:08,955 : INFO : EPOCH - 18 : training on 23279529 raw words (22951015 effective words) took 35.5s, 645962 effective words/s
    2020-09-30 22:55:09,973 : INFO : EPOCH 19 - PROGRESS: at 2.92% examples, 667894 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:11,004 : INFO : EPOCH 19 - PROGRESS: at 6.05% examples, 686248 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:12,010 : INFO : EPOCH 19 - PROGRESS: at 9.07% examples, 690883 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:13,018 : INFO : EPOCH 19 - PROGRESS: at 12.24% examples, 700009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:14,020 : INFO : EPOCH 19 - PROGRESS: at 15.34% examples, 700928 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:15,024 : INFO : EPOCH 19 - PROGRESS: at 18.36% examples, 701698 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:16,029 : INFO : EPOCH 19 - PROGRESS: at 21.11% examples, 692465 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:17,036 : INFO : EPOCH 19 - PROGRESS: at 24.18% examples, 691167 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:18,041 : INFO : EPOCH 19 - PROGRESS: at 27.33% examples, 694584 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:19,065 : INFO : EPOCH 19 - PROGRESS: at 30.53% examples, 697638 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:20,072 : INFO : EPOCH 19 - PROGRESS: at 33.62% examples, 696896 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:21,083 : INFO : EPOCH 19 - PROGRESS: at 36.74% examples, 697079 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:22,087 : INFO : EPOCH 19 - PROGRESS: at 39.84% examples, 698456 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:23,098 : INFO : EPOCH 19 - PROGRESS: at 42.92% examples, 698485 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:24,117 : INFO : EPOCH 19 - PROGRESS: at 46.03% examples, 697997 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:25,129 : INFO : EPOCH 19 - PROGRESS: at 49.09% examples, 698011 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:26,133 : INFO : EPOCH 19 - PROGRESS: at 52.15% examples, 697298 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:27,136 : INFO : EPOCH 19 - PROGRESS: at 55.24% examples, 697716 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:55:28,149 : INFO : EPOCH 19 - PROGRESS: at 58.31% examples, 698816 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:29,151 : INFO : EPOCH 19 - PROGRESS: at 61.41% examples, 699481 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:30,168 : INFO : EPOCH 19 - PROGRESS: at 64.50% examples, 698831 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:31,177 : INFO : EPOCH 19 - PROGRESS: at 67.58% examples, 698901 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:32,182 : INFO : EPOCH 19 - PROGRESS: at 70.66% examples, 699172 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:33,185 : INFO : EPOCH 19 - PROGRESS: at 73.77% examples, 700597 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:34,207 : INFO : EPOCH 19 - PROGRESS: at 76.78% examples, 699842 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:35,233 : INFO : EPOCH 19 - PROGRESS: at 79.88% examples, 699410 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:36,235 : INFO : EPOCH 19 - PROGRESS: at 82.94% examples, 699296 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:37,251 : INFO : EPOCH 19 - PROGRESS: at 86.07% examples, 699232 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:38,258 : INFO : EPOCH 19 - PROGRESS: at 89.05% examples, 698619 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:39,259 : INFO : EPOCH 19 - PROGRESS: at 92.06% examples, 697976 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:40,276 : INFO : EPOCH 19 - PROGRESS: at 95.17% examples, 697601 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:41,289 : INFO : EPOCH 19 - PROGRESS: at 98.30% examples, 697944 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:41,757 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:55:41,759 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:55:41,765 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:55:41,766 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:55:41,768 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:55:41,773 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:55:41,787 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:55:41,791 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:55:41,791 : INFO : EPOCH - 19 : training on 23279529 raw words (22951015 effective words) took 32.8s, 699046 effective words/s
    2020-09-30 22:55:42,798 : INFO : EPOCH 20 - PROGRESS: at 3.00% examples, 693893 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:43,819 : INFO : EPOCH 20 - PROGRESS: at 6.04% examples, 692915 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:44,823 : INFO : EPOCH 20 - PROGRESS: at 9.20% examples, 705231 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:45,829 : INFO : EPOCH 20 - PROGRESS: at 12.15% examples, 699174 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:46,831 : INFO : EPOCH 20 - PROGRESS: at 15.21% examples, 698414 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:47,836 : INFO : EPOCH 20 - PROGRESS: at 18.23% examples, 699395 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:48,847 : INFO : EPOCH 20 - PROGRESS: at 21.34% examples, 700869 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:49,854 : INFO : EPOCH 20 - PROGRESS: at 24.45% examples, 699767 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:50,861 : INFO : EPOCH 20 - PROGRESS: at 27.57% examples, 701948 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:51,863 : INFO : EPOCH 20 - PROGRESS: at 30.61% examples, 702048 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:52,863 : INFO : EPOCH 20 - PROGRESS: at 33.81% examples, 703137 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:53,871 : INFO : EPOCH 20 - PROGRESS: at 36.87% examples, 702140 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:54,889 : INFO : EPOCH 20 - PROGRESS: at 39.89% examples, 700894 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:55,899 : INFO : EPOCH 20 - PROGRESS: at 42.84% examples, 698704 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:56,945 : INFO : EPOCH 20 - PROGRESS: at 46.03% examples, 698265 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:57,951 : INFO : EPOCH 20 - PROGRESS: at 49.09% examples, 698564 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:55:58,970 : INFO : EPOCH 20 - PROGRESS: at 52.27% examples, 698875 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:55:59,971 : INFO : EPOCH 20 - PROGRESS: at 55.41% examples, 699848 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:00,972 : INFO : EPOCH 20 - PROGRESS: at 58.48% examples, 701197 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:01,979 : INFO : EPOCH 20 - PROGRESS: at 61.46% examples, 700180 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:02,996 : INFO : EPOCH 20 - PROGRESS: at 64.50% examples, 699042 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:04,041 : INFO : EPOCH 20 - PROGRESS: at 67.70% examples, 699260 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:05,055 : INFO : EPOCH 20 - PROGRESS: at 70.73% examples, 698822 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:06,077 : INFO : EPOCH 20 - PROGRESS: at 73.81% examples, 699319 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:07,079 : INFO : EPOCH 20 - PROGRESS: at 76.86% examples, 699554 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:08,117 : INFO : EPOCH 20 - PROGRESS: at 79.88% examples, 698096 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:09,133 : INFO : EPOCH 20 - PROGRESS: at 82.94% examples, 697667 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:10,153 : INFO : EPOCH 20 - PROGRESS: at 86.07% examples, 697557 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:11,154 : INFO : EPOCH 20 - PROGRESS: at 89.23% examples, 698457 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:12,156 : INFO : EPOCH 20 - PROGRESS: at 92.27% examples, 698116 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:13,175 : INFO : EPOCH 20 - PROGRESS: at 95.30% examples, 697082 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:14,180 : INFO : EPOCH 20 - PROGRESS: at 98.25% examples, 696440 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:14,687 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:56:14,702 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:56:14,703 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:56:14,708 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:56:14,712 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:56:14,714 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:56:14,727 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:56:14,730 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:56:14,731 : INFO : EPOCH - 20 : training on 23279529 raw words (22951015 effective words) took 32.9s, 696805 effective words/s
    2020-09-30 22:56:14,731 : INFO : training on a 465590580 raw words (459020300 effective words) took 667.1s, 688091 effective words/s

    Evaluating Doc2Vec(dm/m,d100,n5,w10,mc2,t8)

    0.168400 Doc2Vec(dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    2020-09-30 22:56:15,472 : INFO : training model with 8 workers on 265409 vocabulary and 1100 features, using sg=0 hs=0 sample=0 negative=5 window=5
    2020-09-30 22:56:16,514 : INFO : EPOCH 1 - PROGRESS: at 0.37% examples, 83752 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:17,544 : INFO : EPOCH 1 - PROGRESS: at 1.94% examples, 219284 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:18,607 : INFO : EPOCH 1 - PROGRESS: at 3.74% examples, 278086 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:19,646 : INFO : EPOCH 1 - PROGRESS: at 5.50% examples, 306209 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:20,668 : INFO : EPOCH 1 - PROGRESS: at 7.31% examples, 326068 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:21,674 : INFO : EPOCH 1 - PROGRESS: at 9.20% examples, 344658 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:22,690 : INFO : EPOCH 1 - PROGRESS: at 10.90% examples, 352182 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:23,721 : INFO : EPOCH 1 - PROGRESS: at 12.76% examples, 358528 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:24,744 : INFO : EPOCH 1 - PROGRESS: at 14.66% examples, 365839 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:25,780 : INFO : EPOCH 1 - PROGRESS: at 16.37% examples, 367683 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:56:26,783 : INFO : EPOCH 1 - PROGRESS: at 18.18% examples, 372860 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:27,789 : INFO : EPOCH 1 - PROGRESS: at 20.03% examples, 377763 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:28,838 : INFO : EPOCH 1 - PROGRESS: at 21.97% examples, 380846 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:29,882 : INFO : EPOCH 1 - PROGRESS: at 24.02% examples, 384766 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:30,904 : INFO : EPOCH 1 - PROGRESS: at 25.93% examples, 388059 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:31,907 : INFO : EPOCH 1 - PROGRESS: at 27.82% examples, 390873 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:32,964 : INFO : EPOCH 1 - PROGRESS: at 29.86% examples, 394288 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:33,975 : INFO : EPOCH 1 - PROGRESS: at 31.86% examples, 397284 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:35,005 : INFO : EPOCH 1 - PROGRESS: at 33.66% examples, 397027 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:36,019 : INFO : EPOCH 1 - PROGRESS: at 35.68% examples, 399598 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:37,027 : INFO : EPOCH 1 - PROGRESS: at 37.63% examples, 401569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:38,040 : INFO : EPOCH 1 - PROGRESS: at 39.59% examples, 403726 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:39,044 : INFO : EPOCH 1 - PROGRESS: at 41.48% examples, 405028 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:40,068 : INFO : EPOCH 1 - PROGRESS: at 43.47% examples, 406613 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:41,092 : INFO : EPOCH 1 - PROGRESS: at 45.35% examples, 407353 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:42,095 : INFO : EPOCH 1 - PROGRESS: at 47.49% examples, 410190 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:43,144 : INFO : EPOCH 1 - PROGRESS: at 49.52% examples, 411438 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:44,174 : INFO : EPOCH 1 - PROGRESS: at 51.39% examples, 411191 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:45,204 : INFO : EPOCH 1 - PROGRESS: at 53.26% examples, 411295 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:46,213 : INFO : EPOCH 1 - PROGRESS: at 55.24% examples, 412602 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:47,213 : INFO : EPOCH 1 - PROGRESS: at 57.13% examples, 413651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:48,220 : INFO : EPOCH 1 - PROGRESS: at 59.11% examples, 415096 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:49,248 : INFO : EPOCH 1 - PROGRESS: at 60.99% examples, 415641 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:50,256 : INFO : EPOCH 1 - PROGRESS: at 62.98% examples, 416399 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:51,263 : INFO : EPOCH 1 - PROGRESS: at 65.05% examples, 417667 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:52,267 : INFO : EPOCH 1 - PROGRESS: at 66.95% examples, 418084 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:53,275 : INFO : EPOCH 1 - PROGRESS: at 68.86% examples, 418505 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:54,278 : INFO : EPOCH 1 - PROGRESS: at 70.73% examples, 418936 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:55,280 : INFO : EPOCH 1 - PROGRESS: at 72.70% examples, 419826 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:56,287 : INFO : EPOCH 1 - PROGRESS: at 74.52% examples, 420125 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:57,292 : INFO : EPOCH 1 - PROGRESS: at 76.67% examples, 421845 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:56:58,296 : INFO : EPOCH 1 - PROGRESS: at 78.53% examples, 421695 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:56:59,328 : INFO : EPOCH 1 - PROGRESS: at 80.50% examples, 422357 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:57:00,331 : INFO : EPOCH 1 - PROGRESS: at 82.47% examples, 422647 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:01,337 : INFO : EPOCH 1 - PROGRESS: at 84.41% examples, 423102 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:02,338 : INFO : EPOCH 1 - PROGRESS: at 86.38% examples, 423578 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:03,353 : INFO : EPOCH 1 - PROGRESS: at 88.31% examples, 423888 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:04,389 : INFO : EPOCH 1 - PROGRESS: at 90.15% examples, 423601 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:05,412 : INFO : EPOCH 1 - PROGRESS: at 92.06% examples, 423508 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:06,419 : INFO : EPOCH 1 - PROGRESS: at 94.03% examples, 423863 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:07,428 : INFO : EPOCH 1 - PROGRESS: at 96.19% examples, 424997 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:08,442 : INFO : EPOCH 1 - PROGRESS: at 98.25% examples, 425831 words/s, in_qsize 15, out_qsize 1
    2020-09-30 22:57:09,238 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:57:09,257 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:57:09,263 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:57:09,277 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:57:09,278 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:57:09,280 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:57:09,306 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:57:09,308 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:57:09,308 : INFO : EPOCH - 1 : training on 23279529 raw words (22951015 effective words) took 53.8s, 426333 effective words/s
    2020-09-30 22:57:10,316 : INFO : EPOCH 2 - PROGRESS: at 1.77% examples, 413667 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:11,326 : INFO : EPOCH 2 - PROGRESS: at 3.66% examples, 422914 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:12,347 : INFO : EPOCH 2 - PROGRESS: at 5.54% examples, 424094 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:13,359 : INFO : EPOCH 2 - PROGRESS: at 7.69% examples, 439786 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:14,391 : INFO : EPOCH 2 - PROGRESS: at 9.61% examples, 439339 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:15,416 : INFO : EPOCH 2 - PROGRESS: at 11.64% examples, 443335 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:16,464 : INFO : EPOCH 2 - PROGRESS: at 13.70% examples, 443245 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:17,480 : INFO : EPOCH 2 - PROGRESS: at 15.71% examples, 445007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:18,488 : INFO : EPOCH 2 - PROGRESS: at 17.58% examples, 444604 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:19,496 : INFO : EPOCH 2 - PROGRESS: at 19.49% examples, 443569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:20,538 : INFO : EPOCH 2 - PROGRESS: at 21.50% examples, 443767 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:21,577 : INFO : EPOCH 2 - PROGRESS: at 23.62% examples, 444817 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:22,605 : INFO : EPOCH 2 - PROGRESS: at 25.68% examples, 446084 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:23,622 : INFO : EPOCH 2 - PROGRESS: at 27.75% examples, 447477 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:24,662 : INFO : EPOCH 2 - PROGRESS: at 29.86% examples, 449229 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:25,678 : INFO : EPOCH 2 - PROGRESS: at 32.03% examples, 451460 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:26,692 : INFO : EPOCH 2 - PROGRESS: at 34.20% examples, 452853 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:27,701 : INFO : EPOCH 2 - PROGRESS: at 36.31% examples, 454288 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:28,733 : INFO : EPOCH 2 - PROGRESS: at 38.34% examples, 454151 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:57:29,734 : INFO : EPOCH 2 - PROGRESS: at 40.56% examples, 456946 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:30,750 : INFO : EPOCH 2 - PROGRESS: at 42.58% examples, 457025 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:31,809 : INFO : EPOCH 2 - PROGRESS: at 44.64% examples, 456544 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:32,848 : INFO : EPOCH 2 - PROGRESS: at 46.62% examples, 455249 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:57:33,853 : INFO : EPOCH 2 - PROGRESS: at 48.51% examples, 454382 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:34,855 : INFO : EPOCH 2 - PROGRESS: at 50.53% examples, 454406 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:35,887 : INFO : EPOCH 2 - PROGRESS: at 52.50% examples, 453523 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:36,899 : INFO : EPOCH 2 - PROGRESS: at 54.52% examples, 453765 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:37,904 : INFO : EPOCH 2 - PROGRESS: at 56.56% examples, 454417 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:38,904 : INFO : EPOCH 2 - PROGRESS: at 58.43% examples, 454101 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:39,938 : INFO : EPOCH 2 - PROGRESS: at 60.53% examples, 454860 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:40,973 : INFO : EPOCH 2 - PROGRESS: at 62.89% examples, 456814 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:41,998 : INFO : EPOCH 2 - PROGRESS: at 64.92% examples, 456430 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:43,003 : INFO : EPOCH 2 - PROGRESS: at 67.03% examples, 457150 words/s, in_qsize 16, out_qsize 1
    2020-09-30 22:57:44,019 : INFO : EPOCH 2 - PROGRESS: at 69.06% examples, 457190 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:45,024 : INFO : EPOCH 2 - PROGRESS: at 70.97% examples, 456824 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:46,055 : INFO : EPOCH 2 - PROGRESS: at 72.97% examples, 456655 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:47,078 : INFO : EPOCH 2 - PROGRESS: at 75.02% examples, 457071 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:48,132 : INFO : EPOCH 2 - PROGRESS: at 77.03% examples, 456648 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:49,151 : INFO : EPOCH 2 - PROGRESS: at 79.11% examples, 456670 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:50,181 : INFO : EPOCH 2 - PROGRESS: at 80.72% examples, 454382 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:51,190 : INFO : EPOCH 2 - PROGRESS: at 82.50% examples, 452918 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:52,232 : INFO : EPOCH 2 - PROGRESS: at 84.17% examples, 450968 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:53,245 : INFO : EPOCH 2 - PROGRESS: at 85.72% examples, 448527 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:54,276 : INFO : EPOCH 2 - PROGRESS: at 87.25% examples, 445972 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:57:55,318 : INFO : EPOCH 2 - PROGRESS: at 88.93% examples, 444275 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:56,331 : INFO : EPOCH 2 - PROGRESS: at 90.59% examples, 442755 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:57:57,359 : INFO : EPOCH 2 - PROGRESS: at 92.27% examples, 441154 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:58,373 : INFO : EPOCH 2 - PROGRESS: at 93.84% examples, 439337 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:57:59,396 : INFO : EPOCH 2 - PROGRESS: at 95.65% examples, 438336 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:00,452 : INFO : EPOCH 2 - PROGRESS: at 97.17% examples, 436130 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:01,504 : INFO : EPOCH 2 - PROGRESS: at 98.17% examples, 431787 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:02,506 : INFO : EPOCH 2 - PROGRESS: at 99.00% examples, 427282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:03,203 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:58:03,209 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:58:03,222 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:58:03,227 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:58:03,242 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:58:03,248 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:58:03,248 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:58:03,252 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:58:03,252 : INFO : EPOCH - 2 : training on 23279529 raw words (22951015 effective words) took 53.9s, 425487 effective words/s
    2020-09-30 22:58:04,285 : INFO : EPOCH 3 - PROGRESS: at 1.23% examples, 282713 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:05,308 : INFO : EPOCH 3 - PROGRESS: at 3.00% examples, 339393 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:06,311 : INFO : EPOCH 3 - PROGRESS: at 4.87% examples, 370909 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:07,337 : INFO : EPOCH 3 - PROGRESS: at 6.78% examples, 384094 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:08,341 : INFO : EPOCH 3 - PROGRESS: at 8.64% examples, 393649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:09,344 : INFO : EPOCH 3 - PROGRESS: at 10.69% examples, 409423 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:10,414 : INFO : EPOCH 3 - PROGRESS: at 12.76% examples, 413014 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:11,434 : INFO : EPOCH 3 - PROGRESS: at 14.96% examples, 423034 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:12,446 : INFO : EPOCH 3 - PROGRESS: at 16.70% examples, 421878 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:13,459 : INFO : EPOCH 3 - PROGRESS: at 18.49% examples, 419911 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:14,488 : INFO : EPOCH 3 - PROGRESS: at 20.37% examples, 421071 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:58:15,524 : INFO : EPOCH 3 - PROGRESS: at 22.74% examples, 428935 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:16,529 : INFO : EPOCH 3 - PROGRESS: at 24.90% examples, 433010 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:17,540 : INFO : EPOCH 3 - PROGRESS: at 26.94% examples, 435487 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:18,563 : INFO : EPOCH 3 - PROGRESS: at 29.05% examples, 438450 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:19,595 : INFO : EPOCH 3 - PROGRESS: at 31.08% examples, 439171 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:20,641 : INFO : EPOCH 3 - PROGRESS: at 33.14% examples, 439362 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:21,646 : INFO : EPOCH 3 - PROGRESS: at 35.24% examples, 441161 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:22,703 : INFO : EPOCH 3 - PROGRESS: at 37.45% examples, 443008 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:23,708 : INFO : EPOCH 3 - PROGRESS: at 39.35% examples, 442629 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:24,711 : INFO : EPOCH 3 - PROGRESS: at 41.53% examples, 445373 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:25,718 : INFO : EPOCH 3 - PROGRESS: at 43.77% examples, 448193 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:26,723 : INFO : EPOCH 3 - PROGRESS: at 45.75% examples, 448361 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:58:27,765 : INFO : EPOCH 3 - PROGRESS: at 47.82% examples, 448684 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:28,781 : INFO : EPOCH 3 - PROGRESS: at 49.93% examples, 449383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:29,786 : INFO : EPOCH 3 - PROGRESS: at 52.06% examples, 450658 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:30,789 : INFO : EPOCH 3 - PROGRESS: at 54.09% examples, 451100 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:31,791 : INFO : EPOCH 3 - PROGRESS: at 56.13% examples, 451933 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:32,799 : INFO : EPOCH 3 - PROGRESS: at 58.02% examples, 451621 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:33,814 : INFO : EPOCH 3 - PROGRESS: at 60.11% examples, 452681 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:34,814 : INFO : EPOCH 3 - PROGRESS: at 62.04% examples, 452153 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:35,816 : INFO : EPOCH 3 - PROGRESS: at 64.23% examples, 453407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:36,826 : INFO : EPOCH 3 - PROGRESS: at 66.53% examples, 455336 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:37,833 : INFO : EPOCH 3 - PROGRESS: at 68.77% examples, 456944 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:38,849 : INFO : EPOCH 3 - PROGRESS: at 70.74% examples, 456714 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:39,860 : INFO : EPOCH 3 - PROGRESS: at 72.74% examples, 456803 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:40,881 : INFO : EPOCH 3 - PROGRESS: at 74.74% examples, 457002 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:41,895 : INFO : EPOCH 3 - PROGRESS: at 76.49% examples, 455559 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:42,934 : INFO : EPOCH 3 - PROGRESS: at 78.61% examples, 455582 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:58:43,959 : INFO : EPOCH 3 - PROGRESS: at 80.91% examples, 457195 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:58:44,965 : INFO : EPOCH 3 - PROGRESS: at 82.94% examples, 457306 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:45,985 : INFO : EPOCH 3 - PROGRESS: at 85.22% examples, 458416 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:46,987 : INFO : EPOCH 3 - PROGRESS: at 87.47% examples, 459647 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:48,005 : INFO : EPOCH 3 - PROGRESS: at 89.51% examples, 459569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:49,013 : INFO : EPOCH 3 - PROGRESS: at 91.59% examples, 459852 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:50,029 : INFO : EPOCH 3 - PROGRESS: at 93.69% examples, 460007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:51,045 : INFO : EPOCH 3 - PROGRESS: at 95.78% examples, 459993 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:52,077 : INFO : EPOCH 3 - PROGRESS: at 97.83% examples, 460020 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:53,036 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:58:53,052 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:58:53,057 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:58:53,076 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:58:53,086 : INFO : EPOCH 3 - PROGRESS: at 99.86% examples, 460015 words/s, in_qsize 3, out_qsize 1
    2020-09-30 22:58:53,086 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:58:53,092 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:58:53,113 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:58:53,122 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:58:53,123 : INFO : EPOCH - 3 : training on 23279529 raw words (22951015 effective words) took 49.9s, 460242 effective words/s
    2020-09-30 22:58:54,171 : INFO : EPOCH 4 - PROGRESS: at 1.69% examples, 379235 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:55,198 : INFO : EPOCH 4 - PROGRESS: at 3.70% examples, 415908 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:56,216 : INFO : EPOCH 4 - PROGRESS: at 5.73% examples, 429233 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:57,221 : INFO : EPOCH 4 - PROGRESS: at 8.02% examples, 453518 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:58,284 : INFO : EPOCH 4 - PROGRESS: at 10.22% examples, 460796 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:58:59,289 : INFO : EPOCH 4 - PROGRESS: at 12.37% examples, 465781 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:00,331 : INFO : EPOCH 4 - PROGRESS: at 14.58% examples, 468065 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:01,345 : INFO : EPOCH 4 - PROGRESS: at 16.62% examples, 469375 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:02,381 : INFO : EPOCH 4 - PROGRESS: at 18.73% examples, 469271 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:03,412 : INFO : EPOCH 4 - PROGRESS: at 21.02% examples, 474068 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:04,414 : INFO : EPOCH 4 - PROGRESS: at 23.01% examples, 471349 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:05,437 : INFO : EPOCH 4 - PROGRESS: at 24.90% examples, 466884 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:06,448 : INFO : EPOCH 4 - PROGRESS: at 26.98% examples, 467683 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:07,456 : INFO : EPOCH 4 - PROGRESS: at 28.97% examples, 466979 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:08,471 : INFO : EPOCH 4 - PROGRESS: at 31.13% examples, 468292 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:09,488 : INFO : EPOCH 4 - PROGRESS: at 33.19% examples, 467438 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:10,490 : INFO : EPOCH 4 - PROGRESS: at 35.52% examples, 471102 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:11,518 : INFO : EPOCH 4 - PROGRESS: at 37.82% examples, 473221 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:12,532 : INFO : EPOCH 4 - PROGRESS: at 39.89% examples, 472993 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:13,565 : INFO : EPOCH 4 - PROGRESS: at 42.09% examples, 473686 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:14,607 : INFO : EPOCH 4 - PROGRESS: at 44.17% examples, 472728 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:15,625 : INFO : EPOCH 4 - PROGRESS: at 46.22% examples, 471989 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:16,630 : INFO : EPOCH 4 - PROGRESS: at 48.16% examples, 471148 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:17,633 : INFO : EPOCH 4 - PROGRESS: at 50.41% examples, 472439 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:18,651 : INFO : EPOCH 4 - PROGRESS: at 52.71% examples, 474104 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:19,670 : INFO : EPOCH 4 - PROGRESS: at 54.94% examples, 475276 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:20,682 : INFO : EPOCH 4 - PROGRESS: at 57.00% examples, 475373 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:21,691 : INFO : EPOCH 4 - PROGRESS: at 59.11% examples, 475847 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:22,713 : INFO : EPOCH 4 - PROGRESS: at 61.27% examples, 476407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:23,719 : INFO : EPOCH 4 - PROGRESS: at 63.60% examples, 477813 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:24,754 : INFO : EPOCH 4 - PROGRESS: at 65.68% examples, 477214 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:25,758 : INFO : EPOCH 4 - PROGRESS: at 67.71% examples, 476748 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:26,773 : INFO : EPOCH 4 - PROGRESS: at 69.81% examples, 476811 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:27,773 : INFO : EPOCH 4 - PROGRESS: at 71.81% examples, 476456 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:28,822 : INFO : EPOCH 4 - PROGRESS: at 74.10% examples, 477629 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:29,827 : INFO : EPOCH 4 - PROGRESS: at 76.36% examples, 478821 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:30,835 : INFO : EPOCH 4 - PROGRESS: at 78.61% examples, 479384 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:31,839 : INFO : EPOCH 4 - PROGRESS: at 80.63% examples, 479199 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:32,865 : INFO : EPOCH 4 - PROGRESS: at 82.94% examples, 479983 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:33,892 : INFO : EPOCH 4 - PROGRESS: at 85.14% examples, 480031 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:34,910 : INFO : EPOCH 4 - PROGRESS: at 87.21% examples, 479689 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:35,978 : INFO : EPOCH 4 - PROGRESS: at 89.28% examples, 478786 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:36,982 : INFO : EPOCH 4 - PROGRESS: at 91.39% examples, 478689 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:37,999 : INFO : EPOCH 4 - PROGRESS: at 93.27% examples, 477340 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:39,008 : INFO : EPOCH 4 - PROGRESS: at 95.40% examples, 477212 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:40,025 : INFO : EPOCH 4 - PROGRESS: at 97.70% examples, 478267 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:41,005 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 22:59:41,041 : INFO : EPOCH 4 - PROGRESS: at 99.73% examples, 477797 words/s, in_qsize 6, out_qsize 1
    2020-09-30 22:59:41,041 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 22:59:41,046 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 22:59:41,051 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 22:59:41,057 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 22:59:41,060 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 22:59:41,081 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 22:59:41,095 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 22:59:41,095 : INFO : EPOCH - 4 : training on 23279529 raw words (22951015 effective words) took 48.0s, 478456 effective words/s
    2020-09-30 22:59:42,103 : INFO : EPOCH 5 - PROGRESS: at 2.05% examples, 480861 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:43,127 : INFO : EPOCH 5 - PROGRESS: at 4.38% examples, 501164 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:44,169 : INFO : EPOCH 5 - PROGRESS: at 6.53% examples, 491634 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:45,175 : INFO : EPOCH 5 - PROGRESS: at 8.55% examples, 486266 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:46,190 : INFO : EPOCH 5 - PROGRESS: at 10.53% examples, 481953 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:47,220 : INFO : EPOCH 5 - PROGRESS: at 12.63% examples, 478324 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:48,220 : INFO : EPOCH 5 - PROGRESS: at 14.75% examples, 478924 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:49,235 : INFO : EPOCH 5 - PROGRESS: at 16.75% examples, 477700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:50,239 : INFO : EPOCH 5 - PROGRESS: at 19.03% examples, 482513 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:51,245 : INFO : EPOCH 5 - PROGRESS: at 21.11% examples, 482451 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:59:52,248 : INFO : EPOCH 5 - PROGRESS: at 23.05% examples, 478069 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:53,295 : INFO : EPOCH 5 - PROGRESS: at 25.28% examples, 478266 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:54,297 : INFO : EPOCH 5 - PROGRESS: at 27.41% examples, 479354 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:55,308 : INFO : EPOCH 5 - PROGRESS: at 29.77% examples, 483936 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:56,348 : INFO : EPOCH 5 - PROGRESS: at 31.77% examples, 480723 words/s, in_qsize 14, out_qsize 1
    2020-09-30 22:59:57,350 : INFO : EPOCH 5 - PROGRESS: at 33.99% examples, 481353 words/s, in_qsize 16, out_qsize 0
    2020-09-30 22:59:58,360 : INFO : EPOCH 5 - PROGRESS: at 36.23% examples, 482826 words/s, in_qsize 15, out_qsize 0
    2020-09-30 22:59:59,375 : INFO : EPOCH 5 - PROGRESS: at 38.50% examples, 484728 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:00:00,407 : INFO : EPOCH 5 - PROGRESS: at 40.68% examples, 484805 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:01,430 : INFO : EPOCH 5 - PROGRESS: at 42.84% examples, 484749 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:02,448 : INFO : EPOCH 5 - PROGRESS: at 45.01% examples, 485186 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:03,512 : INFO : EPOCH 5 - PROGRESS: at 47.32% examples, 485448 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:04,517 : INFO : EPOCH 5 - PROGRESS: at 49.38% examples, 484851 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:05,574 : INFO : EPOCH 5 - PROGRESS: at 51.65% examples, 484542 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:00:06,599 : INFO : EPOCH 5 - PROGRESS: at 54.01% examples, 486295 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:07,613 : INFO : EPOCH 5 - PROGRESS: at 55.97% examples, 484899 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:08,625 : INFO : EPOCH 5 - PROGRESS: at 58.10% examples, 485405 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:09,662 : INFO : EPOCH 5 - PROGRESS: at 60.32% examples, 485986 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:10,691 : INFO : EPOCH 5 - PROGRESS: at 62.47% examples, 485478 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:11,716 : INFO : EPOCH 5 - PROGRESS: at 64.73% examples, 485692 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:12,721 : INFO : EPOCH 5 - PROGRESS: at 66.91% examples, 486129 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:13,726 : INFO : EPOCH 5 - PROGRESS: at 69.10% examples, 486629 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:14,737 : INFO : EPOCH 5 - PROGRESS: at 71.13% examples, 486123 words/s, in_qsize 15, out_qsize 1
    2020-09-30 23:00:15,739 : INFO : EPOCH 5 - PROGRESS: at 73.28% examples, 486607 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:16,748 : INFO : EPOCH 5 - PROGRESS: at 75.39% examples, 486667 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:17,792 : INFO : EPOCH 5 - PROGRESS: at 77.56% examples, 486282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:18,823 : INFO : EPOCH 5 - PROGRESS: at 79.84% examples, 486863 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:19,828 : INFO : EPOCH 5 - PROGRESS: at 81.87% examples, 486009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:20,836 : INFO : EPOCH 5 - PROGRESS: at 83.66% examples, 484146 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:21,844 : INFO : EPOCH 5 - PROGRESS: at 85.86% examples, 484322 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:22,845 : INFO : EPOCH 5 - PROGRESS: at 87.92% examples, 484062 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:23,853 : INFO : EPOCH 5 - PROGRESS: at 89.91% examples, 483258 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:24,855 : INFO : EPOCH 5 - PROGRESS: at 91.88% examples, 482436 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:25,866 : INFO : EPOCH 5 - PROGRESS: at 93.99% examples, 482130 words/s, in_qsize 15, out_qsize 1
    2020-09-30 23:00:26,890 : INFO : EPOCH 5 - PROGRESS: at 96.19% examples, 482180 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:27,946 : INFO : EPOCH 5 - PROGRESS: at 97.82% examples, 479400 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:28,831 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:00:28,852 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:00:28,855 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:00:28,865 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:00:28,896 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:00:28,914 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:00:28,915 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:00:28,916 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:00:28,916 : INFO : EPOCH - 5 : training on 23279529 raw words (22951015 effective words) took 47.8s, 479966 effective words/s
    2020-09-30 23:00:29,959 : INFO : EPOCH 6 - PROGRESS: at 1.82% examples, 409020 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:31,004 : INFO : EPOCH 6 - PROGRESS: at 4.19% examples, 464356 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:32,050 : INFO : EPOCH 6 - PROGRESS: at 6.05% examples, 448376 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:33,058 : INFO : EPOCH 6 - PROGRESS: at 8.18% examples, 457905 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:34,065 : INFO : EPOCH 6 - PROGRESS: at 10.15% examples, 458317 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:35,083 : INFO : EPOCH 6 - PROGRESS: at 12.24% examples, 460966 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:36,107 : INFO : EPOCH 6 - PROGRESS: at 14.37% examples, 462518 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:37,153 : INFO : EPOCH 6 - PROGRESS: at 16.51% examples, 464985 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:38,188 : INFO : EPOCH 6 - PROGRESS: at 18.82% examples, 470664 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:39,233 : INFO : EPOCH 6 - PROGRESS: at 20.94% examples, 470929 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:40,246 : INFO : EPOCH 6 - PROGRESS: at 22.91% examples, 467995 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:41,296 : INFO : EPOCH 6 - PROGRESS: at 24.98% examples, 465955 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:42,307 : INFO : EPOCH 6 - PROGRESS: at 26.85% examples, 463232 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:43,324 : INFO : EPOCH 6 - PROGRESS: at 29.01% examples, 465239 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:44,348 : INFO : EPOCH 6 - PROGRESS: at 31.04% examples, 464485 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:45,389 : INFO : EPOCH 6 - PROGRESS: at 33.27% examples, 465536 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:46,405 : INFO : EPOCH 6 - PROGRESS: at 35.53% examples, 467829 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:47,446 : INFO : EPOCH 6 - PROGRESS: at 37.78% examples, 469248 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:48,451 : INFO : EPOCH 6 - PROGRESS: at 40.12% examples, 472419 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:49,481 : INFO : EPOCH 6 - PROGRESS: at 42.22% examples, 472263 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:50,481 : INFO : EPOCH 6 - PROGRESS: at 44.37% examples, 473209 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:00:51,506 : INFO : EPOCH 6 - PROGRESS: at 46.50% examples, 473130 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:52,533 : INFO : EPOCH 6 - PROGRESS: at 48.46% examples, 471824 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:53,543 : INFO : EPOCH 6 - PROGRESS: at 50.67% examples, 472579 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:54,545 : INFO : EPOCH 6 - PROGRESS: at 52.67% examples, 471850 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:55,549 : INFO : EPOCH 6 - PROGRESS: at 54.84% examples, 473019 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:56,571 : INFO : EPOCH 6 - PROGRESS: at 56.96% examples, 473379 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:57,603 : INFO : EPOCH 6 - PROGRESS: at 59.23% examples, 474884 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:00:58,604 : INFO : EPOCH 6 - PROGRESS: at 61.46% examples, 476124 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:00:59,619 : INFO : EPOCH 6 - PROGRESS: at 63.60% examples, 476148 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:00,645 : INFO : EPOCH 6 - PROGRESS: at 65.55% examples, 474814 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:01,663 : INFO : EPOCH 6 - PROGRESS: at 67.71% examples, 475119 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:02,670 : INFO : EPOCH 6 - PROGRESS: at 69.67% examples, 474472 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:01:03,680 : INFO : EPOCH 6 - PROGRESS: at 71.65% examples, 473782 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:01:04,721 : INFO : EPOCH 6 - PROGRESS: at 73.65% examples, 473245 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:05,778 : INFO : EPOCH 6 - PROGRESS: at 75.74% examples, 472808 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:01:06,780 : INFO : EPOCH 6 - PROGRESS: at 77.78% examples, 472580 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:07,785 : INFO : EPOCH 6 - PROGRESS: at 80.05% examples, 473797 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:08,787 : INFO : EPOCH 6 - PROGRESS: at 82.12% examples, 473587 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:09,787 : INFO : EPOCH 6 - PROGRESS: at 84.17% examples, 473613 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:10,796 : INFO : EPOCH 6 - PROGRESS: at 86.16% examples, 472872 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:11,808 : INFO : EPOCH 6 - PROGRESS: at 88.22% examples, 472752 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:12,818 : INFO : EPOCH 6 - PROGRESS: at 90.37% examples, 473117 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:01:13,829 : INFO : EPOCH 6 - PROGRESS: at 92.60% examples, 473497 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:14,840 : INFO : EPOCH 6 - PROGRESS: at 94.69% examples, 473437 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:15,857 : INFO : EPOCH 6 - PROGRESS: at 96.96% examples, 474150 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:16,863 : INFO : EPOCH 6 - PROGRESS: at 99.05% examples, 474285 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:17,139 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:01:17,147 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:01:17,153 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:01:17,167 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:01:17,169 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:01:17,191 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:01:17,203 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:01:17,210 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:01:17,210 : INFO : EPOCH - 6 : training on 23279529 raw words (22951015 effective words) took 48.3s, 475265 effective words/s
    2020-09-30 23:01:18,235 : INFO : EPOCH 7 - PROGRESS: at 2.01% examples, 463404 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:19,235 : INFO : EPOCH 7 - PROGRESS: at 4.19% examples, 478685 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:20,254 : INFO : EPOCH 7 - PROGRESS: at 6.40% examples, 487075 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:21,255 : INFO : EPOCH 7 - PROGRESS: at 8.30% examples, 476092 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:22,267 : INFO : EPOCH 7 - PROGRESS: at 10.34% examples, 476019 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:23,272 : INFO : EPOCH 7 - PROGRESS: at 12.46% examples, 477017 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:24,293 : INFO : EPOCH 7 - PROGRESS: at 14.54% examples, 474959 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:01:25,295 : INFO : EPOCH 7 - PROGRESS: at 16.62% examples, 477353 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:26,298 : INFO : EPOCH 7 - PROGRESS: at 18.69% examples, 477015 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:27,314 : INFO : EPOCH 7 - PROGRESS: at 20.84% examples, 478939 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:28,346 : INFO : EPOCH 7 - PROGRESS: at 23.18% examples, 481424 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:29,350 : INFO : EPOCH 7 - PROGRESS: at 25.55% examples, 486214 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:30,359 : INFO : EPOCH 7 - PROGRESS: at 27.56% examples, 484199 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:31,374 : INFO : EPOCH 7 - PROGRESS: at 29.89% examples, 487680 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:32,387 : INFO : EPOCH 7 - PROGRESS: at 32.07% examples, 487504 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:33,389 : INFO : EPOCH 7 - PROGRESS: at 34.20% examples, 486598 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:34,389 : INFO : EPOCH 7 - PROGRESS: at 36.08% examples, 483578 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:35,401 : INFO : EPOCH 7 - PROGRESS: at 38.03% examples, 481188 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:36,403 : INFO : EPOCH 7 - PROGRESS: at 40.12% examples, 480848 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:37,409 : INFO : EPOCH 7 - PROGRESS: at 42.22% examples, 480829 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:38,416 : INFO : EPOCH 7 - PROGRESS: at 44.25% examples, 479862 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:39,422 : INFO : EPOCH 7 - PROGRESS: at 46.43% examples, 480324 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:40,461 : INFO : EPOCH 7 - PROGRESS: at 48.55% examples, 480084 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:41,495 : INFO : EPOCH 7 - PROGRESS: at 50.84% examples, 480852 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:42,504 : INFO : EPOCH 7 - PROGRESS: at 52.86% examples, 480028 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:43,505 : INFO : EPOCH 7 - PROGRESS: at 55.15% examples, 481649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:44,507 : INFO : EPOCH 7 - PROGRESS: at 57.43% examples, 483519 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:45,543 : INFO : EPOCH 7 - PROGRESS: at 59.50% examples, 483171 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:46,544 : INFO : EPOCH 7 - PROGRESS: at 61.67% examples, 483522 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:47,578 : INFO : EPOCH 7 - PROGRESS: at 63.81% examples, 482987 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:48,602 : INFO : EPOCH 7 - PROGRESS: at 66.07% examples, 483612 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:01:49,613 : INFO : EPOCH 7 - PROGRESS: at 68.47% examples, 485556 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:50,657 : INFO : EPOCH 7 - PROGRESS: at 70.44% examples, 484052 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:51,687 : INFO : EPOCH 7 - PROGRESS: at 72.56% examples, 483919 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:52,745 : INFO : EPOCH 7 - PROGRESS: at 74.70% examples, 483651 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:53,771 : INFO : EPOCH 7 - PROGRESS: at 77.03% examples, 484915 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:54,804 : INFO : EPOCH 7 - PROGRESS: at 79.46% examples, 486040 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:55,815 : INFO : EPOCH 7 - PROGRESS: at 81.40% examples, 484864 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:56,820 : INFO : EPOCH 7 - PROGRESS: at 83.44% examples, 484533 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:57,829 : INFO : EPOCH 7 - PROGRESS: at 85.51% examples, 483964 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:58,842 : INFO : EPOCH 7 - PROGRESS: at 87.76% examples, 484505 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:01:59,850 : INFO : EPOCH 7 - PROGRESS: at 89.91% examples, 484604 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:00,852 : INFO : EPOCH 7 - PROGRESS: at 92.14% examples, 485057 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:01,857 : INFO : EPOCH 7 - PROGRESS: at 94.42% examples, 485657 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:02,892 : INFO : EPOCH 7 - PROGRESS: at 96.44% examples, 484653 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:03,897 : INFO : EPOCH 7 - PROGRESS: at 98.56% examples, 484608 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:04,441 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:02:04,457 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:02:04,460 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:02:04,487 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:02:04,503 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:02:04,504 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:02:04,516 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:02:04,527 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:02:04,527 : INFO : EPOCH - 7 : training on 23279529 raw words (22951015 effective words) took 47.3s, 485084 effective words/s
    2020-09-30 23:02:05,537 : INFO : EPOCH 8 - PROGRESS: at 1.85% examples, 431863 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:06,561 : INFO : EPOCH 8 - PROGRESS: at 4.23% examples, 481472 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:07,572 : INFO : EPOCH 8 - PROGRESS: at 6.44% examples, 490143 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:08,582 : INFO : EPOCH 8 - PROGRESS: at 8.51% examples, 486913 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:02:09,597 : INFO : EPOCH 8 - PROGRESS: at 10.69% examples, 491967 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:10,601 : INFO : EPOCH 8 - PROGRESS: at 12.90% examples, 491810 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:11,622 : INFO : EPOCH 8 - PROGRESS: at 15.26% examples, 497478 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:02:12,641 : INFO : EPOCH 8 - PROGRESS: at 17.45% examples, 499466 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:02:13,654 : INFO : EPOCH 8 - PROGRESS: at 19.61% examples, 498231 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:14,658 : INFO : EPOCH 8 - PROGRESS: at 21.66% examples, 495787 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:15,660 : INFO : EPOCH 8 - PROGRESS: at 23.93% examples, 496372 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:16,692 : INFO : EPOCH 8 - PROGRESS: at 25.76% examples, 489128 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:17,694 : INFO : EPOCH 8 - PROGRESS: at 27.82% examples, 487947 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:02:18,707 : INFO : EPOCH 8 - PROGRESS: at 29.99% examples, 488492 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:19,716 : INFO : EPOCH 8 - PROGRESS: at 32.03% examples, 486510 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:20,732 : INFO : EPOCH 8 - PROGRESS: at 34.15% examples, 485217 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:21,763 : INFO : EPOCH 8 - PROGRESS: at 36.27% examples, 484211 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:22,773 : INFO : EPOCH 8 - PROGRESS: at 38.41% examples, 484562 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:02:23,783 : INFO : EPOCH 8 - PROGRESS: at 40.60% examples, 485220 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:24,815 : INFO : EPOCH 8 - PROGRESS: at 42.62% examples, 483504 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:25,839 : INFO : EPOCH 8 - PROGRESS: at 44.76% examples, 483399 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:26,841 : INFO : EPOCH 8 - PROGRESS: at 46.95% examples, 483768 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:27,863 : INFO : EPOCH 8 - PROGRESS: at 49.13% examples, 484168 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:28,875 : INFO : EPOCH 8 - PROGRESS: at 51.35% examples, 484347 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:29,880 : INFO : EPOCH 8 - PROGRESS: at 53.63% examples, 485732 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:30,900 : INFO : EPOCH 8 - PROGRESS: at 55.80% examples, 486094 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:31,923 : INFO : EPOCH 8 - PROGRESS: at 57.86% examples, 485665 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:02:32,951 : INFO : EPOCH 8 - PROGRESS: at 59.87% examples, 484669 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:33,965 : INFO : EPOCH 8 - PROGRESS: at 62.04% examples, 484785 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:34,995 : INFO : EPOCH 8 - PROGRESS: at 63.93% examples, 482358 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:35,998 : INFO : EPOCH 8 - PROGRESS: at 66.02% examples, 482093 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:37,024 : INFO : EPOCH 8 - PROGRESS: at 68.09% examples, 481468 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:38,026 : INFO : EPOCH 8 - PROGRESS: at 70.15% examples, 481281 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:02:39,078 : INFO : EPOCH 8 - PROGRESS: at 72.18% examples, 480367 words/s, in_qsize 13, out_qsize 2
    2020-09-30 23:02:40,079 : INFO : EPOCH 8 - PROGRESS: at 74.35% examples, 481248 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:41,101 : INFO : EPOCH 8 - PROGRESS: at 76.53% examples, 481593 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:42,106 : INFO : EPOCH 8 - PROGRESS: at 78.85% examples, 482620 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:43,107 : INFO : EPOCH 8 - PROGRESS: at 81.04% examples, 483168 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:44,121 : INFO : EPOCH 8 - PROGRESS: at 83.28% examples, 483740 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:45,139 : INFO : EPOCH 8 - PROGRESS: at 85.47% examples, 483811 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:46,140 : INFO : EPOCH 8 - PROGRESS: at 87.34% examples, 482400 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:47,147 : INFO : EPOCH 8 - PROGRESS: at 89.51% examples, 482564 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:48,164 : INFO : EPOCH 8 - PROGRESS: at 91.67% examples, 482685 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:49,177 : INFO : EPOCH 8 - PROGRESS: at 93.81% examples, 482569 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:50,216 : INFO : EPOCH 8 - PROGRESS: at 96.00% examples, 482240 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:51,217 : INFO : EPOCH 8 - PROGRESS: at 98.00% examples, 481875 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:52,060 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:02:52,065 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:02:52,080 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:02:52,085 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:02:52,096 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:02:52,110 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:02:52,125 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:02:52,137 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:02:52,137 : INFO : EPOCH - 8 : training on 23279529 raw words (22951015 effective words) took 47.6s, 482090 effective words/s
    2020-09-30 23:02:53,208 : INFO : EPOCH 9 - PROGRESS: at 2.01% examples, 443560 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:54,228 : INFO : EPOCH 9 - PROGRESS: at 4.04% examples, 449797 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:55,229 : INFO : EPOCH 9 - PROGRESS: at 6.32% examples, 473491 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:56,257 : INFO : EPOCH 9 - PROGRESS: at 8.55% examples, 481694 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:57,266 : INFO : EPOCH 9 - PROGRESS: at 10.73% examples, 488318 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:58,342 : INFO : EPOCH 9 - PROGRESS: at 13.06% examples, 487757 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:02:59,369 : INFO : EPOCH 9 - PROGRESS: at 15.34% examples, 490799 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:00,381 : INFO : EPOCH 9 - PROGRESS: at 17.45% examples, 491632 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:03:01,394 : INFO : EPOCH 9 - PROGRESS: at 19.61% examples, 491296 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:02,407 : INFO : EPOCH 9 - PROGRESS: at 21.75% examples, 491000 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:03,414 : INFO : EPOCH 9 - PROGRESS: at 23.92% examples, 490046 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:04,429 : INFO : EPOCH 9 - PROGRESS: at 26.25% examples, 493595 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:05,442 : INFO : EPOCH 9 - PROGRESS: at 28.27% examples, 490880 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:06,465 : INFO : EPOCH 9 - PROGRESS: at 30.57% examples, 492897 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:07,482 : INFO : EPOCH 9 - PROGRESS: at 32.92% examples, 494787 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:08,493 : INFO : EPOCH 9 - PROGRESS: at 34.94% examples, 492037 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:09,531 : INFO : EPOCH 9 - PROGRESS: at 36.87% examples, 487654 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:10,550 : INFO : EPOCH 9 - PROGRESS: at 39.26% examples, 490740 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:11,567 : INFO : EPOCH 9 - PROGRESS: at 41.17% examples, 487925 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:12,599 : INFO : EPOCH 9 - PROGRESS: at 43.47% examples, 488809 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:13,637 : INFO : EPOCH 9 - PROGRESS: at 45.75% examples, 489510 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:14,639 : INFO : EPOCH 9 - PROGRESS: at 48.07% examples, 491368 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:03:15,669 : INFO : EPOCH 9 - PROGRESS: at 50.44% examples, 492494 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:16,686 : INFO : EPOCH 9 - PROGRESS: at 52.70% examples, 493014 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:17,734 : INFO : EPOCH 9 - PROGRESS: at 54.84% examples, 492184 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:18,745 : INFO : EPOCH 9 - PROGRESS: at 57.25% examples, 494585 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:19,745 : INFO : EPOCH 9 - PROGRESS: at 59.38% examples, 494817 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:20,761 : INFO : EPOCH 9 - PROGRESS: at 61.50% examples, 494177 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:21,795 : INFO : EPOCH 9 - PROGRESS: at 63.81% examples, 494563 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:22,809 : INFO : EPOCH 9 - PROGRESS: at 66.15% examples, 495613 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:23,840 : INFO : EPOCH 9 - PROGRESS: at 68.34% examples, 495364 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:24,853 : INFO : EPOCH 9 - PROGRESS: at 70.48% examples, 495161 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:25,909 : INFO : EPOCH 9 - PROGRESS: at 72.52% examples, 493744 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:26,973 : INFO : EPOCH 9 - PROGRESS: at 74.82% examples, 494212 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:27,988 : INFO : EPOCH 9 - PROGRESS: at 76.91% examples, 493736 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:29,019 : INFO : EPOCH 9 - PROGRESS: at 79.19% examples, 493853 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:30,044 : INFO : EPOCH 9 - PROGRESS: at 81.39% examples, 493801 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:31,101 : INFO : EPOCH 9 - PROGRESS: at 83.57% examples, 493317 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:32,110 : INFO : EPOCH 9 - PROGRESS: at 85.73% examples, 493015 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:33,118 : INFO : EPOCH 9 - PROGRESS: at 87.88% examples, 492918 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:34,139 : INFO : EPOCH 9 - PROGRESS: at 89.98% examples, 492444 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:35,151 : INFO : EPOCH 9 - PROGRESS: at 92.14% examples, 492158 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:36,163 : INFO : EPOCH 9 - PROGRESS: at 94.03% examples, 490525 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:37,190 : INFO : EPOCH 9 - PROGRESS: at 96.19% examples, 490141 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:38,197 : INFO : EPOCH 9 - PROGRESS: at 98.34% examples, 490155 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:38,851 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:03:38,894 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:03:38,900 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:03:38,903 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:03:38,908 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:03:38,917 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:03:38,925 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:03:38,928 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:03:38,928 : INFO : EPOCH - 9 : training on 23279529 raw words (22951015 effective words) took 46.8s, 490538 effective words/s
    2020-09-30 23:03:39,951 : INFO : EPOCH 10 - PROGRESS: at 2.01% examples, 464928 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:40,963 : INFO : EPOCH 10 - PROGRESS: at 4.19% examples, 476471 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:41,978 : INFO : EPOCH 10 - PROGRESS: at 6.44% examples, 489429 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:43,026 : INFO : EPOCH 10 - PROGRESS: at 8.76% examples, 496051 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:44,030 : INFO : EPOCH 10 - PROGRESS: at 11.05% examples, 504236 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:45,046 : INFO : EPOCH 10 - PROGRESS: at 13.15% examples, 497855 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:46,056 : INFO : EPOCH 10 - PROGRESS: at 15.21% examples, 493882 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:47,076 : INFO : EPOCH 10 - PROGRESS: at 17.41% examples, 496242 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:48,108 : INFO : EPOCH 10 - PROGRESS: at 19.77% examples, 499588 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:49,132 : INFO : EPOCH 10 - PROGRESS: at 22.02% examples, 499850 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:50,149 : INFO : EPOCH 10 - PROGRESS: at 24.15% examples, 496841 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:51,150 : INFO : EPOCH 10 - PROGRESS: at 26.40% examples, 499595 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:52,152 : INFO : EPOCH 10 - PROGRESS: at 28.57% examples, 498924 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:53,155 : INFO : EPOCH 10 - PROGRESS: at 30.77% examples, 499739 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:54,167 : INFO : EPOCH 10 - PROGRESS: at 33.18% examples, 502000 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:03:55,175 : INFO : EPOCH 10 - PROGRESS: at 35.63% examples, 504826 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:03:56,185 : INFO : EPOCH 10 - PROGRESS: at 37.78% examples, 503891 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:57,209 : INFO : EPOCH 10 - PROGRESS: at 40.07% examples, 504323 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:58,227 : INFO : EPOCH 10 - PROGRESS: at 42.22% examples, 503280 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:03:59,229 : INFO : EPOCH 10 - PROGRESS: at 44.37% examples, 502692 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:00,245 : INFO : EPOCH 10 - PROGRESS: at 46.57% examples, 502311 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:01,281 : INFO : EPOCH 10 - PROGRESS: at 48.84% examples, 502414 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:02,291 : INFO : EPOCH 10 - PROGRESS: at 51.10% examples, 502315 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:03,306 : INFO : EPOCH 10 - PROGRESS: at 53.51% examples, 504023 words/s, in_qsize 15, out_qsize 1
    2020-09-30 23:04:04,306 : INFO : EPOCH 10 - PROGRESS: at 55.52% examples, 502516 words/s, in_qsize 16, out_qsize 2
    2020-09-30 23:04:05,339 : INFO : EPOCH 10 - PROGRESS: at 57.58% examples, 501221 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:06,367 : INFO : EPOCH 10 - PROGRESS: at 59.66% examples, 500323 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:07,399 : INFO : EPOCH 10 - PROGRESS: at 62.00% examples, 500912 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:08,425 : INFO : EPOCH 10 - PROGRESS: at 64.14% examples, 499912 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:09,437 : INFO : EPOCH 10 - PROGRESS: at 66.28% examples, 499195 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:10,442 : INFO : EPOCH 10 - PROGRESS: at 68.61% examples, 500209 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:11,444 : INFO : EPOCH 10 - PROGRESS: at 70.81% examples, 500597 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:12,477 : INFO : EPOCH 10 - PROGRESS: at 72.84% examples, 499315 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:13,497 : INFO : EPOCH 10 - PROGRESS: at 74.90% examples, 498585 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:14,500 : INFO : EPOCH 10 - PROGRESS: at 76.99% examples, 498134 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:15,505 : INFO : EPOCH 10 - PROGRESS: at 79.37% examples, 499029 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:16,532 : INFO : EPOCH 10 - PROGRESS: at 81.53% examples, 498540 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:17,534 : INFO : EPOCH 10 - PROGRESS: at 83.87% examples, 499658 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:18,585 : INFO : EPOCH 10 - PROGRESS: at 86.12% examples, 499138 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:19,588 : INFO : EPOCH 10 - PROGRESS: at 88.27% examples, 498963 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:20,589 : INFO : EPOCH 10 - PROGRESS: at 90.32% examples, 498337 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:21,590 : INFO : EPOCH 10 - PROGRESS: at 92.60% examples, 498495 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:22,598 : INFO : EPOCH 10 - PROGRESS: at 94.71% examples, 497876 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:23,639 : INFO : EPOCH 10 - PROGRESS: at 96.74% examples, 496717 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:24,657 : INFO : EPOCH 10 - PROGRESS: at 98.97% examples, 496868 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:25,029 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:04:25,039 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:04:25,041 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:04:25,052 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:04:25,054 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:04:25,059 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:04:25,067 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:04:25,081 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:04:25,081 : INFO : EPOCH - 10 : training on 23279529 raw words (22951015 effective words) took 46.1s, 497323 effective words/s
    2020-09-30 23:04:26,093 : INFO : EPOCH 11 - PROGRESS: at 2.18% examples, 507648 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:27,152 : INFO : EPOCH 11 - PROGRESS: at 4.38% examples, 491727 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:28,170 : INFO : EPOCH 11 - PROGRESS: at 6.49% examples, 486148 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:29,186 : INFO : EPOCH 11 - PROGRESS: at 8.85% examples, 499805 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:30,191 : INFO : EPOCH 11 - PROGRESS: at 10.90% examples, 497584 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:31,196 : INFO : EPOCH 11 - PROGRESS: at 13.11% examples, 496472 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:32,210 : INFO : EPOCH 11 - PROGRESS: at 15.13% examples, 491023 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:33,253 : INFO : EPOCH 11 - PROGRESS: at 17.32% examples, 492407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:34,281 : INFO : EPOCH 11 - PROGRESS: at 19.57% examples, 493243 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:35,337 : INFO : EPOCH 11 - PROGRESS: at 21.58% examples, 487845 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:36,384 : INFO : EPOCH 11 - PROGRESS: at 23.80% examples, 486270 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:37,440 : INFO : EPOCH 11 - PROGRESS: at 25.88% examples, 483844 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:38,465 : INFO : EPOCH 11 - PROGRESS: at 27.73% examples, 478559 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:39,477 : INFO : EPOCH 11 - PROGRESS: at 29.55% examples, 474409 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:40,490 : INFO : EPOCH 11 - PROGRESS: at 31.56% examples, 472702 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:41,498 : INFO : EPOCH 11 - PROGRESS: at 33.45% examples, 469502 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:42,510 : INFO : EPOCH 11 - PROGRESS: at 35.52% examples, 469441 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:43,521 : INFO : EPOCH 11 - PROGRESS: at 37.63% examples, 469431 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:44,533 : INFO : EPOCH 11 - PROGRESS: at 39.44% examples, 466474 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:45,590 : INFO : EPOCH 11 - PROGRESS: at 41.48% examples, 465547 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:46,593 : INFO : EPOCH 11 - PROGRESS: at 43.68% examples, 467207 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:47,602 : INFO : EPOCH 11 - PROGRESS: at 45.84% examples, 468138 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:48,694 : INFO : EPOCH 11 - PROGRESS: at 48.03% examples, 467829 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:49,745 : INFO : EPOCH 11 - PROGRESS: at 50.27% examples, 468308 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:50,779 : INFO : EPOCH 11 - PROGRESS: at 52.67% examples, 470567 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:04:51,788 : INFO : EPOCH 11 - PROGRESS: at 54.63% examples, 469889 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:52,822 : INFO : EPOCH 11 - PROGRESS: at 56.51% examples, 468069 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:53,841 : INFO : EPOCH 11 - PROGRESS: at 58.39% examples, 466979 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:54,847 : INFO : EPOCH 11 - PROGRESS: at 60.15% examples, 465107 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:55,850 : INFO : EPOCH 11 - PROGRESS: at 62.13% examples, 464447 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:56,852 : INFO : EPOCH 11 - PROGRESS: at 63.81% examples, 461660 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:57,882 : INFO : EPOCH 11 - PROGRESS: at 66.11% examples, 463138 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:04:58,914 : INFO : EPOCH 11 - PROGRESS: at 68.39% examples, 464455 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:04:59,948 : INFO : EPOCH 11 - PROGRESS: at 70.85% examples, 467115 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:00,950 : INFO : EPOCH 11 - PROGRESS: at 72.88% examples, 467278 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:01,955 : INFO : EPOCH 11 - PROGRESS: at 74.90% examples, 467413 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:02,973 : INFO : EPOCH 11 - PROGRESS: at 77.07% examples, 468128 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:03,974 : INFO : EPOCH 11 - PROGRESS: at 79.46% examples, 469803 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:04,974 : INFO : EPOCH 11 - PROGRESS: at 81.44% examples, 469453 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:05,974 : INFO : EPOCH 11 - PROGRESS: at 83.44% examples, 469319 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:06,976 : INFO : EPOCH 11 - PROGRESS: at 85.84% examples, 471079 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:08,000 : INFO : EPOCH 11 - PROGRESS: at 88.09% examples, 471786 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:09,010 : INFO : EPOCH 11 - PROGRESS: at 89.89% examples, 470386 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:10,016 : INFO : EPOCH 11 - PROGRESS: at 92.14% examples, 471111 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:11,017 : INFO : EPOCH 11 - PROGRESS: at 94.24% examples, 471180 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:12,024 : INFO : EPOCH 11 - PROGRESS: at 96.50% examples, 471844 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:05:13,031 : INFO : EPOCH 11 - PROGRESS: at 98.72% examples, 472650 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:13,460 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:05:13,463 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:05:13,464 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:05:13,470 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:05:13,477 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:05:13,496 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:05:13,506 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:05:13,516 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:05:13,517 : INFO : EPOCH - 11 : training on 23279529 raw words (22951015 effective words) took 48.4s, 473876 effective words/s
    2020-09-30 23:05:14,576 : INFO : EPOCH 12 - PROGRESS: at 2.28% examples, 503825 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:15,582 : INFO : EPOCH 12 - PROGRESS: at 4.42% examples, 497770 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:16,596 : INFO : EPOCH 12 - PROGRESS: at 6.67% examples, 500279 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:17,623 : INFO : EPOCH 12 - PROGRESS: at 8.81% examples, 497305 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:18,656 : INFO : EPOCH 12 - PROGRESS: at 10.99% examples, 498604 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:19,658 : INFO : EPOCH 12 - PROGRESS: at 13.11% examples, 494338 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:20,660 : INFO : EPOCH 12 - PROGRESS: at 15.18% examples, 491397 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:21,660 : INFO : EPOCH 12 - PROGRESS: at 17.16% examples, 489371 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:22,687 : INFO : EPOCH 12 - PROGRESS: at 19.37% examples, 489627 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:23,706 : INFO : EPOCH 12 - PROGRESS: at 21.54% examples, 490081 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:05:24,706 : INFO : EPOCH 12 - PROGRESS: at 23.97% examples, 494708 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:25,745 : INFO : EPOCH 12 - PROGRESS: at 26.05% examples, 492175 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:26,761 : INFO : EPOCH 12 - PROGRESS: at 28.07% examples, 489500 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:27,789 : INFO : EPOCH 12 - PROGRESS: at 30.15% examples, 488065 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:28,807 : INFO : EPOCH 12 - PROGRESS: at 32.62% examples, 492104 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:29,842 : INFO : EPOCH 12 - PROGRESS: at 35.07% examples, 494737 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:30,876 : INFO : EPOCH 12 - PROGRESS: at 37.45% examples, 496406 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:31,883 : INFO : EPOCH 12 - PROGRESS: at 39.71% examples, 497749 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:05:32,891 : INFO : EPOCH 12 - PROGRESS: at 41.66% examples, 494791 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:33,899 : INFO : EPOCH 12 - PROGRESS: at 43.94% examples, 495908 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:34,933 : INFO : EPOCH 12 - PROGRESS: at 46.12% examples, 495017 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:35,964 : INFO : EPOCH 12 - PROGRESS: at 48.42% examples, 495993 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:36,994 : INFO : EPOCH 12 - PROGRESS: at 50.44% examples, 493644 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:38,063 : INFO : EPOCH 12 - PROGRESS: at 52.55% examples, 491490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:39,065 : INFO : EPOCH 12 - PROGRESS: at 54.77% examples, 492355 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:40,094 : INFO : EPOCH 12 - PROGRESS: at 57.14% examples, 494046 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:41,105 : INFO : EPOCH 12 - PROGRESS: at 59.22% examples, 493791 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:42,107 : INFO : EPOCH 12 - PROGRESS: at 61.54% examples, 495082 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:43,140 : INFO : EPOCH 12 - PROGRESS: at 63.69% examples, 494155 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:44,151 : INFO : EPOCH 12 - PROGRESS: at 66.03% examples, 495282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:45,177 : INFO : EPOCH 12 - PROGRESS: at 68.38% examples, 496334 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:05:46,203 : INFO : EPOCH 12 - PROGRESS: at 70.73% examples, 497389 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:05:47,217 : INFO : EPOCH 12 - PROGRESS: at 73.16% examples, 499383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:48,289 : INFO : EPOCH 12 - PROGRESS: at 75.48% examples, 499555 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:49,297 : INFO : EPOCH 12 - PROGRESS: at 77.89% examples, 500660 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:50,303 : INFO : EPOCH 12 - PROGRESS: at 80.09% examples, 500901 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:51,305 : INFO : EPOCH 12 - PROGRESS: at 82.43% examples, 501476 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:05:52,312 : INFO : EPOCH 12 - PROGRESS: at 84.53% examples, 500973 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:53,342 : INFO : EPOCH 12 - PROGRESS: at 86.80% examples, 500914 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:54,362 : INFO : EPOCH 12 - PROGRESS: at 88.89% examples, 500221 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:55,390 : INFO : EPOCH 12 - PROGRESS: at 91.26% examples, 500696 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:56,405 : INFO : EPOCH 12 - PROGRESS: at 93.63% examples, 501281 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:57,426 : INFO : EPOCH 12 - PROGRESS: at 95.70% examples, 500250 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:58,476 : INFO : EPOCH 12 - PROGRESS: at 97.79% examples, 499361 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:05:59,311 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:05:59,326 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:05:59,326 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:05:59,351 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:05:59,401 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:05:59,402 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:05:59,406 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:05:59,408 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:05:59,408 : INFO : EPOCH - 12 : training on 23279529 raw words (22951015 effective words) took 45.9s, 500151 effective words/s
    2020-09-30 23:06:00,434 : INFO : EPOCH 13 - PROGRESS: at 2.01% examples, 463668 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:01,467 : INFO : EPOCH 13 - PROGRESS: at 4.27% examples, 480695 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:02,491 : INFO : EPOCH 13 - PROGRESS: at 6.63% examples, 496838 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:03,517 : INFO : EPOCH 13 - PROGRESS: at 8.89% examples, 501875 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:04,535 : INFO : EPOCH 13 - PROGRESS: at 10.86% examples, 494222 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:06:05,547 : INFO : EPOCH 13 - PROGRESS: at 13.06% examples, 493067 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:06,552 : INFO : EPOCH 13 - PROGRESS: at 15.21% examples, 492821 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:07,593 : INFO : EPOCH 13 - PROGRESS: at 17.32% examples, 491638 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:06:08,622 : INFO : EPOCH 13 - PROGRESS: at 19.61% examples, 493671 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:09,626 : INFO : EPOCH 13 - PROGRESS: at 21.66% examples, 491648 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:10,635 : INFO : EPOCH 13 - PROGRESS: at 23.83% examples, 490526 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:11,638 : INFO : EPOCH 13 - PROGRESS: at 26.12% examples, 493754 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:12,676 : INFO : EPOCH 13 - PROGRESS: at 28.40% examples, 494361 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:13,677 : INFO : EPOCH 13 - PROGRESS: at 30.65% examples, 496262 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:14,693 : INFO : EPOCH 13 - PROGRESS: at 32.84% examples, 495511 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:15,733 : INFO : EPOCH 13 - PROGRESS: at 35.15% examples, 495940 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:16,735 : INFO : EPOCH 13 - PROGRESS: at 37.19% examples, 494019 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:17,766 : INFO : EPOCH 13 - PROGRESS: at 39.52% examples, 495328 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:06:18,797 : INFO : EPOCH 13 - PROGRESS: at 41.35% examples, 490995 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:19,813 : INFO : EPOCH 13 - PROGRESS: at 43.31% examples, 488293 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:20,834 : INFO : EPOCH 13 - PROGRESS: at 45.43% examples, 488070 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:06:21,870 : INFO : EPOCH 13 - PROGRESS: at 47.78% examples, 489278 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:22,878 : INFO : EPOCH 13 - PROGRESS: at 50.15% examples, 490917 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:23,884 : INFO : EPOCH 13 - PROGRESS: at 52.58% examples, 493305 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:06:24,893 : INFO : EPOCH 13 - PROGRESS: at 54.77% examples, 493592 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:25,909 : INFO : EPOCH 13 - PROGRESS: at 56.96% examples, 494021 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:26,969 : INFO : EPOCH 13 - PROGRESS: at 59.11% examples, 493271 words/s, in_qsize 13, out_qsize 2
    2020-09-30 23:06:27,985 : INFO : EPOCH 13 - PROGRESS: at 61.32% examples, 493672 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:28,990 : INFO : EPOCH 13 - PROGRESS: at 63.47% examples, 493264 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:06:29,996 : INFO : EPOCH 13 - PROGRESS: at 65.51% examples, 492249 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:31,014 : INFO : EPOCH 13 - PROGRESS: at 67.66% examples, 491993 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:32,046 : INFO : EPOCH 13 - PROGRESS: at 70.04% examples, 493115 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:33,060 : INFO : EPOCH 13 - PROGRESS: at 72.40% examples, 494645 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:34,060 : INFO : EPOCH 13 - PROGRESS: at 74.48% examples, 494602 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:35,115 : INFO : EPOCH 13 - PROGRESS: at 76.53% examples, 493313 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:36,142 : INFO : EPOCH 13 - PROGRESS: at 78.65% examples, 492435 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:37,158 : INFO : EPOCH 13 - PROGRESS: at 80.68% examples, 491734 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:38,172 : INFO : EPOCH 13 - PROGRESS: at 82.87% examples, 491616 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:39,181 : INFO : EPOCH 13 - PROGRESS: at 85.05% examples, 491582 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:40,188 : INFO : EPOCH 13 - PROGRESS: at 87.13% examples, 491082 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:41,213 : INFO : EPOCH 13 - PROGRESS: at 89.63% examples, 492687 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:06:42,235 : INFO : EPOCH 13 - PROGRESS: at 91.81% examples, 492506 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:43,236 : INFO : EPOCH 13 - PROGRESS: at 94.11% examples, 493193 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:06:44,243 : INFO : EPOCH 13 - PROGRESS: at 96.25% examples, 492760 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:45,249 : INFO : EPOCH 13 - PROGRESS: at 98.55% examples, 493562 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:45,801 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:06:45,842 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:06:45,847 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:06:45,871 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:06:45,871 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:06:45,873 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:06:45,886 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:06:45,892 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:06:45,892 : INFO : EPOCH - 13 : training on 23279529 raw words (22951015 effective words) took 46.5s, 493784 effective words/s
    2020-09-30 23:06:46,995 : INFO : EPOCH 14 - PROGRESS: at 2.37% examples, 501401 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:47,998 : INFO : EPOCH 14 - PROGRESS: at 4.67% examples, 515838 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:49,011 : INFO : EPOCH 14 - PROGRESS: at 6.75% examples, 500124 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:50,017 : INFO : EPOCH 14 - PROGRESS: at 8.90% examples, 499882 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:51,025 : INFO : EPOCH 14 - PROGRESS: at 10.96% examples, 497353 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:52,025 : INFO : EPOCH 14 - PROGRESS: at 13.06% examples, 493469 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:53,039 : INFO : EPOCH 14 - PROGRESS: at 15.21% examples, 492584 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:54,041 : INFO : EPOCH 14 - PROGRESS: at 17.58% examples, 500898 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:55,069 : INFO : EPOCH 14 - PROGRESS: at 19.99% examples, 506126 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:56,073 : INFO : EPOCH 14 - PROGRESS: at 22.23% examples, 505760 words/s, in_qsize 15, out_qsize 1
    2020-09-30 23:06:57,088 : INFO : EPOCH 14 - PROGRESS: at 24.71% examples, 509266 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:06:58,091 : INFO : EPOCH 14 - PROGRESS: at 26.90% examples, 509339 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:06:59,093 : INFO : EPOCH 14 - PROGRESS: at 29.05% examples, 508504 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:00,127 : INFO : EPOCH 14 - PROGRESS: at 31.26% examples, 506955 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:01,138 : INFO : EPOCH 14 - PROGRESS: at 33.62% examples, 508138 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:07:02,173 : INFO : EPOCH 14 - PROGRESS: at 36.09% examples, 510266 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:03,185 : INFO : EPOCH 14 - PROGRESS: at 38.49% examples, 512420 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:04,196 : INFO : EPOCH 14 - PROGRESS: at 40.60% examples, 510457 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:05,206 : INFO : EPOCH 14 - PROGRESS: at 42.49% examples, 506407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:06,227 : INFO : EPOCH 14 - PROGRESS: at 44.53% examples, 503776 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:07,240 : INFO : EPOCH 14 - PROGRESS: at 46.75% examples, 503383 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:08,245 : INFO : EPOCH 14 - PROGRESS: at 49.09% examples, 505028 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:09,251 : INFO : EPOCH 14 - PROGRESS: at 51.23% examples, 503596 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:10,280 : INFO : EPOCH 14 - PROGRESS: at 53.30% examples, 501874 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:11,325 : INFO : EPOCH 14 - PROGRESS: at 55.80% examples, 504060 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:12,338 : INFO : EPOCH 14 - PROGRESS: at 57.86% examples, 503118 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:13,358 : INFO : EPOCH 14 - PROGRESS: at 60.20% examples, 504416 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:07:14,364 : INFO : EPOCH 14 - PROGRESS: at 62.60% examples, 505689 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:15,432 : INFO : EPOCH 14 - PROGRESS: at 65.09% examples, 506409 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:16,442 : INFO : EPOCH 14 - PROGRESS: at 67.28% examples, 506148 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:17,466 : INFO : EPOCH 14 - PROGRESS: at 69.41% examples, 505402 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:18,470 : INFO : EPOCH 14 - PROGRESS: at 71.69% examples, 505882 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:19,476 : INFO : EPOCH 14 - PROGRESS: at 73.69% examples, 504858 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:20,489 : INFO : EPOCH 14 - PROGRESS: at 75.69% examples, 503490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:21,493 : INFO : EPOCH 14 - PROGRESS: at 77.83% examples, 502906 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:22,498 : INFO : EPOCH 14 - PROGRESS: at 79.92% examples, 502305 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:23,501 : INFO : EPOCH 14 - PROGRESS: at 82.25% examples, 502862 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:24,514 : INFO : EPOCH 14 - PROGRESS: at 84.31% examples, 501965 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:25,555 : INFO : EPOCH 14 - PROGRESS: at 86.51% examples, 501246 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:26,573 : INFO : EPOCH 14 - PROGRESS: at 88.75% examples, 501323 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:27,586 : INFO : EPOCH 14 - PROGRESS: at 91.17% examples, 502395 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:28,590 : INFO : EPOCH 14 - PROGRESS: at 93.32% examples, 501921 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:29,599 : INFO : EPOCH 14 - PROGRESS: at 95.58% examples, 501898 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:30,613 : INFO : EPOCH 14 - PROGRESS: at 97.70% examples, 501587 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:31,548 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:07:31,561 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:07:31,575 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:07:31,587 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:07:31,601 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:07:31,615 : INFO : EPOCH 14 - PROGRESS: at 99.91% examples, 501592 words/s, in_qsize 2, out_qsize 1
    2020-09-30 23:07:31,615 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:07:31,621 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:07:31,626 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:07:31,627 : INFO : EPOCH - 14 : training on 23279529 raw words (22951015 effective words) took 45.7s, 501871 effective words/s
    2020-09-30 23:07:32,638 : INFO : EPOCH 15 - PROGRESS: at 2.09% examples, 488539 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:33,642 : INFO : EPOCH 15 - PROGRESS: at 4.38% examples, 505310 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:34,655 : INFO : EPOCH 15 - PROGRESS: at 6.58% examples, 502207 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:07:35,656 : INFO : EPOCH 15 - PROGRESS: at 8.89% examples, 511723 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:36,670 : INFO : EPOCH 15 - PROGRESS: at 11.13% examples, 513878 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:37,671 : INFO : EPOCH 15 - PROGRESS: at 13.39% examples, 513540 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:38,734 : INFO : EPOCH 15 - PROGRESS: at 15.67% examples, 510319 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:39,746 : INFO : EPOCH 15 - PROGRESS: at 17.89% examples, 511120 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:40,753 : INFO : EPOCH 15 - PROGRESS: at 19.99% examples, 508903 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:41,757 : INFO : EPOCH 15 - PROGRESS: at 22.19% examples, 507359 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:42,795 : INFO : EPOCH 15 - PROGRESS: at 24.37% examples, 503494 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:43,813 : INFO : EPOCH 15 - PROGRESS: at 26.49% examples, 502660 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:07:44,814 : INFO : EPOCH 15 - PROGRESS: at 28.84% examples, 505394 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:45,818 : INFO : EPOCH 15 - PROGRESS: at 30.94% examples, 503706 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:46,820 : INFO : EPOCH 15 - PROGRESS: at 33.23% examples, 504130 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:47,826 : INFO : EPOCH 15 - PROGRESS: at 35.72% examples, 507490 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:07:48,831 : INFO : EPOCH 15 - PROGRESS: at 37.78% examples, 505385 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:07:49,837 : INFO : EPOCH 15 - PROGRESS: at 40.20% examples, 507872 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:50,840 : INFO : EPOCH 15 - PROGRESS: at 42.35% examples, 507011 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:51,845 : INFO : EPOCH 15 - PROGRESS: at 44.45% examples, 505702 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:52,868 : INFO : EPOCH 15 - PROGRESS: at 46.54% examples, 503603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:53,903 : INFO : EPOCH 15 - PROGRESS: at 48.88% examples, 504561 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:54,932 : INFO : EPOCH 15 - PROGRESS: at 51.10% examples, 503535 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:55,936 : INFO : EPOCH 15 - PROGRESS: at 53.13% examples, 501889 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:56,943 : INFO : EPOCH 15 - PROGRESS: at 55.41% examples, 502583 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:57,948 : INFO : EPOCH 15 - PROGRESS: at 57.63% examples, 503287 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:07:58,949 : INFO : EPOCH 15 - PROGRESS: at 59.95% examples, 504936 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:07:59,950 : INFO : EPOCH 15 - PROGRESS: at 62.39% examples, 506620 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:00,970 : INFO : EPOCH 15 - PROGRESS: at 64.55% examples, 505517 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:01,985 : INFO : EPOCH 15 - PROGRESS: at 66.91% examples, 506436 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:03,004 : INFO : EPOCH 15 - PROGRESS: at 69.06% examples, 505770 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:04,015 : INFO : EPOCH 15 - PROGRESS: at 71.24% examples, 505843 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:05,021 : INFO : EPOCH 15 - PROGRESS: at 73.20% examples, 504245 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:06,029 : INFO : EPOCH 15 - PROGRESS: at 75.27% examples, 503513 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:07,051 : INFO : EPOCH 15 - PROGRESS: at 77.42% examples, 502933 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:08,062 : INFO : EPOCH 15 - PROGRESS: at 79.65% examples, 502824 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:09,070 : INFO : EPOCH 15 - PROGRESS: at 81.91% examples, 503004 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:10,090 : INFO : EPOCH 15 - PROGRESS: at 84.08% examples, 502761 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:11,098 : INFO : EPOCH 15 - PROGRESS: at 86.47% examples, 503434 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:12,120 : INFO : EPOCH 15 - PROGRESS: at 88.89% examples, 504573 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:13,125 : INFO : EPOCH 15 - PROGRESS: at 91.31% examples, 505448 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:14,137 : INFO : EPOCH 15 - PROGRESS: at 93.40% examples, 504589 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:15,143 : INFO : EPOCH 15 - PROGRESS: at 95.90% examples, 505876 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:08:16,156 : INFO : EPOCH 15 - PROGRESS: at 97.87% examples, 504615 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:16,969 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:08:16,996 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:08:17,016 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:08:17,017 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:08:17,025 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:08:17,032 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:08:17,033 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:08:17,047 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:08:17,047 : INFO : EPOCH - 15 : training on 23279529 raw words (22951015 effective words) took 45.4s, 505339 effective words/s
    2020-09-30 23:08:18,083 : INFO : EPOCH 16 - PROGRESS: at 2.01% examples, 458709 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:19,085 : INFO : EPOCH 16 - PROGRESS: at 4.27% examples, 485413 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:20,091 : INFO : EPOCH 16 - PROGRESS: at 6.50% examples, 493328 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:21,094 : INFO : EPOCH 16 - PROGRESS: at 8.72% examples, 499947 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:22,109 : INFO : EPOCH 16 - PROGRESS: at 11.01% examples, 506167 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:23,120 : INFO : EPOCH 16 - PROGRESS: at 13.48% examples, 514301 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:24,161 : INFO : EPOCH 16 - PROGRESS: at 15.58% examples, 507073 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:25,165 : INFO : EPOCH 16 - PROGRESS: at 17.84% examples, 509965 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:08:26,188 : INFO : EPOCH 16 - PROGRESS: at 20.08% examples, 510232 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:08:27,189 : INFO : EPOCH 16 - PROGRESS: at 22.37% examples, 510534 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:08:28,224 : INFO : EPOCH 16 - PROGRESS: at 24.75% examples, 510981 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:29,234 : INFO : EPOCH 16 - PROGRESS: at 26.98% examples, 511380 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:30,234 : INFO : EPOCH 16 - PROGRESS: at 29.17% examples, 511244 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:31,248 : INFO : EPOCH 16 - PROGRESS: at 31.56% examples, 512950 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:32,252 : INFO : EPOCH 16 - PROGRESS: at 33.95% examples, 513986 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:33,254 : INFO : EPOCH 16 - PROGRESS: at 36.27% examples, 514974 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:34,293 : INFO : EPOCH 16 - PROGRESS: at 38.62% examples, 515481 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:35,311 : INFO : EPOCH 16 - PROGRESS: at 40.68% examples, 512632 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:36,318 : INFO : EPOCH 16 - PROGRESS: at 42.75% examples, 510529 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:37,324 : INFO : EPOCH 16 - PROGRESS: at 45.23% examples, 513323 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:38,348 : INFO : EPOCH 16 - PROGRESS: at 47.52% examples, 513171 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:08:39,349 : INFO : EPOCH 16 - PROGRESS: at 49.93% examples, 514427 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:40,364 : INFO : EPOCH 16 - PROGRESS: at 52.15% examples, 513671 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:41,378 : INFO : EPOCH 16 - PROGRESS: at 54.46% examples, 514173 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:42,382 : INFO : EPOCH 16 - PROGRESS: at 56.64% examples, 513671 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:43,415 : INFO : EPOCH 16 - PROGRESS: at 59.03% examples, 514834 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:44,429 : INFO : EPOCH 16 - PROGRESS: at 61.13% examples, 513776 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:08:45,470 : INFO : EPOCH 16 - PROGRESS: at 63.47% examples, 513351 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:46,471 : INFO : EPOCH 16 - PROGRESS: at 65.61% examples, 512356 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:47,472 : INFO : EPOCH 16 - PROGRESS: at 67.83% examples, 512358 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:48,490 : INFO : EPOCH 16 - PROGRESS: at 70.02% examples, 511834 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:08:49,500 : INFO : EPOCH 16 - PROGRESS: at 72.40% examples, 512897 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:50,516 : INFO : EPOCH 16 - PROGRESS: at 74.48% examples, 512074 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:08:51,546 : INFO : EPOCH 16 - PROGRESS: at 76.67% examples, 511385 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:52,553 : INFO : EPOCH 16 - PROGRESS: at 78.77% examples, 510251 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:53,562 : INFO : EPOCH 16 - PROGRESS: at 81.17% examples, 511288 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:54,570 : INFO : EPOCH 16 - PROGRESS: at 83.44% examples, 511476 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:55,588 : INFO : EPOCH 16 - PROGRESS: at 85.72% examples, 511324 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:56,627 : INFO : EPOCH 16 - PROGRESS: at 88.05% examples, 511344 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:57,643 : INFO : EPOCH 16 - PROGRESS: at 90.37% examples, 511645 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:58,671 : INFO : EPOCH 16 - PROGRESS: at 92.80% examples, 512064 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:08:59,713 : INFO : EPOCH 16 - PROGRESS: at 95.11% examples, 511856 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:00,722 : INFO : EPOCH 16 - PROGRESS: at 97.55% examples, 512716 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:01,696 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:09:01,706 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:09:01,718 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:09:01,726 : INFO : EPOCH 16 - PROGRESS: at 99.81% examples, 512875 words/s, in_qsize 4, out_qsize 1
    2020-09-30 23:09:01,726 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:09:01,734 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:09:01,745 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:09:01,755 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:09:01,756 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:09:01,757 : INFO : EPOCH - 16 : training on 23279529 raw words (22951015 effective words) took 44.7s, 513374 effective words/s
    2020-09-30 23:09:02,771 : INFO : EPOCH 17 - PROGRESS: at 2.09% examples, 487384 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:03,786 : INFO : EPOCH 17 - PROGRESS: at 4.43% examples, 506745 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:04,872 : INFO : EPOCH 17 - PROGRESS: at 6.90% examples, 513106 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:09:05,907 : INFO : EPOCH 17 - PROGRESS: at 9.07% examples, 508349 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:06,922 : INFO : EPOCH 17 - PROGRESS: at 11.42% examples, 514917 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:07,968 : INFO : EPOCH 17 - PROGRESS: at 13.56% examples, 506022 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:08,969 : INFO : EPOCH 17 - PROGRESS: at 15.83% examples, 508233 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:09,979 : INFO : EPOCH 17 - PROGRESS: at 17.97% examples, 507086 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:10,982 : INFO : EPOCH 17 - PROGRESS: at 20.50% examples, 516068 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:11,984 : INFO : EPOCH 17 - PROGRESS: at 22.70% examples, 513809 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:09:12,990 : INFO : EPOCH 17 - PROGRESS: at 25.28% examples, 519490 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:13,998 : INFO : EPOCH 17 - PROGRESS: at 27.49% examples, 518580 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:15,016 : INFO : EPOCH 17 - PROGRESS: at 29.64% examples, 516574 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:16,022 : INFO : EPOCH 17 - PROGRESS: at 31.95% examples, 516732 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:17,031 : INFO : EPOCH 17 - PROGRESS: at 34.40% examples, 518604 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:18,048 : INFO : EPOCH 17 - PROGRESS: at 36.74% examples, 518882 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:19,072 : INFO : EPOCH 17 - PROGRESS: at 38.79% examples, 515644 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:09:20,075 : INFO : EPOCH 17 - PROGRESS: at 40.88% examples, 513777 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:21,114 : INFO : EPOCH 17 - PROGRESS: at 43.00% examples, 511274 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:22,124 : INFO : EPOCH 17 - PROGRESS: at 45.23% examples, 511079 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:23,129 : INFO : EPOCH 17 - PROGRESS: at 47.58% examples, 511910 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:24,135 : INFO : EPOCH 17 - PROGRESS: at 49.69% examples, 510490 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:09:25,157 : INFO : EPOCH 17 - PROGRESS: at 51.91% examples, 509413 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:26,160 : INFO : EPOCH 17 - PROGRESS: at 54.05% examples, 508630 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:27,177 : INFO : EPOCH 17 - PROGRESS: at 56.34% examples, 509282 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:28,186 : INFO : EPOCH 17 - PROGRESS: at 58.73% examples, 511100 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:29,202 : INFO : EPOCH 17 - PROGRESS: at 61.05% examples, 511897 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:09:30,211 : INFO : EPOCH 17 - PROGRESS: at 63.42% examples, 512444 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:09:31,246 : INFO : EPOCH 17 - PROGRESS: at 65.64% examples, 511544 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:32,252 : INFO : EPOCH 17 - PROGRESS: at 67.87% examples, 511471 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:33,266 : INFO : EPOCH 17 - PROGRESS: at 70.23% examples, 512307 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:09:34,268 : INFO : EPOCH 17 - PROGRESS: at 72.36% examples, 511682 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:35,272 : INFO : EPOCH 17 - PROGRESS: at 74.34% examples, 510481 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:36,276 : INFO : EPOCH 17 - PROGRESS: at 76.67% examples, 511089 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:37,297 : INFO : EPOCH 17 - PROGRESS: at 78.76% examples, 509783 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:09:38,326 : INFO : EPOCH 17 - PROGRESS: at 80.55% examples, 506806 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:39,340 : INFO : EPOCH 17 - PROGRESS: at 82.51% examples, 504728 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:40,357 : INFO : EPOCH 17 - PROGRESS: at 84.71% examples, 504489 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:41,357 : INFO : EPOCH 17 - PROGRESS: at 86.92% examples, 504488 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:42,379 : INFO : EPOCH 17 - PROGRESS: at 89.18% examples, 504637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:43,413 : INFO : EPOCH 17 - PROGRESS: at 91.59% examples, 505172 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:44,419 : INFO : EPOCH 17 - PROGRESS: at 93.81% examples, 505053 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:45,423 : INFO : EPOCH 17 - PROGRESS: at 96.28% examples, 506150 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:46,426 : INFO : EPOCH 17 - PROGRESS: at 98.58% examples, 506706 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:46,935 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:09:46,952 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:09:46,958 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:09:46,964 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:09:46,964 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:09:46,993 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:09:47,004 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:09:47,005 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:09:47,006 : INFO : EPOCH - 17 : training on 23279529 raw words (22951015 effective words) took 45.2s, 507256 effective words/s
    2020-09-30 23:09:48,014 : INFO : EPOCH 18 - PROGRESS: at 2.01% examples, 471454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:49,023 : INFO : EPOCH 18 - PROGRESS: at 4.19% examples, 480747 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:50,199 : INFO : EPOCH 18 - PROGRESS: at 6.03% examples, 437065 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:51,223 : INFO : EPOCH 18 - PROGRESS: at 7.51% examples, 413217 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:52,225 : INFO : EPOCH 18 - PROGRESS: at 9.37% examples, 417051 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:53,231 : INFO : EPOCH 18 - PROGRESS: at 11.17% examples, 417822 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:54,237 : INFO : EPOCH 18 - PROGRESS: at 13.02% examples, 417087 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:55,270 : INFO : EPOCH 18 - PROGRESS: at 15.09% examples, 422362 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:56,275 : INFO : EPOCH 18 - PROGRESS: at 16.75% examples, 419486 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:57,308 : INFO : EPOCH 18 - PROGRESS: at 18.41% examples, 414143 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:58,320 : INFO : EPOCH 18 - PROGRESS: at 19.89% examples, 407919 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:09:59,336 : INFO : EPOCH 18 - PROGRESS: at 21.88% examples, 411291 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:00,381 : INFO : EPOCH 18 - PROGRESS: at 23.22% examples, 401526 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:01,383 : INFO : EPOCH 18 - PROGRESS: at 24.98% examples, 401214 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:02,392 : INFO : EPOCH 18 - PROGRESS: at 26.72% examples, 401231 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:03,403 : INFO : EPOCH 18 - PROGRESS: at 28.61% examples, 402913 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:10:04,410 : INFO : EPOCH 18 - PROGRESS: at 30.27% examples, 401835 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:05,415 : INFO : EPOCH 18 - PROGRESS: at 32.18% examples, 403455 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:06,420 : INFO : EPOCH 18 - PROGRESS: at 33.19% examples, 394005 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:07,427 : INFO : EPOCH 18 - PROGRESS: at 34.27% examples, 386430 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:08,433 : INFO : EPOCH 18 - PROGRESS: at 35.16% examples, 377794 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:09,479 : INFO : EPOCH 18 - PROGRESS: at 36.40% examples, 372657 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:10,484 : INFO : EPOCH 18 - PROGRESS: at 37.41% examples, 366603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:11,499 : INFO : EPOCH 18 - PROGRESS: at 38.33% examples, 360141 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:12,512 : INFO : EPOCH 18 - PROGRESS: at 39.56% examples, 356842 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:13,597 : INFO : EPOCH 18 - PROGRESS: at 40.72% examples, 352447 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:14,639 : INFO : EPOCH 18 - PROGRESS: at 41.83% examples, 348287 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:15,740 : INFO : EPOCH 18 - PROGRESS: at 42.92% examples, 343731 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:16,806 : INFO : EPOCH 18 - PROGRESS: at 43.86% examples, 338505 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:10:17,820 : INFO : EPOCH 18 - PROGRESS: at 44.97% examples, 335872 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:18,843 : INFO : EPOCH 18 - PROGRESS: at 46.38% examples, 334792 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:19,857 : INFO : EPOCH 18 - PROGRESS: at 47.61% examples, 333317 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:20,883 : INFO : EPOCH 18 - PROGRESS: at 48.97% examples, 332339 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:21,935 : INFO : EPOCH 18 - PROGRESS: at 50.44% examples, 331771 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:22,966 : INFO : EPOCH 18 - PROGRESS: at 51.95% examples, 331717 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:24,036 : INFO : EPOCH 18 - PROGRESS: at 53.39% examples, 331009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:25,049 : INFO : EPOCH 18 - PROGRESS: at 54.43% examples, 328554 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:26,064 : INFO : EPOCH 18 - PROGRESS: at 55.76% examples, 327950 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:27,065 : INFO : EPOCH 18 - PROGRESS: at 56.96% examples, 326778 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:28,073 : INFO : EPOCH 18 - PROGRESS: at 58.16% examples, 325626 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:29,081 : INFO : EPOCH 18 - PROGRESS: at 59.42% examples, 324887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:30,095 : INFO : EPOCH 18 - PROGRESS: at 60.62% examples, 323772 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:31,144 : INFO : EPOCH 18 - PROGRESS: at 62.04% examples, 323308 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:32,394 : INFO : EPOCH 18 - PROGRESS: at 63.30% examples, 320590 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:33,474 : INFO : EPOCH 18 - PROGRESS: at 64.55% examples, 319201 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:34,505 : INFO : EPOCH 18 - PROGRESS: at 65.72% examples, 317981 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:10:35,587 : INFO : EPOCH 18 - PROGRESS: at 66.67% examples, 315256 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:36,615 : INFO : EPOCH 18 - PROGRESS: at 67.62% examples, 313221 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:10:37,682 : INFO : EPOCH 18 - PROGRESS: at 68.78% examples, 311803 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:38,695 : INFO : EPOCH 18 - PROGRESS: at 69.86% examples, 310582 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:10:39,781 : INFO : EPOCH 18 - PROGRESS: at 70.86% examples, 308594 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:40,835 : INFO : EPOCH 18 - PROGRESS: at 71.56% examples, 305603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:41,856 : INFO : EPOCH 18 - PROGRESS: at 72.77% examples, 305035 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:42,913 : INFO : EPOCH 18 - PROGRESS: at 73.89% examples, 304116 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:43,967 : INFO : EPOCH 18 - PROGRESS: at 75.02% examples, 303062 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:44,986 : INFO : EPOCH 18 - PROGRESS: at 76.15% examples, 302265 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:46,021 : INFO : EPOCH 18 - PROGRESS: at 76.94% examples, 300066 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:10:47,048 : INFO : EPOCH 18 - PROGRESS: at 77.89% examples, 298336 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:48,059 : INFO : EPOCH 18 - PROGRESS: at 78.69% examples, 296417 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:49,070 : INFO : EPOCH 18 - PROGRESS: at 79.77% examples, 295644 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:50,127 : INFO : EPOCH 18 - PROGRESS: at 80.73% examples, 294215 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:51,230 : INFO : EPOCH 18 - PROGRESS: at 81.75% examples, 292636 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:52,234 : INFO : EPOCH 18 - PROGRESS: at 82.47% examples, 290650 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:53,308 : INFO : EPOCH 18 - PROGRESS: at 83.28% examples, 288867 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:54,349 : INFO : EPOCH 18 - PROGRESS: at 84.26% examples, 287717 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:55,427 : INFO : EPOCH 18 - PROGRESS: at 85.26% examples, 286438 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:56,443 : INFO : EPOCH 18 - PROGRESS: at 86.12% examples, 285049 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:57,474 : INFO : EPOCH 18 - PROGRESS: at 86.95% examples, 283608 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:58,522 : INFO : EPOCH 18 - PROGRESS: at 88.09% examples, 283120 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:10:59,545 : INFO : EPOCH 18 - PROGRESS: at 89.05% examples, 282185 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:00,593 : INFO : EPOCH 18 - PROGRESS: at 90.06% examples, 281318 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:01,674 : INFO : EPOCH 18 - PROGRESS: at 91.17% examples, 280517 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:02,712 : INFO : EPOCH 18 - PROGRESS: at 92.18% examples, 279737 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:03,747 : INFO : EPOCH 18 - PROGRESS: at 92.93% examples, 278111 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:04,765 : INFO : EPOCH 18 - PROGRESS: at 93.75% examples, 276835 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:05,809 : INFO : EPOCH 18 - PROGRESS: at 94.46% examples, 275269 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:06,812 : INFO : EPOCH 18 - PROGRESS: at 95.53% examples, 274733 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:07,899 : INFO : EPOCH 18 - PROGRESS: at 96.19% examples, 272964 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:11:09,046 : INFO : EPOCH 18 - PROGRESS: at 96.87% examples, 271045 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:10,126 : INFO : EPOCH 18 - PROGRESS: at 97.92% examples, 270431 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:11,213 : INFO : EPOCH 18 - PROGRESS: at 98.97% examples, 269813 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:11,940 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:11:11,944 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:11:11,949 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:11:12,011 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:11:12,030 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:11:12,033 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:11:12,049 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:11:12,068 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:11:12,069 : INFO : EPOCH - 18 : training on 23279529 raw words (22951015 effective words) took 85.1s, 269818 effective words/s
    2020-09-30 23:11:13,082 : INFO : EPOCH 19 - PROGRESS: at 0.86% examples, 203295 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:14,132 : INFO : EPOCH 19 - PROGRESS: at 1.78% examples, 202215 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:15,193 : INFO : EPOCH 19 - PROGRESS: at 2.91% examples, 217435 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:16,221 : INFO : EPOCH 19 - PROGRESS: at 3.86% examples, 217234 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:17,388 : INFO : EPOCH 19 - PROGRESS: at 4.47% examples, 195142 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:18,401 : INFO : EPOCH 19 - PROGRESS: at 5.32% examples, 195981 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:19,442 : INFO : EPOCH 19 - PROGRESS: at 5.85% examples, 184040 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:20,535 : INFO : EPOCH 19 - PROGRESS: at 6.54% examples, 178521 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:21,632 : INFO : EPOCH 19 - PROGRESS: at 7.35% examples, 178253 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:22,687 : INFO : EPOCH 19 - PROGRESS: at 8.34% examples, 182239 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:23,690 : INFO : EPOCH 19 - PROGRESS: at 9.32% examples, 186454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:24,714 : INFO : EPOCH 19 - PROGRESS: at 10.02% examples, 184291 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:11:25,800 : INFO : EPOCH 19 - PROGRESS: at 10.66% examples, 180962 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:26,812 : INFO : EPOCH 19 - PROGRESS: at 11.55% examples, 182346 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:11:27,819 : INFO : EPOCH 19 - PROGRESS: at 12.62% examples, 185972 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:28,822 : INFO : EPOCH 19 - PROGRESS: at 13.43% examples, 185844 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:29,822 : INFO : EPOCH 19 - PROGRESS: at 14.33% examples, 186807 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:11:30,873 : INFO : EPOCH 19 - PROGRESS: at 15.54% examples, 191307 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:31,873 : INFO : EPOCH 19 - PROGRESS: at 16.68% examples, 195364 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:32,956 : INFO : EPOCH 19 - PROGRESS: at 17.62% examples, 195867 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:33,994 : INFO : EPOCH 19 - PROGRESS: at 18.57% examples, 196373 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:35,006 : INFO : EPOCH 19 - PROGRESS: at 19.34% examples, 195327 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:36,019 : INFO : EPOCH 19 - PROGRESS: at 20.28% examples, 196735 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:11:37,038 : INFO : EPOCH 19 - PROGRESS: at 21.06% examples, 195726 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:38,064 : INFO : EPOCH 19 - PROGRESS: at 21.98% examples, 195825 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:39,126 : INFO : EPOCH 19 - PROGRESS: at 22.74% examples, 194549 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:40,160 : INFO : EPOCH 19 - PROGRESS: at 23.62% examples, 194274 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:41,160 : INFO : EPOCH 19 - PROGRESS: at 24.41% examples, 193623 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:42,185 : INFO : EPOCH 19 - PROGRESS: at 25.51% examples, 195670 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:43,204 : INFO : EPOCH 19 - PROGRESS: at 26.29% examples, 195170 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:44,237 : INFO : EPOCH 19 - PROGRESS: at 27.20% examples, 195236 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:45,345 : INFO : EPOCH 19 - PROGRESS: at 27.99% examples, 194235 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:46,413 : INFO : EPOCH 19 - PROGRESS: at 29.01% examples, 195178 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:47,455 : INFO : EPOCH 19 - PROGRESS: at 29.94% examples, 195470 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:48,511 : INFO : EPOCH 19 - PROGRESS: at 30.95% examples, 196161 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:49,559 : INFO : EPOCH 19 - PROGRESS: at 32.16% examples, 197873 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:11:50,610 : INFO : EPOCH 19 - PROGRESS: at 33.19% examples, 198477 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:51,627 : INFO : EPOCH 19 - PROGRESS: at 34.23% examples, 199257 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:52,747 : INFO : EPOCH 19 - PROGRESS: at 35.40% examples, 200413 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:53,751 : INFO : EPOCH 19 - PROGRESS: at 36.52% examples, 201628 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:54,757 : INFO : EPOCH 19 - PROGRESS: at 37.55% examples, 202324 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:55,838 : INFO : EPOCH 19 - PROGRESS: at 38.70% examples, 203551 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:56,847 : INFO : EPOCH 19 - PROGRESS: at 39.70% examples, 204146 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:57,873 : INFO : EPOCH 19 - PROGRESS: at 40.64% examples, 204195 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:11:58,949 : INFO : EPOCH 19 - PROGRESS: at 41.61% examples, 204271 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:11:59,954 : INFO : EPOCH 19 - PROGRESS: at 42.71% examples, 205260 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:12:00,960 : INFO : EPOCH 19 - PROGRESS: at 43.90% examples, 206556 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:01,991 : INFO : EPOCH 19 - PROGRESS: at 44.88% examples, 206942 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:03,000 : INFO : EPOCH 19 - PROGRESS: at 45.61% examples, 206068 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:12:04,041 : INFO : EPOCH 19 - PROGRESS: at 46.43% examples, 205285 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:05,145 : INFO : EPOCH 19 - PROGRESS: at 47.32% examples, 205035 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:06,184 : INFO : EPOCH 19 - PROGRESS: at 48.30% examples, 205203 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:07,209 : INFO : EPOCH 19 - PROGRESS: at 49.09% examples, 204726 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:08,257 : INFO : EPOCH 19 - PROGRESS: at 49.88% examples, 204003 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:09,442 : INFO : EPOCH 19 - PROGRESS: at 50.70% examples, 203017 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:10,449 : INFO : EPOCH 19 - PROGRESS: at 51.62% examples, 203007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:11,466 : INFO : EPOCH 19 - PROGRESS: at 52.37% examples, 202454 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:12,525 : INFO : EPOCH 19 - PROGRESS: at 53.35% examples, 202599 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:13,542 : INFO : EPOCH 19 - PROGRESS: at 54.39% examples, 203183 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:14,550 : INFO : EPOCH 19 - PROGRESS: at 55.45% examples, 203789 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:15,566 : INFO : EPOCH 19 - PROGRESS: at 56.47% examples, 204340 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:16,568 : INFO : EPOCH 19 - PROGRESS: at 57.38% examples, 204479 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:17,622 : INFO : EPOCH 19 - PROGRESS: at 58.39% examples, 204880 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:18,683 : INFO : EPOCH 19 - PROGRESS: at 59.11% examples, 204070 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:19,685 : INFO : EPOCH 19 - PROGRESS: at 60.11% examples, 204609 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:20,709 : INFO : EPOCH 19 - PROGRESS: at 61.27% examples, 205374 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:21,717 : INFO : EPOCH 19 - PROGRESS: at 62.26% examples, 205603 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:22,718 : INFO : EPOCH 19 - PROGRESS: at 63.20% examples, 205706 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:12:23,760 : INFO : EPOCH 19 - PROGRESS: at 64.09% examples, 205543 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:24,827 : INFO : EPOCH 19 - PROGRESS: at 64.77% examples, 204542 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:25,971 : INFO : EPOCH 19 - PROGRESS: at 65.56% examples, 203859 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:27,015 : INFO : EPOCH 19 - PROGRESS: at 66.33% examples, 203334 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:28,041 : INFO : EPOCH 19 - PROGRESS: at 67.03% examples, 202753 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:29,220 : INFO : EPOCH 19 - PROGRESS: at 67.75% examples, 201792 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:30,356 : INFO : EPOCH 19 - PROGRESS: at 68.43% examples, 200847 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:31,429 : INFO : EPOCH 19 - PROGRESS: at 68.98% examples, 199724 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:32,437 : INFO : EPOCH 19 - PROGRESS: at 69.45% examples, 198668 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:33,506 : INFO : EPOCH 19 - PROGRESS: at 70.40% examples, 198683 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:34,671 : INFO : EPOCH 19 - PROGRESS: at 70.81% examples, 197054 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:12:35,832 : INFO : EPOCH 19 - PROGRESS: at 71.38% examples, 195934 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:36,984 : INFO : EPOCH 19 - PROGRESS: at 71.99% examples, 194877 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:37,986 : INFO : EPOCH 19 - PROGRESS: at 72.52% examples, 194071 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:39,014 : INFO : EPOCH 19 - PROGRESS: at 72.89% examples, 192775 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:40,174 : INFO : EPOCH 19 - PROGRESS: at 73.33% examples, 191451 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:41,309 : INFO : EPOCH 19 - PROGRESS: at 73.77% examples, 190205 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:42,388 : INFO : EPOCH 19 - PROGRESS: at 74.27% examples, 189217 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:43,414 : INFO : EPOCH 19 - PROGRESS: at 74.65% examples, 188046 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:44,540 : INFO : EPOCH 19 - PROGRESS: at 75.13% examples, 187007 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:45,593 : INFO : EPOCH 19 - PROGRESS: at 75.78% examples, 186459 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:46,689 : INFO : EPOCH 19 - PROGRESS: at 76.32% examples, 185637 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:47,692 : INFO : EPOCH 19 - PROGRESS: at 77.03% examples, 185401 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:48,715 : INFO : EPOCH 19 - PROGRESS: at 77.87% examples, 185347 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:49,733 : INFO : EPOCH 19 - PROGRESS: at 78.65% examples, 185206 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:50,744 : INFO : EPOCH 19 - PROGRESS: at 79.42% examples, 185075 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:51,756 : INFO : EPOCH 19 - PROGRESS: at 80.25% examples, 185224 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:52,766 : INFO : EPOCH 19 - PROGRESS: at 81.04% examples, 185111 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:12:53,829 : INFO : EPOCH 19 - PROGRESS: at 82.08% examples, 185461 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:54,851 : INFO : EPOCH 19 - PROGRESS: at 82.55% examples, 184649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:55,901 : INFO : EPOCH 19 - PROGRESS: at 83.49% examples, 184927 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:56,916 : INFO : EPOCH 19 - PROGRESS: at 84.31% examples, 184900 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:57,933 : INFO : EPOCH 19 - PROGRESS: at 85.05% examples, 184675 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:12:58,995 : INFO : EPOCH 19 - PROGRESS: at 85.81% examples, 184484 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:00,166 : INFO : EPOCH 19 - PROGRESS: at 86.42% examples, 183739 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:01,174 : INFO : EPOCH 19 - PROGRESS: at 87.20% examples, 183718 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:13:02,204 : INFO : EPOCH 19 - PROGRESS: at 87.84% examples, 183318 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:03,206 : INFO : EPOCH 19 - PROGRESS: at 88.53% examples, 183060 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:04,247 : INFO : EPOCH 19 - PROGRESS: at 89.19% examples, 182735 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:05,253 : INFO : EPOCH 19 - PROGRESS: at 89.74% examples, 182221 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:06,379 : INFO : EPOCH 19 - PROGRESS: at 90.37% examples, 181703 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:07,649 : INFO : EPOCH 19 - PROGRESS: at 91.12% examples, 181143 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:08,803 : INFO : EPOCH 19 - PROGRESS: at 91.72% examples, 180517 words/s, in_qsize 16, out_qsize 1
    2020-09-30 23:13:09,810 : INFO : EPOCH 19 - PROGRESS: at 92.51% examples, 180451 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:10,853 : INFO : EPOCH 19 - PROGRESS: at 93.23% examples, 180252 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:11,866 : INFO : EPOCH 19 - PROGRESS: at 93.86% examples, 179940 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:12,895 : INFO : EPOCH 19 - PROGRESS: at 94.56% examples, 179700 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:13,945 : INFO : EPOCH 19 - PROGRESS: at 95.35% examples, 179583 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:14,959 : INFO : EPOCH 19 - PROGRESS: at 96.24% examples, 179764 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:15,986 : INFO : EPOCH 19 - PROGRESS: at 96.87% examples, 179453 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:17,005 : INFO : EPOCH 19 - PROGRESS: at 97.62% examples, 179386 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:18,012 : INFO : EPOCH 19 - PROGRESS: at 98.51% examples, 179563 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:19,025 : INFO : EPOCH 19 - PROGRESS: at 99.45% examples, 179809 words/s, in_qsize 13, out_qsize 0
    2020-09-30 23:13:19,445 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:13:19,463 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:13:19,464 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:13:19,468 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:13:19,572 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:13:19,645 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:13:19,654 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:13:19,661 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:13:19,662 : INFO : EPOCH - 19 : training on 23279529 raw words (22951015 effective words) took 127.6s, 179887 effective words/s
    2020-09-30 23:13:21,047 : INFO : EPOCH 20 - PROGRESS: at 0.70% examples, 120671 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:13:22,048 : INFO : EPOCH 20 - PROGRESS: at 1.47% examples, 146590 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:23,065 : INFO : EPOCH 20 - PROGRESS: at 2.27% examples, 156740 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:24,093 : INFO : EPOCH 20 - PROGRESS: at 2.91% examples, 153298 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:25,187 : INFO : EPOCH 20 - PROGRESS: at 3.61% examples, 152704 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:26,299 : INFO : EPOCH 20 - PROGRESS: at 4.27% examples, 149022 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:13:27,340 : INFO : EPOCH 20 - PROGRESS: at 4.95% examples, 150333 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:28,362 : INFO : EPOCH 20 - PROGRESS: at 5.63% examples, 150425 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:29,412 : INFO : EPOCH 20 - PROGRESS: at 6.40% examples, 152050 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:30,430 : INFO : EPOCH 20 - PROGRESS: at 6.95% examples, 149326 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:13:31,532 : INFO : EPOCH 20 - PROGRESS: at 7.56% examples, 147666 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:13:32,577 : INFO : EPOCH 20 - PROGRESS: at 8.14% examples, 146135 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:33,618 : INFO : EPOCH 20 - PROGRESS: at 8.67% examples, 144263 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:34,892 : INFO : EPOCH 20 - PROGRESS: at 9.20% examples, 140407 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:36,012 : INFO : EPOCH 20 - PROGRESS: at 9.82% examples, 139559 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:37,099 : INFO : EPOCH 20 - PROGRESS: at 10.49% examples, 140272 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:38,183 : INFO : EPOCH 20 - PROGRESS: at 11.17% examples, 140458 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:39,330 : INFO : EPOCH 20 - PROGRESS: at 11.81% examples, 139628 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:40,430 : INFO : EPOCH 20 - PROGRESS: at 12.41% examples, 138758 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:41,507 : INFO : EPOCH 20 - PROGRESS: at 12.98% examples, 137618 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:42,568 : INFO : EPOCH 20 - PROGRESS: at 13.69% examples, 138480 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:43,624 : INFO : EPOCH 20 - PROGRESS: at 14.29% examples, 138024 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:44,759 : INFO : EPOCH 20 - PROGRESS: at 14.96% examples, 137916 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:45,810 : INFO : EPOCH 20 - PROGRESS: at 15.67% examples, 138694 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:46,812 : INFO : EPOCH 20 - PROGRESS: at 16.40% examples, 139985 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:47,837 : INFO : EPOCH 20 - PROGRESS: at 17.04% examples, 140406 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:48,886 : INFO : EPOCH 20 - PROGRESS: at 17.70% examples, 140655 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:49,943 : INFO : EPOCH 20 - PROGRESS: at 18.45% examples, 141230 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:13:50,964 : INFO : EPOCH 20 - PROGRESS: at 19.17% examples, 141900 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:13:52,008 : INFO : EPOCH 20 - PROGRESS: at 19.89% examples, 142687 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:53,038 : INFO : EPOCH 20 - PROGRESS: at 20.63% examples, 143520 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:54,045 : INFO : EPOCH 20 - PROGRESS: at 21.20% examples, 142984 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:55,059 : INFO : EPOCH 20 - PROGRESS: at 21.89% examples, 143270 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:13:56,113 : INFO : EPOCH 20 - PROGRESS: at 22.78% examples, 144682 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:57,155 : INFO : EPOCH 20 - PROGRESS: at 23.50% examples, 144781 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:58,162 : INFO : EPOCH 20 - PROGRESS: at 24.37% examples, 146055 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:13:59,171 : INFO : EPOCH 20 - PROGRESS: at 25.06% examples, 146496 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:00,197 : INFO : EPOCH 20 - PROGRESS: at 25.76% examples, 146814 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:01,217 : INFO : EPOCH 20 - PROGRESS: at 26.49% examples, 147401 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:02,289 : INFO : EPOCH 20 - PROGRESS: at 27.45% examples, 148692 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:03,315 : INFO : EPOCH 20 - PROGRESS: at 28.19% examples, 149153 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:04,343 : INFO : EPOCH 20 - PROGRESS: at 29.01% examples, 150027 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:05,371 : INFO : EPOCH 20 - PROGRESS: at 29.73% examples, 150268 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:06,450 : INFO : EPOCH 20 - PROGRESS: at 30.36% examples, 149898 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:07,464 : INFO : EPOCH 20 - PROGRESS: at 31.26% examples, 150957 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:08,597 : INFO : EPOCH 20 - PROGRESS: at 31.94% examples, 150635 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:09,696 : INFO : EPOCH 20 - PROGRESS: at 32.62% examples, 150384 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:10,810 : INFO : EPOCH 20 - PROGRESS: at 33.49% examples, 150887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:11,818 : INFO : EPOCH 20 - PROGRESS: at 34.44% examples, 152061 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:12,959 : INFO : EPOCH 20 - PROGRESS: at 34.98% examples, 151170 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:14,032 : INFO : EPOCH 20 - PROGRESS: at 35.57% examples, 150665 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:15,056 : INFO : EPOCH 20 - PROGRESS: at 36.14% examples, 150144 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:16,094 : INFO : EPOCH 20 - PROGRESS: at 36.79% examples, 149962 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:17,144 : INFO : EPOCH 20 - PROGRESS: at 37.33% examples, 149410 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:18,165 : INFO : EPOCH 20 - PROGRESS: at 37.90% examples, 149134 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:19,195 : INFO : EPOCH 20 - PROGRESS: at 38.62% examples, 149325 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:20,227 : INFO : EPOCH 20 - PROGRESS: at 39.38% examples, 149662 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:21,229 : INFO : EPOCH 20 - PROGRESS: at 39.93% examples, 149270 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:22,311 : INFO : EPOCH 20 - PROGRESS: at 40.52% examples, 148832 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:23,312 : INFO : EPOCH 20 - PROGRESS: at 41.09% examples, 148632 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:24,333 : INFO : EPOCH 20 - PROGRESS: at 41.74% examples, 148527 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:25,433 : INFO : EPOCH 20 - PROGRESS: at 42.22% examples, 147670 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:26,444 : INFO : EPOCH 20 - PROGRESS: at 42.70% examples, 147170 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:27,482 : INFO : EPOCH 20 - PROGRESS: at 43.47% examples, 147475 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:28,498 : INFO : EPOCH 20 - PROGRESS: at 44.21% examples, 147685 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:29,595 : INFO : EPOCH 20 - PROGRESS: at 44.64% examples, 146896 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:30,753 : INFO : EPOCH 20 - PROGRESS: at 45.66% examples, 147760 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:31,884 : INFO : EPOCH 20 - PROGRESS: at 46.70% examples, 148657 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:32,906 : INFO : EPOCH 20 - PROGRESS: at 47.37% examples, 148713 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:33,915 : INFO : EPOCH 20 - PROGRESS: at 47.98% examples, 148644 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:34,999 : INFO : EPOCH 20 - PROGRESS: at 48.55% examples, 148173 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:36,028 : INFO : EPOCH 20 - PROGRESS: at 49.14% examples, 147954 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:37,144 : INFO : EPOCH 20 - PROGRESS: at 49.69% examples, 147438 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:38,200 : INFO : EPOCH 20 - PROGRESS: at 50.10% examples, 146571 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:39,220 : INFO : EPOCH 20 - PROGRESS: at 50.71% examples, 146408 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:40,246 : INFO : EPOCH 20 - PROGRESS: at 51.19% examples, 145867 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:41,249 : INFO : EPOCH 20 - PROGRESS: at 51.60% examples, 145261 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:42,419 : INFO : EPOCH 20 - PROGRESS: at 52.15% examples, 144725 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:43,614 : INFO : EPOCH 20 - PROGRESS: at 52.83% examples, 144510 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:44,640 : INFO : EPOCH 20 - PROGRESS: at 53.46% examples, 144474 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:45,726 : INFO : EPOCH 20 - PROGRESS: at 54.00% examples, 144106 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:46,924 : INFO : EPOCH 20 - PROGRESS: at 54.55% examples, 143584 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:48,203 : INFO : EPOCH 20 - PROGRESS: at 55.24% examples, 143259 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:49,260 : INFO : EPOCH 20 - PROGRESS: at 55.84% examples, 143185 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:50,266 : INFO : EPOCH 20 - PROGRESS: at 56.21% examples, 142560 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:51,301 : INFO : EPOCH 20 - PROGRESS: at 56.84% examples, 142538 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:52,340 : INFO : EPOCH 20 - PROGRESS: at 57.22% examples, 141887 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:53,409 : INFO : EPOCH 20 - PROGRESS: at 57.79% examples, 141725 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:54,447 : INFO : EPOCH 20 - PROGRESS: at 58.24% examples, 141291 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:55,497 : INFO : EPOCH 20 - PROGRESS: at 58.60% examples, 140649 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:56,548 : INFO : EPOCH 20 - PROGRESS: at 59.15% examples, 140409 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:57,623 : INFO : EPOCH 20 - PROGRESS: at 59.70% examples, 140234 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:14:58,670 : INFO : EPOCH 20 - PROGRESS: at 60.19% examples, 139926 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:14:59,846 : INFO : EPOCH 20 - PROGRESS: at 60.74% examples, 139552 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:15:00,945 : INFO : EPOCH 20 - PROGRESS: at 61.32% examples, 139276 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:01,974 : INFO : EPOCH 20 - PROGRESS: at 62.00% examples, 139387 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:03,025 : INFO : EPOCH 20 - PROGRESS: at 62.47% examples, 139009 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:04,066 : INFO : EPOCH 20 - PROGRESS: at 63.25% examples, 139288 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:05,106 : INFO : EPOCH 20 - PROGRESS: at 63.81% examples, 139097 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:06,189 : INFO : EPOCH 20 - PROGRESS: at 64.46% examples, 139061 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:07,229 : INFO : EPOCH 20 - PROGRESS: at 65.09% examples, 139066 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:08,333 : INFO : EPOCH 20 - PROGRESS: at 65.68% examples, 138902 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:15:09,364 : INFO : EPOCH 20 - PROGRESS: at 66.25% examples, 138737 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:10,377 : INFO : EPOCH 20 - PROGRESS: at 66.82% examples, 138684 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:11,455 : INFO : EPOCH 20 - PROGRESS: at 67.42% examples, 138565 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:12,568 : INFO : EPOCH 20 - PROGRESS: at 68.00% examples, 138402 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:13,603 : INFO : EPOCH 20 - PROGRESS: at 68.69% examples, 138512 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:14,712 : INFO : EPOCH 20 - PROGRESS: at 69.33% examples, 138524 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:15,768 : INFO : EPOCH 20 - PROGRESS: at 69.68% examples, 137937 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:16,802 : INFO : EPOCH 20 - PROGRESS: at 70.19% examples, 137718 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:17,841 : INFO : EPOCH 20 - PROGRESS: at 70.73% examples, 137567 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:18,843 : INFO : EPOCH 20 - PROGRESS: at 71.33% examples, 137626 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:19,927 : INFO : EPOCH 20 - PROGRESS: at 72.03% examples, 137676 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:20,941 : INFO : EPOCH 20 - PROGRESS: at 72.47% examples, 137406 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:22,095 : INFO : EPOCH 20 - PROGRESS: at 73.16% examples, 137453 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:23,147 : INFO : EPOCH 20 - PROGRESS: at 73.80% examples, 137538 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:24,236 : INFO : EPOCH 20 - PROGRESS: at 74.52% examples, 137650 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:15:25,253 : INFO : EPOCH 20 - PROGRESS: at 75.06% examples, 137536 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:15:26,307 : INFO : EPOCH 20 - PROGRESS: at 75.69% examples, 137541 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:27,409 : INFO : EPOCH 20 - PROGRESS: at 76.39% examples, 137652 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:28,487 : INFO : EPOCH 20 - PROGRESS: at 76.99% examples, 137544 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:29,501 : INFO : EPOCH 20 - PROGRESS: at 77.75% examples, 137740 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:30,660 : INFO : EPOCH 20 - PROGRESS: at 78.53% examples, 137857 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:31,679 : INFO : EPOCH 20 - PROGRESS: at 79.37% examples, 138260 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:32,691 : INFO : EPOCH 20 - PROGRESS: at 79.97% examples, 138293 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:33,795 : INFO : EPOCH 20 - PROGRESS: at 80.68% examples, 138386 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:34,835 : INFO : EPOCH 20 - PROGRESS: at 81.41% examples, 138475 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:35,875 : INFO : EPOCH 20 - PROGRESS: at 82.21% examples, 138764 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:36,997 : INFO : EPOCH 20 - PROGRESS: at 82.91% examples, 138826 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:38,022 : INFO : EPOCH 20 - PROGRESS: at 83.53% examples, 138851 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:39,117 : INFO : EPOCH 20 - PROGRESS: at 84.09% examples, 138668 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:15:40,173 : INFO : EPOCH 20 - PROGRESS: at 84.71% examples, 138588 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:41,266 : INFO : EPOCH 20 - PROGRESS: at 85.38% examples, 138617 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:15:42,271 : INFO : EPOCH 20 - PROGRESS: at 86.07% examples, 138730 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:43,272 : INFO : EPOCH 20 - PROGRESS: at 86.83% examples, 138974 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:44,432 : INFO : EPOCH 20 - PROGRESS: at 87.56% examples, 138994 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:45,477 : INFO : EPOCH 20 - PROGRESS: at 88.40% examples, 139329 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:46,540 : INFO : EPOCH 20 - PROGRESS: at 89.10% examples, 139432 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:15:47,554 : INFO : EPOCH 20 - PROGRESS: at 89.86% examples, 139653 words/s, in_qsize 14, out_qsize 1
    2020-09-30 23:15:48,586 : INFO : EPOCH 20 - PROGRESS: at 90.59% examples, 139800 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:49,648 : INFO : EPOCH 20 - PROGRESS: at 91.38% examples, 139976 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:50,765 : INFO : EPOCH 20 - PROGRESS: at 92.19% examples, 140161 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:51,800 : INFO : EPOCH 20 - PROGRESS: at 93.05% examples, 140481 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:52,896 : INFO : EPOCH 20 - PROGRESS: at 93.81% examples, 140613 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:53,928 : INFO : EPOCH 20 - PROGRESS: at 94.56% examples, 140747 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:54,931 : INFO : EPOCH 20 - PROGRESS: at 95.25% examples, 140838 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:55,984 : INFO : EPOCH 20 - PROGRESS: at 96.04% examples, 141009 words/s, in_qsize 16, out_qsize 0
    2020-09-30 23:15:57,047 : INFO : EPOCH 20 - PROGRESS: at 96.74% examples, 141107 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:58,139 : INFO : EPOCH 20 - PROGRESS: at 97.55% examples, 141298 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:15:59,177 : INFO : EPOCH 20 - PROGRESS: at 98.30% examples, 141469 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:16:00,234 : INFO : EPOCH 20 - PROGRESS: at 99.05% examples, 141623 words/s, in_qsize 15, out_qsize 0
    2020-09-30 23:16:01,085 : INFO : worker thread finished; awaiting finish of 7 more threads
    2020-09-30 23:16:01,119 : INFO : worker thread finished; awaiting finish of 6 more threads
    2020-09-30 23:16:01,130 : INFO : worker thread finished; awaiting finish of 5 more threads
    2020-09-30 23:16:01,190 : INFO : worker thread finished; awaiting finish of 4 more threads
    2020-09-30 23:16:01,201 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-09-30 23:16:01,315 : INFO : EPOCH 20 - PROGRESS: at 99.92% examples, 141871 words/s, in_qsize 2, out_qsize 1
    2020-09-30 23:16:01,316 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 23:16:01,316 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 23:16:01,352 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 23:16:01,353 : INFO : EPOCH - 20 : training on 23279529 raw words (22951015 effective words) took 161.7s, 141953 effective words/s
    2020-09-30 23:16:01,353 : INFO : training on a 465590580 raw words (459020300 effective words) took 1185.9s, 387059 effective words/s

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.304920 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/m,d100,n5,w10,mc2,t8)

    0.1042 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.10632 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)





Achieved Sentiment-Prediction Accuracy
--------------------------------------
Compare error rates achieved, best-to-worst


.. code-block:: default

    print("Err_rate Model")
    for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
        print(f"{rate} {name}")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Err_rate Model
    0.1042 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/m,d100,n5,w10,mc2,t8)
    0.1056 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.10632 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.1684 Doc2Vec(dm/m,d100,n5,w10,mc2,t8)
    0.30492 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)




In our testing, contrary to the results of the paper, on this problem,
PV-DBOW alone performs as good as anything else. Concatenating vectors from
different models only sometimes offers a tiny predictive improvement – and
stays generally close to the best-performing solo model included.

The best results achieved here are just around 10% error rate, still a long
way from the paper's reported 7.42% error rate.

(Other trials not shown, with larger vectors and other changes, also don't
come close to the paper's reported value. Others around the net have reported
a similar inability to reproduce the paper's best numbers. The PV-DM/C mode
improves a bit with many more training epochs – but doesn't reach parity with
PV-DBOW.)


Examining Results
-----------------

Let's look for answers to the following questions:

#. Are inferred vectors close to the precalculated ones?
#. Do close documents seem more related than distant ones?
#. Do the word vectors show useful similarities?
#. Are the word vectors from this dataset any good at analogies?


Are inferred vectors close to the precalculated ones?
-----------------------------------------------------


.. code-block:: default

    doc_id = np.random.randint(len(simple_models[0].dv))  # Pick random doc; re-run cell for more examples
    print(f'for doc {doc_id}...')
    for model in simple_models:
        inferred_docvec = model.infer_vector(alldocs[doc_id].words)
        print(f'{model}:\n {model.dv.most_similar([inferred_docvec], topn=3)}')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    for doc 43085...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(43085, 0.9763993620872498), (60169, 0.6021242737770081), (19410, 0.6011009812355042)]
    Doc2Vec(dm/m,d100,n5,w10,mc2,t8):
     [(43085, 0.8938843011856079), (95363, 0.5461075305938721), (14357, 0.5367367267608643)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(43085, 0.8192846775054932), (35174, 0.48924335837364197), (63795, 0.4852815568447113)]




(Yes, here the stored vector from 20 epochs of training is usually one of the
closest to a freshly-inferred vector for the same words. Defaults for
inference may benefit from tuning for each dataset or model parameters.)


Do close documents seem more related than distant ones?
-------------------------------------------------------


.. code-block:: default

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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    TARGET (94198): «Believe it or not this is a good movie. I don't hate this at all. Why? It has horrific dialog, plot holes large enough so you can plow a cruise ship thorough, and not much going on in the plot department. I like this movie for its setting(so exotic, so alive, and up to date) In fact, so up to date that this is one of the first movies that had modern cellphones in them. Believe it or not the hero in this movie takes the damn bus! THE BUS!!! Sharon Stone and Slyvester Stallone make a great couple especially in "sex in the hotel shower" scene. Talk about chemistry. James Woods makes adds some hostility and a pulse to this melodrama with his wisecracks, outbursts, and foul-mouthed banter as the villain. But what's with the late Rod Stieger and Eric Roberts fronting as Cuban mobsters? C'mon, here. The strange thing is, they make it work. The action sequences such as the Sly kicking ass and taking names while encountering a couple of thugs on a bus and throwing one out the window, a hotel room explodes and collapses into the ocean with the bad guys still in it, and the funky way Sly sets traps for the bad guys gives this movie redemption. However, the characters in question could've been a little bolder, a little more gunplay is needed, and I have a major beef with discrepancies in the May Munro story. She looks 35, now if her parents were killed when she was a little girl, shouldn't Tomas be a tad older(he looked the same age in her flashback) The Specialist would've been a great movie if the plot was better explained and more intense.»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dbow,d100,n5,mc2,t8)%s:

    MOST (41214, 0.625079333782196): «Back in the forties, when movies touched on matters not yet admissible in "polite" society, they resorted to codes which supposedly floated over the heads of most of the audience while alerting those in the know to just what was up. Probably no film of the decade was so freighted with innuendo as the oddly obscure Desert Fury, set in a small gambling oasis called Chuckawalla somewhere in the California desert. Proprietress of the Purple Sage saloon and casino is the astonishing Mary Astor, in slacks and sporting a cigarette holder; into town drives her handful-of-a-daughter, Lizabeth Scott, looking, in Technicolor, like 20-million bucks. But listen to the dialogue between them, which suggests an older Lesbian and her young, restless companion (one can only wonder if A.I. Bezzerides' original script made this relationship explicit). Even more blatant are John Hodiak as a gangster and Wendell Corey as his insanely jealous torpedo. Add Burt Lancaster as the town sheriff, stir, and sit back. Both Lancaster and (surprisingly) Hodiak fall for Scott. It seems, however, that Hodiak not only has a past with Astor, but had a wife who died under suspicious circumstances. The desert sun heats these ingredients up to a hard boil, with face-slappings aplenty and empurpled exchanges. Don't pass up this hothouse melodrama, chock full of creepily exotic blooms, if it comes your way; it's a remarkable movie.»

    MEDIAN (62786, 0.2857852280139923): «This is a superb storyline and has excellent music. Set in the background of Rajasthan, this picture gives a beautiful insight into the Rajasthani culture and way of life. One wonders why more such enjoyable pictures are not made. More such pictures will enhance the stock of Bollywood films.»

    LEAST (82757, -0.04075651988387108): «Taj Mahal Badalandabad (played by Kal Penn) had a few funny moments in the movie (i.e. jokes regarding the former British aristocracy in India), but the movie almost nothing but sex jokes for the entire movie. This sort of humor included nearly a dozen different euphemisms for the female anatomy (I lost count), and countless jokes about how to "score" with the opposite gender. I laughed at the first few sex jokes, but by the end of the movie, I was really tired of hearing crude jokes.<br /><br />If this sort of humor is appealing to you, then you'll enjoy this movie. Otherwise, don't bother seeing it. Overall, I felt "Harold and Kumar" was much more humorous. I'm a fan of Kal Penn's, but unfortunately, I cannot recommend this movie.<br /><br />Brief Plot Synopsis (no spoilers below): Taj heads to Camford University (name taken from Cambridge and Oxford) in England to pursue his higher education and to follow in his fathers footsteps. He believes that he has been accepted to the prestigious fraternal guild "Fox and Hounds". However, upon arriving at the University, he is told by a "Fox and Hounds" member (nicknamed "Pip") that he was mistakenly sent this acceptance letter.<br /><br />Taj is forced to become a teacher's assistant at the "Barn" residence, which contains a group of social outcasts. Pip and the other "Fox and Hounds" members treat Taj and his house mates very poorly. Determined to help the members of the Barn residence gain acceptance, Taj starts the "Cock and Bulls" fraternal guild. He hopes that the "Cock and Bulls" guild will win the Camford Cup, which is an annual academic, social, and athletic competition between fraternities. The bitterness between Taj and Pip only increases as the fraternities compete in several competitions to earn points towards the Camford Cup.<br /><br />I cannot say anything more about the plot, or else I'll spoil it for you.»





Somewhat, in terms of reviewer tone, movie genre, etc... the MOST
cosine-similar docs usually seem more like the TARGET than the MEDIAN or
LEAST... especially if the MOST has a cosine-similarity > 0.5. Re-run the
cell to try another random target document.


Do the word vectors show useful similarities?
---------------------------------------------



.. code-block:: default

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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    target_word: 'GAY' model: Doc2Vec(dbow,d100,n5,mc2,t8) similar words:
        1. 0.46 'Penn'
        2. 0.45 'diverts'
        3. 0.43 '"Gas-s-s-s"'
        4. 0.42 'Lance."'
        5. 0.42 "1930's;"
        6. 0.42 "aren't?"
        7. 0.40 'Halder'
        8. 0.40 'dates.<br'
        9. 0.40 'INDEPENDENCE'
        10. 0.40 'Airplane'

    target_word: 'GAY' model: Doc2Vec(dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.50 'judge.<br'
        2. 0.48 'crappy!<br'
        3. 0.46 '/>OUR'
        4. 0.46 '/>Girl'
        5. 0.45 'esteem.<br'
        6. 0.45 'proclivities,'
        7. 0.45 'brains.<br'
        8. 0.45 'arrested.<br'
        9. 0.45 'true),'
        10. 0.45 'origins)'

    target_word: 'GAY' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.59 "'every"
        2. 0.59 'Venezuelan'
        3. 0.58 'challenged)'
        4. 0.56 'deceiving'
        5. 0.55 'best-looking'
        6. 0.55 "'lone"
        7. 0.54 'Noona'
        8. 0.54 '"peace"'
        9. 0.54 'fruitiest'
        10. 0.54 'tele'





Do the DBOW words look meaningless? That's because the gensim DBOW model
doesn't train word vectors – they remain at their random initialized values –
unless you ask with the ``dbow_words=1`` initialization parameter. Concurrent
word-training slows DBOW mode significantly, and offers little improvement
(and sometimes a little worsening) of the error rate on this IMDB
sentiment-prediction task, but may be appropriate on other tasks, or if you
also need word-vectors.

Words from DM models tend to show meaningfully similar words when there are
many examples in the training data (as with 'plot' or 'actor'). (All DM modes
inherently involve word-vector training concurrent with doc-vector training.)


Are the word vectors from this dataset any good at analogies?
-------------------------------------------------------------


.. code-block:: default


    from gensim.test.utils import datapath
    questions_filename = datapath('questions-words.txt')

    # Note: this analysis takes many minutes
    for model in word_models:
        score, sections = model.wv.evaluate_word_analogies(questions_filename)
        correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
        print(f'{model}: {float(correct*100)/(correct+incorrect):0.2f}%% correct ({correct} of {correct+incorrect}')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    2020-09-30 23:16:20,972 : INFO : Evaluating word analogies for top 300000 words in the model on /Volumes/work/workspace/gensim/trunk/gensim/test/test_data/questions-words.txt
    2020-09-30 23:16:29,726 : INFO : capital-common-countries: 0.0% (0/420)
    2020-09-30 23:16:49,935 : INFO : capital-world: 0.0% (0/902)
    2020-09-30 23:16:51,662 : INFO : currency: 0.0% (0/86)
    2020-09-30 23:17:21,119 : INFO : city-in-state: 0.0% (0/1510)
    2020-09-30 23:17:30,185 : INFO : family: 0.0% (0/506)
    2020-09-30 23:17:48,710 : INFO : gram1-adjective-to-adverb: 0.0% (0/992)
    2020-09-30 23:18:02,794 : INFO : gram2-opposite: 0.0% (0/756)
    2020-09-30 23:18:27,825 : INFO : gram3-comparative: 0.0% (0/1332)
    2020-09-30 23:18:47,756 : INFO : gram4-superlative: 0.0% (0/1056)
    2020-09-30 23:19:06,340 : INFO : gram5-present-participle: 0.0% (0/992)
    2020-09-30 23:19:32,225 : INFO : gram6-nationality-adjective: 0.0% (0/1445)
    2020-09-30 23:19:58,736 : INFO : gram7-past-tense: 0.0% (0/1560)
    2020-09-30 23:20:33,145 : INFO : gram8-plural: 0.0% (0/1190)
    2020-09-30 23:20:49,979 : INFO : gram9-plural-verbs: 0.0% (0/870)
    2020-09-30 23:20:49,980 : INFO : Quadruplets with out-of-vocabulary words: 30.3%
    2020-09-30 23:20:49,998 : INFO : NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"
    2020-09-30 23:20:49,998 : INFO : Total accuracy: 0.0% (0/13617)
    Doc2Vec(dbow,d100,n5,mc2,t8): 0.00%% correct (0 of 13617
    2020-09-30 23:20:50,933 : INFO : Evaluating word analogies for top 300000 words in the model on /Volumes/work/workspace/gensim/trunk/gensim/test/test_data/questions-words.txt
    2020-09-30 23:20:58,733 : INFO : capital-common-countries: 3.8% (16/420)
    2020-09-30 23:21:16,437 : INFO : capital-world: 0.9% (8/902)
    2020-09-30 23:21:17,872 : INFO : currency: 0.0% (0/86)
    2020-09-30 23:21:43,204 : INFO : city-in-state: 0.3% (4/1510)
    2020-09-30 23:21:52,906 : INFO : family: 39.7% (201/506)
    2020-09-30 23:22:11,367 : INFO : gram1-adjective-to-adverb: 3.2% (32/992)
    2020-09-30 23:22:24,694 : INFO : gram2-opposite: 5.7% (43/756)
    2020-09-30 23:22:49,286 : INFO : gram3-comparative: 49.3% (657/1332)
    2020-09-30 23:23:08,496 : INFO : gram4-superlative: 25.2% (266/1056)
    2020-09-30 23:23:27,399 : INFO : gram5-present-participle: 21.9% (217/992)
    2020-09-30 23:23:51,888 : INFO : gram6-nationality-adjective: 2.8% (41/1445)
    2020-09-30 23:24:20,544 : INFO : gram7-past-tense: 28.5% (445/1560)
    2020-09-30 23:24:38,103 : INFO : gram8-plural: 19.3% (230/1190)
    2020-09-30 23:24:53,145 : INFO : gram9-plural-verbs: 45.3% (394/870)
    2020-09-30 23:24:53,147 : INFO : Quadruplets with out-of-vocabulary words: 30.3%
    2020-09-30 23:24:53,147 : INFO : NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"
    2020-09-30 23:24:53,147 : INFO : Total accuracy: 18.8% (2554/13617)
    Doc2Vec(dm/m,d100,n5,w10,mc2,t8): 18.76%% correct (2554 of 13617
    2020-09-30 23:24:53,978 : INFO : Evaluating word analogies for top 300000 words in the model on /Volumes/work/workspace/gensim/trunk/gensim/test/test_data/questions-words.txt
    2020-09-30 23:25:01,268 : INFO : capital-common-countries: 1.9% (8/420)
    2020-09-30 23:25:20,800 : INFO : capital-world: 0.6% (5/902)
    2020-09-30 23:25:22,770 : INFO : currency: 0.0% (0/86)
    2020-09-30 23:25:50,052 : INFO : city-in-state: 0.3% (4/1510)
    2020-09-30 23:25:58,619 : INFO : family: 39.3% (199/506)
    2020-09-30 23:26:15,657 : INFO : gram1-adjective-to-adverb: 6.9% (68/992)
    2020-09-30 23:26:30,514 : INFO : gram2-opposite: 4.6% (35/756)
    2020-09-30 23:26:54,775 : INFO : gram3-comparative: 33.9% (452/1332)
    2020-09-30 23:27:17,559 : INFO : gram4-superlative: 24.3% (257/1056)
    2020-09-30 23:27:38,425 : INFO : gram5-present-participle: 37.0% (367/992)
    2020-09-30 23:28:01,531 : INFO : gram6-nationality-adjective: 1.9% (27/1445)
    2020-09-30 23:28:26,685 : INFO : gram7-past-tense: 26.5% (414/1560)
    2020-09-30 23:28:46,289 : INFO : gram8-plural: 8.5% (101/1190)
    2020-09-30 23:28:58,855 : INFO : gram9-plural-verbs: 46.2% (402/870)
    2020-09-30 23:28:58,858 : INFO : Quadruplets with out-of-vocabulary words: 30.3%
    2020-09-30 23:28:58,859 : INFO : NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"
    2020-09-30 23:28:58,859 : INFO : Total accuracy: 17.2% (2339/13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 17.18%% correct (2339 of 13617




Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 52 minutes  12.903 seconds)

**Estimated memory usage:**  3494 MB


.. _sphx_glr_download_auto_examples_howtos_run_doc2vec_imdb.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_doc2vec_imdb.py <run_doc2vec_imdb.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_doc2vec_imdb.ipynb <run_doc2vec_imdb.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
