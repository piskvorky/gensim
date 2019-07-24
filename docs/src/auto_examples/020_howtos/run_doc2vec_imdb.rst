.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_020_howtos_run_doc2vec_imdb.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_020_howtos_run_doc2vec_imdb.py:


.. _run_doc2vec_imdb.py:

How to Reproduce Results of Le and Mikolov 2014
===============================================

Shows how to reproduce results of the Le and Mikolov paper using Gensim.

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
        # We can't read it on the fly because of
        # https://github.com/RaRe-Technologies/smart_open/issues/331
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







Working with the entire 100k document corpus takes close to 40 minutes.
Let's reduce the size of the dataset so we can run the example more quickly.



.. code-block:: default

    import random

    def shrink_list(the_list, fraction):
        sample_size = int(fraction * len(the_list))
        return random.sample(the_list, sample_size)

    def shrink_dataset(docs, fraction=0.1):
        train = shrink_list([d for d in docs if d.split == 'train'], fraction)
        test = shrink_list([d for d in docs if d.split == 'test'], fraction)
        extra = shrink_list([d for d in docs if d.split == 'extra'], fraction)

        return [
            SentimentDocument(d.words, [i], d.split, d.sentiment)
            for (i, d) in enumerate(train + test + extra)
        ]

    # comment the line below to use the full dataset
    # alldocs = shrink_dataset(alldocs)







Here's what a single document looks like


.. code-block:: default

    print(alldocs[27])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SentimentDocument(words=['I', 'was', 'looking', 'forward', 'to', 'this', 'movie.', 'Trustworthy', 'actors,', 'interesting', 'plot.', 'Great', 'atmosphere', 'then', '?????', 'IF', 'you', 'are', 'going', 'to', 'attempt', 'something', 'that', 'is', 'meant', 'to', 'encapsulate', 'the', 'meaning', 'of', 'life.', 'First.', 'Know', 'it.', 'OK', 'I', 'did', 'not', 'expect', 'the', 'directors', 'or', 'writers', 'to', 'actually', 'know', 'the', 'meaning', 'but', 'I', 'thought', 'they', 'may', 'have', 'offered', 'crumbs', 'to', 'peck', 'at', 'and', 'treats', 'to', 'add', 'fuel', 'to', 'the', 'fire-Which!', 'they', 'almost', 'did.', 'Things', 'I', "didn't", 'get.', 'A', 'woman', 'wandering', 'around', 'in', 'dark', 'places', 'and', 'lonely', 'car', 'parks', 'alone-oblivious', 'to', 'the', 'consequences.', 'Great', 'riddles', 'that', 'fell', 'by', 'the', 'wayside.', 'The', 'promise', 'of', 'the', 'knowledge', 'therein', 'contained', 'by', 'the', 'original', 'so-called', 'criminal.', 'I', 'had', 'no', 'problem', 'with', 'the', 'budget', 'and', 'enjoyed', 'the', 'suspense.', 'I', 'understood', 'and', 'can', 'wax', 'lyrical', 'about', 'the', 'fool', 'and', 'found', 'Adrian', 'Pauls', 'role', 'crucial', 'and', 'penetrating', 'and', 'then', '?????', 'Basically', 'the', 'story', 'line', 'and', 'the', 'script', 'where', 'good', 'up', 'to', 'a', 'point', 'and', 'that', 'point', 'was', 'the', 'last', '10', 'minutes', 'or', 'so.', 'What?', 'Run', 'out', 'of', 'ideas!', 'Such', 'a', 'pity', 'that', 'this', 'movie', 'had', 'to', 'let', 'us', 'down', 'so', 'badly.', 'It', 'may', 'not', 'comprehend', 'the', 'meaning', 'and', 'I', 'really', 'did', 'not', 'expect', 'the', 'writers', 'to', 'understand', 'it', 'but', 'I', 'was', 'hoping', 'for', 'an', 'intellectual,', 'if', 'not', 'spiritual', 'ride', 'and', 'got', 'a', 'bump', 'in', 'the', 'road'], tags=[27], split='test', sentiment=0.0)


Extract our documents and split into training/test sets


.. code-block:: default

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))





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
        print("%s vocabulary scanned & state initialized" % model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Doc2Vec(dbow,d100,n5,mc2,t8) vocabulary scanned & state initialized
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) vocabulary scanned & state initialized
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







Sanity checking.  Let's see if our models give meaningful results.


.. code-block:: default

    for word, sim in simple_models[1].wv.most_similar('head', topn=5):
        print('%.2f %r' % (sim, word))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.42 'Montesi'
    0.41 'libido,'
    0.40 'actors?'
    0.40 "Chloe's"
    0.39 'internet.'


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
        train_regressors = [test_model.docvecs[doc.tags[0]] for doc in train_set]
        train_regressors = sm.add_constant(train_regressors)
        predictor = logistic_predictor_from_data(train_targets, train_regressors)

        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_set]
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
        print("Training %s" % model)
        model.train(shuffled_alldocs, total_examples=len(shuffled_alldocs), epochs=model.epochs)

        print("\nEvaluating %s" % model)
        err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
        error_rates[str(model)] = err_rate
        print("\n%f %s\n" % (err_rate, model))

    for model in [models_by_name['dbow+dmm'], models_by_name['dbow+dmc']]:
        print("\nEvaluating %s" % model)
        err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
        error_rates[str(model)] = err_rate
        print("\n%f %s\n" % (err_rate, model))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Training Doc2Vec(dbow,d100,n5,mc2,t8)

    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)

    0.105760 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Evaluating Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.169960 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.305680 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.104880 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.106360 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


Achieved Sentiment-Prediction Accuracy
--------------------------------------
Compare error rates achieved, best-to-worst


.. code-block:: default

    print("Err_rate Model")
    for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
        print("%f %s" % (rate, name))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Err_rate Model
    0.104880 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.105760 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.106360 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.169960 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.305680 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # Pick random doc; re-run cell for more examples
    print('for doc %d...' % doc_id)
    for model in simple_models:
        inferred_docvec = model.infer_vector(alldocs[doc_id].words)
        print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    for doc 20816...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(20816, 0.9887647032737732), (92834, 0.585157573223114), (24294, 0.5718448162078857)]
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):
     [(20816, 0.9381554126739502), (44297, 0.5630723834037781), (57837, 0.5299752354621887)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(20816, 0.8699843287467957), (93622, 0.47360754013061523), (16566, 0.4439433813095093)]


(Yes, here the stored vector from 20 epochs of training is usually one of the
closest to a freshly-inferred vector for the same words. Defaults for
inference may benefit from tuning for each dataset or model parameters.)


Do close documents seem more related than distant ones?
-------------------------------------------------------


.. code-block:: default

    import random

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
    model = random.choice(simple_models)  # and a random model
    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
    print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        s = sims[index]
        i = sims[index][0]
        words = ' '.join(alldocs[i].words)
        print(u'%s %s: «%s»\n' % (label, s, words))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    TARGET (4181): «How pointless, hideous characters and boring film. Saved by brief sex scenes, mad witch, gorgeous desert island and Brooks body. The plot is tenuous, the characters are shallow and unlikeable. Having said that I did manage to watch it all, mainly because I was totally transfixed by the jiggling and kind of hoping that her character would come good in the end. The film was well shot, well directed but perhaps the casting let it down in some ways. Disappointing. Really summed the review up in the first line but this website dictates that you need to write 10 lines minimum. It would be better to spend the time watching another film.»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):

    MOST (6061, 0.9222484827041626): «How pointless, hideous characters and boring film. Saved by brief sex scenes, mad witch, gorgeous desert island and Brooks body. The plot is tenuous, the characters are shallow and unlikeable. Having said that I did manage to watch it all, mainly because I was totally transfixed by the jiggling and kind of hoping that her character would come good in the end. The film was well shot, well directed but perhaps the casting let it down in some ways. Disappointing. Really summed the review up in the first line but this website dictates that you need to write 10 lines minimum. It would be better to spend the time watching another film.»

    MEDIAN (71443, 0.3313252925872803): «While a decent enough sequel, Ring 2 is a bit of a disappointment in that it tries to weave too many threads and characters, tries to have too much going on and too many characters and ultimately gets a bit muddled and incoherent towards the end. Another thing that disappoints, after the masterpiece that was "Ring", is that it lacks any one big scare or standout sequence to match the first film's finale or most memorable moments.<br /><br />One thing that Ring 2 scores in is it's series of little scares. There is still a creepy atmosphere (just without the pay off of the first film). The scene with Shizuko combing her hair, gliding along the floor and peering round the corner with a eerie expression is a genuinely chilling moment. There's also an effective sequence where a survivor from the first film sets off the hospital television<br /><br />and the other patients start to panic at the sight of the infamous well. The confusing ending also gives us the<br /><br />return of Sadako (now with the clayface of her real corpse) climbing up the well. The continuation of style, situations and characters from the first film, prevents this from just being an unrelated sequel, but it tends to ask new questions and then provides no answers. A certain hint of ambiguity is introduced to the story of Sadako, who was a convincingly malevolent presence in the first film (despite the beautifully poignant scene where Reiko finds her body in the well and the skull almost appears to cry the gloop off from it). The more sympathetic view of Sadako is successfully followed up in the prequel "Ring 0: Birthday".<br /><br />One of the film's assets for me was Kyoko Fukada as Kanae, a young student who makes the mistake of trusting a reporter to keep his word. Apparently she's some kind of pop star in real life but I was unaware of that before being impressed with her performance. Kanae is sweet and cute and seems undeserving of the fate that the reporter gets her into, by shamefully manipulating her and then lying repeatedly until it's too late. The scene where the reporter then tries to conveniently erase his guilt along with footage of Kanae backfires on him and provides one of the film's memorably eerie scenes (but also one of the confusing plot threads that I didn't really understand until reading the FAQ at RingWorld). Later there is a certain satisfactory poetic justice when Kanae appears, giggling and smiling, behind the reporter in the hospital. Ultimately much of the satisfaction to be had from this<br /><br />sequel, comes from the groundwork and impact of the original film. However it is a decent, if flawed effort and still proves to be fascinating and creepy in places. A far better sequel than many. Just needed a bit more overall coherency.»

    LEAST (46784, -0.07450056076049805): «Panahi, whose previous films such as The Circle and Crimson Gold have seemed to range from dour to grim, has produced in his new Offside a funny, obstreperous, joyously chaotic ensemble piece that ends on a note of liberation and heartfelt fun  yet the movie deals with material quite as challenging and relevant as anything else he's done. By focusing on a group of ardent girl soccer fans caught sneaking into the pre-World Cup Bahrain-Iran match in Tehran stadium, Panahi brings up issues of national spirit and independent-mindedness, and the contradictions  and sheer absurdity  of the regime's religious gender apartheid in a world of modern competition with a majority youth population and urban girls who increasingly think for themselves.<br /><br />As the film opens we breathlessly join one of the girls in a bus, with a father pursing a lost daughter. This one has a disguise and has national colors as warpaint, but we cringe with her in the knowledge of what's going to happen: she's still easily spotted. The thing is, most of the men around don't really care. Still, rules are rules, and once they try to make it through the various checkpoints on the way into the big stadium the would-be soccer girls, or some of them anyway, get rounded up and held in a little compound upstairs in the stadium by some mostly young, green, and rustic soldier-cops who have no idea how to deal with these big city girls' independent ideas and would rather be watching the game  whose roar we constantly hear in the background  themselves. Each girl is different  represents a different set of reasons for wanting to break the rules and different ways of doing it. One wore a soldier's uniform and got into the officers' section. One is tough and masculine and mocking and provocative (she could pass for a pretty boy, and teasingly hints at that: "Are you a girl or a boy?" "Which would you like me to be?"). One doesn't care very much about soccer but went to honor a dead comrade. One (Aida Sadeghi) is an ardent soccer player herself  and so on. These Tehrani girls are stubborn and smart and they walk all over the uptight rural lieutenant in charge of them (Safar Samandar). One of the rural cops (Mohamad Kheirabadi) takes the girl soccer player to the men's restroom (of course there's no ladies'), forcing her to wear a poster of an Italian football star as a mask. A comedy of errors and chaos follows in which the girl escapes.<br /><br />Later a spiffy looking van comes with an officer who directs the cops to take the girls to the Vice Department  violating sexual segregation rules qualifies as vice. A male gets mixed in with them  a kid who's chronically guilty of smuggling fireworks into the games. The van turns out not to be so spiffy: the radio aerial is broken. But one cop holds it in place so they can listen to the increasingly heart-stopping reportage. Cops and prisoners are all joined in a common excitement now. There's no score, the game goes to penalty kicks, and the winner will go to Germany.<br /><br />In the background through all this is a real game, a real stadium, and real masses of young men crazy about the outcome of this event. The excitement is tremendous, and the streets are jammed with cars and flags and a milling mob of supporters praying for an Iranian win and united in their excitement.<br /><br />What makes this film so good, as may be clear by now, is that it's shot during the evening of an actual game with a real finale that turns everything around. This, in contrast to Panahi's previous highly calculated narrative trajectories, is spontaneous vérité film-making that improvises in rhythm with a captured background of actual events and sweeps you into its excitement in ways that are quite thrilling.<br /><br />The essence of Offside is the disconnect between modern world soccer madness and retro-Islamic social prohibitions repressing women  the latter existing at a time when young Iranian women are becoming part of a global world in which females participate in sport and share in the ardor of national team spirit. How exactly do you reconcile the country's ambition to become a modern global power with social attitudes that are medieval? A lot of Offisde is astonishingly real, including the way everybody tries to talk their way out of everything. The director's decision to inject young actors into an actual sports mega-event leads to a stunningly effective blend of documentary, polemic, and fiction that is too energetic to seem to have a bone to pick, and that ends in a way that's brilliant and moving.<br /><br />I've had reservations about Panahi's films before, but this one kicks ass. Panahi does something remarkable here. He critiques his society, presents an unusual drama, and touches our hearts with a sense of a nation's aspirations.»


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
            word = random.choice(model.wv.index2word)
            if model.wv.vocab[word].count > threshold:
                return word

    target_word = pick_random_word(word_models[0])
    # or uncomment below line, to just pick a word from the relevant domain:
    # target_word = 'comedy/drama'

    for model in word_models:
        print('target_word: %r model: %s similar words:' % (target_word, model))
        for i, (word, sim) in enumerate(model.wv.most_similar(target_word, topn=10), 1):
            print('    %d. %.2f %r' % (i, sim, word))
        print()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    target_word: 'tried,' model: Doc2Vec(dbow,d100,n5,mc2,t8) similar words:
        1. 0.47 'crashes,'
        2. 0.40 'green.<br'
        3. 0.40 'harpies.'
        4. 0.39 'unalloyed'
        5. 0.39 'dislike?'
        6. 0.39 'maintenance'
        7. 0.39 '"unofficially"'
        8. 0.39 'remained,'
        9. 0.38 '1,000'
        10. 0.38 'Callow,'

    target_word: 'tried,' model: Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.56 'believe,'
        2. 0.55 'chuckled'
        3. 0.55 'think,'
        4. 0.55 'GUARANTEE'
        5. 0.54 'confess,'
        6. 0.54 'read,'
        7. 0.54 'FAILED,'
        8. 0.53 'thought'
        9. 0.53 'commend'
        10. 0.53 'liked,'

    target_word: 'tried,' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.67 'returned,'
        2. 0.65 'noticed,'
        3. 0.64 'liked,'
        4. 0.63 'MAYBE,'
        5. 0.62 'arrived,'
        6. 0.62 'cried,'
        7. 0.61 'failed,'
        8. 0.61 'said...'
        9. 0.60 "hadn't,"
        10. 0.60 'dined'


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


    # grab the file if not already local
    questions_filename = 'questions-words.txt'
    if not os.path.isfile(questions_filename):
        # Download IMDB archive
        print("Downloading analogy questions file...")
        url = u'https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt'
        with smart_open.open(url, 'rb') as fin:
            with smart_open.open(questions_filename, 'wb') as fout:
                fout.write(fin.read())
    assert os.path.isfile(questions_filename), "questions-words.txt unavailable"
    print("Success, questions-words.txt is available for next steps.")

    # Note: this analysis takes many minutes
    for model in word_models:
        score, sections = model.wv.evaluate_word_analogies('questions-words.txt')
        correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
        print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Success, questions-words.txt is available for next steps.
    Doc2Vec(dbow,d100,n5,mc2,t8): 0.00% correct (0 of 13617)
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8): 19.01% correct (2588 of 13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 18.65% correct (2540 of 13617)


Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 36 minutes  30.419 seconds)

**Estimated memory usage:**  4061 MB


.. _sphx_glr_download_auto_examples_020_howtos_run_doc2vec_imdb.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/020_howtos/run_doc2vec_imdb.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_doc2vec_imdb.py <run_doc2vec_imdb.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_doc2vec_imdb.ipynb <run_doc2vec_imdb.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
