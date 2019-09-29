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

    0.43 'black-market'
    0.42 'boxcar'
    0.42 '"Agent'
    0.42 'full?'
    0.42 'Distress"'


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

    0.104040 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Evaluating Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.171680 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.305920 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.104320 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.104440 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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
    0.104040 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.104320 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.104440 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.171680 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.305920 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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

    for doc 58241...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(58241, 0.9865663051605225), (59721, 0.5768244862556458), (82051, 0.5744959115982056)]
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):
     [(58241, 0.9134382009506226), (52381, 0.5564887523651123), (66819, 0.5543081760406494)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(58241, 0.7879481315612793), (55231, 0.43922340869903564), (59569, 0.41210946440696716)]


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

    TARGET (86190): «absurdly overrated vehicle for Marin county anorexics, of interest<br /><br />only to older men charmed by, but largely excluded from, the<br /><br />pretentions of the privileged young. also, of course, to historians of<br /><br />the lesser directors of the old new wave.<br /><br />the mannered failure of the principal actress to - well - act is<br /><br />insupportable, and the annoyance is not decreased by a sense<br /><br />that the director doesnt particularly care, in fact might prefer it that<br /><br />way. the looseness of his direction is only pointed by the<br /><br />conspicuousness of passages in which actual actors rebel, and<br /><br />introduce verve.<br /><br />in short, a film for the voyeur.<br /><br />»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):

    MOST (91336, 0.6613749265670776): «this movie corrupted all that was good about the hobbit.<br /><br />the music was frightening.<br /><br />the script was a product of too many late nights of white lining.<br /><br />it gives you a brief, misleading summary that not even a child could enjoy.<br /><br />at least it is better than Ralph Bakshi's LOTR.<br /><br />apologies to those who actually finished watching the movie.»

    MEDIAN (39475, 0.31675153970718384): «i'm really getting old,,am in the midst of watching this 40 year old flick,and wonder what my grandchildren will be watching 40 years from now,,its an old saying,,but they don't make em like that anymore..it's not only the story,its the music,the acting both by young and old..the cast ,it would seem,were born to play their roles,,young oliver,,old Fagin..too many to mention them all,the role played by the judge oliver stands before,i've seen in other roles over the years..the artful dodger,,Ron moody as Fagin,,Mr and Mrs bumble,,the movie not only won 5 Oscars,,but took a few golden globe awards too..if you decide to see this film..do yourself a favor,,take a few if not all the children,to see this masterpiece»

    LEAST (93747, -0.07457995414733887): «Kiefer plays an undercover cop who infiltrates a gang and takes part in a diamond heist where things get messy. They find out he's a cop and he almost gets iced. So here enters Phillips as a quiet dignified Lakota Indian who teams up with the hard boiled Kiefer because the diamond baddies stole his people's sacred lance. Yes, I said sacred lance. The supporting cast don't help matters any. Bill Smitrovich appears in the one note role as Sutherland's police superior and the moment Michael Moriarty shows up it's the "Hey I'm this movie's dirty cop. Yeah me!" dance.<br /><br />The only noteworthy element to Renegades is how quick the police response time is. Anything breaks out and they're on the scene pronto! In most movies, there's enough time to kill twenty people, blow up a building and have a car chase before the police even show up.<br /><br />I should mention the whole Native American Indian sub-plot and it's attempt at mixing culture fails miserably too. It's another variation on the whole buddy cop movie. The random factor is who's going to be the white guy's partner and having it be an Native American Indian doesn't change anything. Partnering up because of the sacred lance is without a doubt the absolute worse excuse to have two people join forces in an action movie I've ever seen too. 'They stole my secret lance - now we are partners'. Give me a break. It's not even so bad, it's funny. Renegades is a sad waste of nearly two hours if you're looking for it.»


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

    target_word: 'scene"' model: Doc2Vec(dbow,d100,n5,mc2,t8) similar words:
        1. 0.44 "'movie'.<br"
        2. 0.43 'dead!<br'
        3. 0.42 'ear!'
        4. 0.42 'Miao'
        5. 0.41 'Gibb'
        6. 0.40 'thieves"'
        7. 0.40 'REQUIRES'
        8. 0.39 'outtakes,'
        9. 0.39 'Persepolis'
        10. 0.39 'anti-Catholic'

    target_word: 'scene"' model: Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.56 'kiss,'
        2. 0.55 'scene),'
        3. 0.55 'moment"'
        4. 0.54 'scene'
        5. 0.53 'ending'
        6. 0.53 'gunfight'
        7. 0.53 'bands"'
        8. 0.52 'funhouse'
        9. 0.52 'matador'
        10. 0.52 'sequence'

    target_word: 'scene"' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.56 'analogical'
        2. 0.56 'Flyer'
        3. 0.55 'Scene,'
        4. 0.54 'Ganges'
        5. 0.54 "1800's"
        6. 0.53 'explosion'
        7. 0.53 'Thames,'
        8. 0.53 'lights...'
        9. 0.53 'Aryans,'
        10. 0.53 'Awakening'


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

    Downloading analogy questions file...
    Success, questions-words.txt is available for next steps.
    Doc2Vec(dbow,d100,n5,mc2,t8): 0.00% correct (0 of 13617)
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8): 18.15% correct (2472 of 13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 18.37% correct (2501 of 13617)


Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 38 minutes  39.115 seconds)

**Estimated memory usage:**  3814 MB


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
