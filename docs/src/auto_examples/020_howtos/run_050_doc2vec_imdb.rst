.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_020_howtos_run_050_doc2vec_imdb.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_020_howtos_run_050_doc2vec_imdb.py:


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

    0.45 'extendedly'
    0.43 'tribes?'
    0.42 'interviews,'
    0.42 'Bilson'
    0.41 "'Union"


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

    0.104640 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Evaluating Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.169400 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.306280 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.103760 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.104920 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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
    0.103760 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.104640 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.104920 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.169400 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.306280 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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

    for doc 86522...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(86522, 0.9909253120422363), (85826, 0.680814266204834), (27682, 0.673947811126709)]
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):
     [(86522, 0.9615100622177124), (98918, 0.5803713798522949), (19233, 0.5563608407974243)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(86522, 0.9066746234893799), (87192, 0.43959346413612366), (36147, 0.4354143738746643)]


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

    TARGET (16202): «David Attenborough brings his fascination of wild life, this time the creatures under the sea, in this extraordinary 8-episode trip to all the animals under the sea!<br /><br />The cinematography is astounding, bringing to the screen truly breathtaking footage of those whales! But the best thing about it, as well as seeing each episode, is how they made it! Whether it is making models of creatures, or those impressive shots of the whales, they explain to you in about 10 minutes how they did it!<br /><br />2001 had some great tv shows to our screen. But, in contrast to this documentary gem, they make them pretty lame! But to even boast this documentary series as the best tv series of 2001 just does not sum up the sheer brilliance that this series provides in quality entertainment!<br /><br />Overall, this is the best TV series of 2001, with no competition, and, maybe, the best TV series of 2000s!»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):

    MOST (81274, 0.6267575025558472): «This short parody of the Council of Elrond scene, included as an "easter egg" on the "The Lord of the Rings: The Fellowship of the Ring" Extended Edition DVD, is really not worth your time. I urge you to make better use of your time than to watch this crude, unfunny, MTV-made parody, especially when there are so many great things to explore on the Extended Edition DVD. After I finished watching it, I wondered, "This is it? This is what they give us as an easter egg on the extended version DVD of one of the best movies in recent memory? Does Peter Jackson really think that this is good enough to put on the otherwise excellent DVD of a truly spectacular film?" As another reviewer said, it's like a train wreck, you want to look away and stop watching but you can't because it's so bad. Good parodies can be incredibly funny, but this, however, is not one of them.»

    MEDIAN (28682, 0.2943798303604126): «It's a poor film, but I must give it to the lead actress in this one....Francine Forbes. She appeared to be acting the least and I personally thought she was kind of cute. Too bad she only appears in one other film in the database. Besides that, the film is filled with laughable gore and fakey death scenes. People get stabbed and they GUSH like 2 gallons of blood! But, if you like to watch poor horror films, I recommend this one highly.»

    LEAST (38756, -0.12590399384498596): «Match 1: Tag Team Table Match Bubba Ray and Spike Dudley vs Eddie Guerrero and Chris Benoit Bubba Ray and Spike Dudley started things off with a Tag Team Table Match against Eddie Guerrero and Chris Benoit. According to the rules of the match, both opponents have to go through tables in order to get the win. Benoit and Guerrero heated up early on by taking turns hammering first Spike and then Bubba Ray. A German suplex by Benoit to Bubba took the wind out of the Dudley brother. Spike tried to help his brother, but the referee restrained him while Benoit and Guerrero ganged up on him in the corner. With Benoit stomping away on Bubba, Guerrero set up a table outside. Spike dashed into the ring and somersaulted over the top rope onto Guerrero on the outside! After recovering and taking care of Spike, Guerrero slipped a table into the ring and helped the Wolverine set it up. The tandem then set up for a double superplex from the middle rope which would have put Bubba through the table, but Spike knocked the table over right before his brother came crashing down! Guerrero and Benoit propped another table in the corner and tried to Irish Whip Spike through it, but Bubba dashed in and blocked his brother. Bubba caught fire and lifted both opponents into back body drops! Bubba slammed Guerrero and Spike stomped on the Wolverine from off the top rope. Bubba held Benoit at bay for Spike to soar into the Wassup! headbutt! Shortly after, Benoit latched Spike in the Crossface, but the match continued even after Spike tapped out. Bubba came to his brother's rescue and managed to sprawl Benoit on a table. Bubba leapt from the middle rope, but Benoit moved and sent Bubba crashing through the wood! But because his opponents didn't force him through the table, Bubba was allowed to stay in the match. The first man was eliminated shortly after, though, as Spike put Eddie through a table with a Dudley Dawg from the ring apron to the outside! Benoit put Spike through a table moments later to even the score. Within seconds, Bubba nailed a Bubba Bomb that put Benoit through a table and gave the Dudleys the win! Winner: Bubba Ray and Spike Dudley<br /><br />Match 2: Cruiserweight Championship Jamie Noble vs Billy Kidman Billy Kidman challenged Jamie Noble, who brought Nidia with him to the ring, for the Cruiserweight Championship. Noble and Kidman locked up and tumbled over the ring, but raced back inside and grappled some more. When Kidman thwarted all Noble's moves, Noble fled outside the ring where Nidia gave him some encouragement. The fight spread outside the ring and Noble threw his girlfriend into the challenger. Kidman tossed Nidia aside but was taken down with a modified arm bar. Noble continued to attack Kidman's injured arm back in the ring. Kidman's injured harm hampered his offense, but he continued to battle hard. Noble tried to put Kidman away with a powerbomb but the challenger countered into a facebuster. Kidman went to finish things with a Shooting Star Press, but Noble broke up the attempt. Kidman went for the Shooting Star Press again, but this time Noble just rolled out of harm's way. Noble flipped Kidman into a power bomb soon after and got the pin to retain his WWE Cruiserweight Championship! Winner: Jamie Noble<br /><br />Match 3: European Championship William Regal vs Jeff Hardy William Regal took on Jeff Hardy next in an attempt to win back the European Championship. Jeff catapulted Regal over the top rope then took him down with a hurracanrana off the ring apron. Back in the ring, Jeff hit the Whisper in the wind to knock Regal for a loop. Jeff went for the Swanton Bomb, but Regal got his knees up to hit Jeff with a devastating shot. Jeff managed to surprise Regal with a quick rollup though and got the pin to keep the European Championship! Regal started bawling at seeing Hardy celebrate on his way back up the ramp. Winner: Jeff Hardy<br /><br />Match 4: Chris Jericho vs John Cena Chris Jericho had promised to end John Cena's career in their match at Vengeance, which came up next. Jericho tried to teach Cena a lesson as their match began by suplexing him to the mat. Jericho continued to knock Cena around the ring until his cockiness got the better of him. While on the top rope, Jericho began to showboat and allowed Cena to grab him for a superplex! Cena followed with a tilt-a-whirl slam but was taken down with a nasty dropkick to the gut. The rookie recovered and hit a belly to belly suplex but couldn't put Y2J away. Jericho launched into the Lionsault but Cena dodged the move. Jericho nailed a bulldog and then connected on the Lionsault, but did not go for the cover. He goaded Cena to his feet so he could put on the Walls of Jericho. Cena had other ideas, reversing the move into a pin attempt and getting the 1-2-3! Jericho went berserk after the match. Winner: John Cena<br /><br />Match 5: Intercontinental Championship RVD vs Brock Lesnar via disqualification The Next Big Thing and Mr. Pay-Per-View tangled with the Intercontinental Championship on the line. Brock grabbed the title from the ref and draped it over his shoulder momentarily while glaring at RVD. Van Dam 's quickness gave Brock fits early on. The big man rolled out of the ring and kicked the steel steps out of frustration. Brock pulled himself together and began to take charge. With Paul Heyman beaming at ringside, Brock slammed RVD to the hard floor outside the ring. From there, Brock began to overpower RVD, throwing him with ease over the top rope. RVD landed painfully on his back, then had to suffer from having his spine cracked against the steel ring steps. The fight returned to the ring with Brock squeezing RVD around the ribs. RVD broke away and soon after leveled Brock with a kick to the temple. RVD followed with the Rolling Thunder but Brock managed to kick out after a two-count. The fight looked like it might be over soon as RVD went for a Five-Star Frog Splash. Brock, though, hoisted Van Dam onto his shoulder and went for the F-5, but RVD whirled Brock into a DDT and followed with the Frog Splash! He went for the pin, but Heyman pulled the ref from the ring! The ref immediately called for a disqualification and soon traded blows with Heyman! After, RVD leapt onto Brock from the top rope and then threatened to hit the Van Terminator! Heyman grabbed RVD's leg and Brock picked up the champ and this time connected with the F-5 onto a steel chair! Winner: RVD<br /><br />Match 6: Booker T vs the Big Show Booker T faced the Big Show one-on-one next. Show withstood Booker T's kicks and punches and slapped Booker into the corner. After being thrown from the ring, Booker picked up a chair at ringside, but Big Show punched it back into Booker's face. Booker tried to get back into the game by choking Show with a camera cable at ringside. Booker smashed a TV monitor from the Spanish announcers' position into Show's skull, then delivered a scissors kick that put both men through the table! Booker crawled back into the ring and Big Show staggered in moments later. Show grabbed Booker's throat but was met by a low blow and a kick to the face. Booker climbed the top rope and nailed a somersaulting leg drop to get the pin! Winner: Booker T<br /><br />Announcement: Triple H entered the ring to a thunderous ovation as fans hoped to learn where The Game would end up competing. Before he could speak, Eric Bishoff stopped The Game to apologize for getting involved in his personal business. If Triple H signed with RAW, Bischoff promised his personal life would never come into play again. Bischoff said he's spent the past two years networking in Hollywood. He said everyone was looking for the next breakout WWE Superstar, and they were all talking about Triple H. Bischoff guaranteed that if Triple H signed with RAW, he'd be getting top opportunities coming his way. Stephanie McMahon stepped out to issue her own pitch. She said that because of her personal history with Triple H, the two of them know each other very well. She said the two of them were once unstoppable and they can be again. Bischoff cut her off and begged her to stop. Stephanie cited that Triple H once told her how Bischoff said Triple H had no talent and no charisma. Bischoff said he was young at the time and didn't know what he had, but he still has a lot more experience that Stephanie. The two continued to bicker back and forth, until Triple H stepped up with his microphone. The Game said it would be easy to say "screw you" to either one of them. Triple H went to shake Bischoff's hand, but pulled it away. He said he would rather go with the devil he knows, rather than the one he doesn't know. Before he could go any further, though, Shawn Michaels came out to shake things up. HBK said the last thing he wanted to do was cause any trouble. He didn't want to get involved, but he remembered pledging to bring Triple H to the nWo. HBK said there's nobody in the world that Triple H is better friends with. HBK told his friend to imagine the two back together again, making Bischoff's life a living hell. Triple H said that was a tempting offer. He then turned and hugged HBK, making official his switch to RAW! Triple H and HBK left, and Bischoff gloated over his victory. Bischoff said the difference between the two of them is that he's got testicles and she doesn't. Stephanie whacked Bischoff on the side of the head and left!<br /><br />Match 7: Tag Team Championship Match Christian and Lance Storm vs Hollywood Hogan and Edge The match started with loud "USA" chants and with Hogan shoving Christian through the ropes and out of the ring. The Canadians took over from there. But Edge scored a kick to Christian's head and planted a facebuster on Storm to get the tag to Hogan. Hogan began to Hulk up and soon caught Christian with a big boot and a leg drop! Storm broke up the count and Christian tossed Hogan from the ring where Storm superkicked the icon. Edge tagged in soon after and dropped both opponents. He speared both of them into the corner turnbuckles, but missed a spear on Strom and hit the ref hard instead. Edge nailed a DDT, but the ref was down and could not count. Test raced down and took down Hogan then leveled Edge with a boot. Storm tried to get the pin, but Edge kicked out after two. Riksihi sprinted in to fend off Test, allowing Edge to recover and spear Storm. Christian distracted the ref, though, and Y2J dashed in and clocked Edge with the Tag Team Championship! Storm rolled over and got the pinfall to win the title! Winners and New Tag Team Champions: Christian and Lance Storm<br /><br />Match 8: WWE Undisputed Championship Triple Threat Match. The Rock vs Kurt Angle and the Undertaker Three of WWE's most successful superstars lined up against each other in a Triple Threat Match with the Undisputed Championship hanging in the balance. Taker and The Rock got face to face with Kurt Angle begging for some attention off to the side. He got attention in the form of a beat down form the two other men. Soon after, Taker spilled out of the ring and The Rock brawled with Angle. Angle gave a series of suplexes that took down Rock, but the Great One countered with a DDT that managed a two-count. The fight continued outside the ring with Taker coming to life and clotheslining Angle and repeatedly smacking The Rock. Taker and Rock got into it back into the ring, and Taker dropped The Rock with a sidewalk slam to get a two-count. Rock rebounded, grabbed Taker by the throat and chokeslammed him! Angle broke up the pin attempt that likely would have given The Rock the title. The Rock retaliated by latching on the ankle lock to Kurt Angle. Angle reversed the move and Rock Bottomed the People's Champion. Soon after, The Rock disposed of Angle and hit the People's Elbow on the Undertaker. Angle tried to take advantage by disabling the Great One outside the ring and covering Taker, who kicked out after a two count. Outside the ring, Rock took a big swig from a nearby water bottle and spewed the liquid into Taker's face to blind the champion. Taker didn't stay disabled for long, and managed to overpower Rock and turn his attention to Angle. Taker landed a guillotine leg drop onto Angle, laying on the ring apron. The Rock picked himself up just in time to break up a pin attempt on Kurt Angle. Taker nailed Rock with a DDT and set him up for a chokeslam. ANgle tried sneaking up with a steel chair, but Taker caught on to that tomfoolery and smacked it out of his hands. The referee got caught in the ensuing fire and didn't see Angle knock Taker silly with a steel chair. Angle went to cover Taker as The Rock lay prone, but the Dead Man somehow got his shoulder up. Angle tried to pin Rock, but he too kicked out. The Rock got up and landed Angle in the sharpshooter! Angle looked like he was about to tap, but Taker kicked The Rock out of the submission hold. Taker picked Rock up and crashed him with the Last Ride. While the Dead Man covered him for the win, Angle raced in and picked Taker up in the ankle lock! Taker went delirious with pain, but managed to counter. He picked Angle up for the last ride, but Angle put on a triangle choke! It looked like Taker was about to pass out, but The Rock broke Angle's hold only to find himself caught in the ankle lock. Rock got out of the hold and watched Taker chokeslam Angle. Rocky hit the Rock Bottom, but Taker refused to go down and kicked out. Angle whirled Taker up into the Angle Slam but was Rock Bottomed by the Great One and pinned! Winner and New WWE Champion: The Rock<br /><br />~Finally there is a decent PPV! Lately the PPV weren't very good, but this one was a winner. I give this PPV a A-<br /><br />»


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
        1. 0.41 'Toufexis'
        2. 0.41 '/>may'
        3. 0.40 'bookended'
        4. 0.40 '"we'
        5. 0.40 '4Kids'
        6. 0.39 'Jade'
        7. 0.39 'Henchmen'
        8. 0.39 'dangler'
        9. 0.39 'one-liners...'
        10. 0.39 'McCool\'s"'

    target_word: 'scene"' model: Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.54 'scene),'
        2. 0.53 '"twist"'
        3. 0.53 '"joke"'
        4. 0.53 'steps,'
        5. 0.52 'nightmare...from'
        6. 0.52 'matador'
        7. 0.52 'gunfight'
        8. 0.52 'victim,'
        9. 0.52 'scene'
        10. 0.51 'expletive'

    target_word: 'scene"' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.62 'tournament'
        2. 0.58 '(1956)'
        3. 0.58 'Escher'
        4. 0.57 'featurette,'
        5. 0.56 'asteroid'
        6. 0.56 'Scene,'
        7. 0.56 'Pyramids,'
        8. 0.56 'ambush,'
        9. 0.55 'Cannon)'
        10. 0.55 'eagle'


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
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8): 18.45% correct (2512 of 13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 18.00% correct (2451 of 13617)


Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 35 minutes  31.710 seconds)

**Estimated memory usage:**  3854 MB


.. _sphx_glr_download_auto_examples_020_howtos_run_050_doc2vec_imdb.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_050_doc2vec_imdb.py <run_050_doc2vec_imdb.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_050_doc2vec_imdb.ipynb <run_050_doc2vec_imdb.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
