.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_020_howtos_run_050_doc2vec_imdb.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_020_howtos_run_050_doc2vec_imdb.py:


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

    0.47 'Shravan'
    0.44 'mecha'
    0.41 'Brewster)'
    0.41 "Almighty'."
    0.40 'wait..'


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

    0.103320 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Evaluating Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.168280 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.302680 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.102800 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.103080 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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
    0.102800 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.103080 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.103320 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.168280 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.302680 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


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

    for doc 71726...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(71726, 0.9907494783401489), (76963, 0.633242130279541), (71727, 0.6189519166946411)]
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):
     [(71726, 0.9467856884002686), (90056, 0.5752678513526917), (71727, 0.5716109275817871)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(71726, 0.893122136592865), (24169, 0.45019519329071045), (96999, 0.4162907600402832)]


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

    TARGET (12857): «Comic secret agents have made a comeback in recent years, with Mike Myers' 'Austin Powers' and Rowan Atkinson's 'Johnny English', and more recently Steve Carell in the big-screen version of the hit '60's show 'Get Smart!'.<br /><br />Back in 1974, it was David Jason who was wearing a shoulder holster and carrying an attaché case full of documents marked 'Classified'.<br /><br />'The Top Secret Life Of Edgar Briggs' was his first starring role in a sitcom, after years of being a supporting actor in such shows as 'Six Dates With Barker', the 'Doctor' series, and 'Hark At Barker'.<br /><br />Humphrey Barclay had found him working in a pier theatre in Bournemouth and was sufficiently impressed to include him alongside Michael Palin, Terry Jones and Eric Idle in the children's comedy show 'Do Not Adjust Your Set!'.<br /><br />'T.T.S.L.O.E.B' cast Jason as 'Edgar Briggs', a well-meaning but incompetent agent for the Secret Intelligence Service. Whereas John Steed wore a bowler hat, Briggs had a trilby. Whereas Napoleon Solo carried a radio pen, Briggs owned a pipe. Objects fell to bits in his hands. He read Confidential documents in bed while his wife ( Barbara Angell ) perused Woman's Own ( on one occasion it would be the other way round ). When he tracked a pair of Russian agents to a heliport, he accidentally switched on the airport's Tannoy system, and broadcast his plans to capture them! When he hid on a train so as to photograph a meeting between an S.I.S. man and his enemy-contact, it moved off with him aboard and took him straight to Brighton! When he tried to organise the defection of a female Russian scientist, he took a 'short cut' to elude his pursuers, only to wind up hopelessly lost in a car park. Yet, like 'Inspector Clouseau', he always seemed to come out on top at the end, much to the dismay of his colleagues.<br /><br />As previously mentioned, he was married. His wife Jennifer was understanding about the sort of work he did. Though they had a row once which resulted in her yelling at him from the window of their high-rise flat: "Secret Service this, Secret Service that! You never stop thinking about the Secret Service!". He shouted back: "Think of the neighbours! They're not supposed to know I'm in the Secret Service!".<br /><br />Briggs was part of a team of agents whose number included 'Coronation Street' villain Mark Eden ( he was the psychotic Alan Bradley ) as 'Spencer', Michael Stainton as 'Buxton', and 'Doctor At Sea''s Elisabeth Counsell as the lovely 'Cathy Strong'. They answered to 'The Commander', played by the late Noel Coleman. The Commander was kidnapped in one episode, leaving Briggs temporarily in charge of the S.I.S. - which naturally horrified everyone.<br /><br />This hilarious show was by Richard Laing and Bernard McKenna, who had written for the 'Doctor' series. Rather than spoof Bond, it was more of a send-up of the serious spy shows such as 'Callan' ( though it had a Bond-style theme tune ). Furtive meetings in underground car parks, code-breaking, stolen missile plans, that kind of thing. Jason brought a lot of energy to the role, doing a lot of his own stunts, such as Briggs falling off a ladder whilst decorating his flat, and tumbling down a hill in a wastepaper bin, and were reminiscent of those to be found in the 'Pink Panther' films.<br /><br />'Briggs' had all the ingredients to be a smash-hit. Unfortunately, it was not networked. In the London area, it was put out on Sundays at 7.25 P.M. where it was trounced in the ratings by the B.B.C.'s soapy drama 'The Brothers'. It was then moved to Fridays at 7 P.M. because I.T.V. wanted to showcase its latest American import - the T.V. version of 'Planet Of The Apes'. Briggs never found an audience. A similar fate befell Jason's next major show: 1976's 'Lucky Feller'. It was not until 1977 and 'A Sharp Intake Of Breath' that he found his first successful solo vehicle.<br /><br />You can see the title sequence ( along with two brief excerpts in German! ) for this series on YouTube. Unfortunately, that is all you can see. Jason will not permit his early starring shows either to be repeated or released on D.V.D. A great shame. For the moment, however, Edgar Briggs' life will have to remain top secret.<br /><br />CODA: I have seen a number of episodes recently and I'm pleased to say it stands up incredibly well.»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/c,d100,n5,w5,mc2,t8):

    MOST (15983, 0.5422624349594116): «I very nearly did not see 'Hi-De-Hi!'. I think it must have been the title that put me off. In those days, the Welsh language editions of 'The Radio Times' only used to print titles of certain shows without imparting a scrap of information as to what they were actually about. 'Hi-De-Hi!' suggested to me a bad quiz show hosted by Leslie Crowther or worse an inane U.S. import. But I managed to catch a later episode, and was surprised to find it written by Jimmy Perry and David Croft.<br /><br />As was the case with 'Dad's Army' and 'It Ain't Half Hot Mum', Perry based it on personal experiences, in this case his time at a Butlins' holiday camp. Before cheap air travel came along in the '60's, these camps sprang up along British coastlines, providing entertainment for working class families and earning millions for their owners.<br /><br />( As a matter of interest, I worked in one such camp in the '80's as a chef - Barry Island, South Wales - known to all and sundry as 'Shag Land' for reasons I won't go into! )<br /><br />Set in the late '50's, it began with university academic Jeffrey Fairbrother ( Simon Cadell ) taking over as the entertainments manager of Maplin's, a job he was ill equipped to handle. His staff included resident comic Ted Bovis ( Paul Shane ), his sidekick Spike ( Jeffrey Holland ), miserable Punch and Judy man Mr.Partridge ( Leslie Dwyer ), snobby ballroom dancers Barry ( Barry Howard ) and Yvonne Stuart-Hargreaves ) Diane Holland ), and the unforgettable Gladys Pugh ( Ruth Madoc ), who lusted after Fairbrother at every opportunity. Bubbly Su Pollard stole the show though as cleaner Peggy Ollerenshaw, whose driving ambition was to be a 'Yellowcoat' ( all the important staff members wore them ). A number of sexy girls occupied these coats too, most notably Nikki Kelly's 'Sylvia' and statuesque Rikki Howard's 'Betty'. We never saw Joe Maplin, the owner. He communicated to his staff in the form of ungrammatical missives, which poor Jeffrey was forced to read aloud. "Hi-De-Hi!" was the campers' greeting, usually met with the equally inane 'Ho-De-Ho!. <br /><br />One fan was the late Sir Fred Pontin, who told Perry and Croft that he recognised most of the characters from real life.<br /><br />I always found Bovis the most convincing of these as well as the most tragic, like Archie Rice he was the comedian whose big break never came, reduced to cracking corny gags for the amusement of drunken late-night audiences. He took advantage of his position to indulge in a few perks, and in one memorable episode Fairbrother's patience snapped and he sounded him out: "Lies, Ted! All lies!".<br /><br />As with every other Perry/Croft series, the cast were excellent, particularly Cadell and Shane. Ruth Madoc's prissy 'Gladys' got on my nerves ( no wonder Anne Robinson hates the Welsh! ), but Leslie Dwyer's misanthropic 'Mr.Partridge' and Felix Bowness' jockey 'Fred Qulley' more than compensated. <br /><br />The visual gag everyone remembers is drunken Mr.Partridge spotting a pantomime horse riding a real one along the beach. Looking at the bottle of whiskey in his hand, he decides to stick with it and instead throws away the banana he had been eating! <br /><br />With its frothy blend of '50's nostalgia and saucy gags, 'Hi-De'Hi' was a big hit for B.B.C.-1 in the '80's, resulting in a massive increase in bookings for Butlins and Pontins. It went downhill when Cadell left to return to the theatre though. I never took to his replacement, Squadron Leader Clive Dempster ( David Griffin ). Worse, Leslie Dwyer's death robbed the show of one of its best characters. Kenneth Connor was brought in to replace him as 'Uncle Sammy'.<br /><br />The period setting occasionally caused problems; in one episode, Sylvia and Betty had to dive into the pool to rescue Peggy who for some reason was dressed as a shark. The revealing costumes they wore were wrong for that era. Still they looked great in them so who's complaining? In another, Ted sang the Tom Jones hit 'Delilah' to campers. It was not composed ( by Les Reed and Barry Mason, incidentally ) until 1968.<br /><br />Maplins closed its doors in 1988, and the last shot was that of Peggy ( now a Yellowcoat ) all alone in the camp, jumping into the air and shouting ( what else? ) 'Hi-De-Hi!'. <br /><br />I don't rate it as highly as Perry and Croft's other shows but its popularity is undeniable. It was probably one of the last British sitcoms to generate tremendous public affection, mainly because it featured likable characters in a recognisable setting. Goodnight campers!»

    MEDIAN (22846, 0.004338208585977554): «Now, I am not prone to much emotion, but I cried seeing this movie. It certainly has more appeal among blacks than other ethnic groups, but there is something here for everyone. The classic song "It's so Hard to Say Goodbye" really makes this one worth watching at least once.»

    LEAST (98632, -0.43389996886253357): «I have seen A LOT of bad movies. I watch them on a regular basis and even some of the really bad ones, I can get through-ones that Mystery Sceince Theater 3000 wouldn't even touch. This is bar none the worst "movie" ever made. I use the term movie loosely in this case because it is more like a group of friends that got together and made a video, but sure enough it has some Troma in it. Even if you like the mindless Troma stuff, this one will make you wince in pain. It IS that bad!! The cover has Lilith Stabs on it (a hot goth/punk type girl of Bad Movie Police fame) in a little outfit with a huge gun. Very misleading. She is NOT in the movie and there are no gun toting babes gunning down zombies. Instead you get to watch an ugly chick with a chocolate stained face annoy the hell out of you with over eccentric bad acting and lisp. The imitation Cosby/Don King character-Bonejack-is something you would laugh at if it were your friend, otherwise it is just lame. The filmmakers felt free to make plenty of inside jokes (the reappearing ice cream poster in tons of shots) without letting the audience in on their little jokes. If this can truly be called a movie versus just a video that some idiots threw together, then yes. This is THE worst movie I have ever seen. It even beats out the Tempe classics such as Humanoids from Atlantis and Robot Ninja. At least with James L Edwards you know what you are getting into. I wonder how many suckers fell for the false advertising on the box cover. Well live and learn the old "You can't judge a book by its cover". Well Mulva didn't kick any zombie ass-but this movie sure kicked mine.»


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

    target_word: 'Ops' model: Doc2Vec(dbow,d100,n5,mc2,t8) similar words:
        1. 0.43 'not.Therefore,'
        2. 0.42 'Eskimo,'
        3. 0.42 'f'
        4. 0.41 'side-note,'
        5. 0.40 'culture?'
        6. 0.39 'couleurs:'
        7. 0.39 'Calamine'
        8. 0.39 'endlessly'
        9. 0.39 'camp-fest'
        10. 0.39 'huge!'

    target_word: 'Ops' model: Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.55 'counter-terrorism'
        2. 0.54 'analyst,'
        3. 0.51 'Kumjorn.<br'
        4. 0.50 'operative.'
        5. 0.49 'McGavin)'
        6. 0.49 'ex-Navy'
        7. 0.48 '(Arnold'
        8. 0.48 'script!!!'
        9. 0.47 'non-profit'
        10. 0.47 'Digicorp,'

    target_word: 'Ops' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.63 'Forces'
        2. 0.61 'ops'
        3. 0.60 'Operations'
        4. 0.60 'ex-military'
        5. 0.58 'Treasury'
        6. 0.57 'ex-Confederate'
        7. 0.57 'ex-army'
        8. 0.56 'Army'
        9. 0.56 'Orthodox'
        10. 0.55 'Olympic'


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
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8): 18.60% correct (2533 of 13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 17.91% correct (2439 of 13617)


Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 37 minutes  57.705 seconds)

**Estimated memory usage:**  4092 MB


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
