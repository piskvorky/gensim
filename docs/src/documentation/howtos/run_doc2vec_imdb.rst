.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_howtos_run_doc2vec_imdb.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_howtos_run_doc2vec_imdb.py:


How to Apply Doc2Vec to Reproduce the 'Paragraph Vector' paper
==============================================================

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

    0.104320 Doc2Vec(dbow,d100,n5,mc2,t8)

    Training Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Evaluating Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.169080 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    Training Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    Evaluating Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.304520 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)

    0.103800 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)


    Evaluating Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)

    0.104760 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)




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
    0.103800 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.104320 Doc2Vec(dbow,d100,n5,mc2,t8)
    0.104760 Doc2Vec(dbow,d100,n5,mc2,t8)+Doc2Vec(dm/c,d100,n5,w5,mc2,t8)
    0.169080 Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8)
    0.304520 Doc2Vec(dm/c,d100,n5,w5,mc2,t8)



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

    for doc 8260...
    Doc2Vec(dbow,d100,n5,mc2,t8):
     [(8260, 0.982399582862854), (14362, 0.6372007727622986), (1126, 0.6309436559677124)]
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8):
     [(8260, 0.8961429595947266), (60918, 0.5842142701148987), (49563, 0.5790035724639893)]
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8):
     [(8260, 0.8670176267623901), (11056, 0.4305807054042816), (13621, 0.4183669090270996)]



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

    TARGET (99757): «I couldn't make it through the whole thing. It just wasn't worth my time. Maybe one-fourth of the dialogue would have been worth listening to (or reading -- since I don't understand French) if the pseudo-profundity and pseudo-wittiness of the other three-fourths of the film were deleted. Then it could be made into a short maybe 13 or 15 min long and then it might be all right.<br /><br />I don't know why this movie even pretends to utilize actors. Actors are used as narrators of the script and little more. I could swear a whole 20-30 minutes of the film went by showing actors from behind while they talked and from across the street while they walked or sitting in low lighting close up but so that you could not see the expressions on their faces nor their eyes. There was little or no interaction between the actors on the screen except the most superficial for the most part.<br /><br />Some of the lines were as profound (or lame, depending on your viewpoint) as those in Forest "Life is like a box of chocolates" Gump. Other pseudo-profundities were simply sad or dumb or poetic (depending again on your viewpoint), but singularly uninspiring.<br /><br />Visually this film is INCREDIBLY boring, especially with the lack of actors. In fact some minutes of this film showed simply a black screen with the white subtitles and French audio. Altogether sophomoric. Don't waste your time.<br /><br />If you like GOOD movies that are stimulating and profound just from listening to conversation while enjoying good actors, check out RICHARD LINKLATER's "Before Sunset" -- or make a double feature of it and watch "Before Sunrise" first. At least these films are interesting and enjoyable, which is much more than I can say about IN PRAISE OF LOVE (Éloge de l'amour). I give this film 2 out of 10 stars. Not quite offensive enough to rate 1 for "awful" (such as "The Devils" with Oliver Reed and Vanessa Redgrave). If you still want to watch it, go ahead. But don't say I didn't warn you!!!»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/c,d100,n5,w5,mc2,t8):

    MOST (53503, 0.49092066287994385): «Verrrry nice. I think this is the first good black movie I have seen that didn't need to be but simply was. Well-known, medium-powered Black actors and actresses in a movie that didn't focus on the Black experience. The BEST part... the whole movie was done with thought-out class. The story wasn't about brothers from the 'hood trying to make it in the White Man's world or some racially-charged Spike Lee joint. Think of a great episode of the Cosby Show re-written for an adult audience, bearing a PG-13 rating and you know how you'll feel when you leave the theater. It's about time.»

    MEDIAN (9222, 0.00015147030353546143): «Right this moment I am watching this movie on TV here in Tokyo. Beautiful scenery, beautiful sets of biblical proportions, beautiful costumes, beautiful color, beautiful Gina. Great climactic scene when God destroys the Sheban idol and a lot more with de Millean thunderbolts at the moment when Yul and Gina are about to consummate their love. Yul does a halfway decent job of delivering his lines, though he sounds a lot like Yul delivering his lines as Ramses or Taras Bulba. George Sanders sounds like George Sanders playing George Sanders. Given the limited range of acting she is asked to display in this role, Gina does a good job, though by the time the movie ends, she is completely converted into a demure remorseful lass and looks likes she might be playing in a biography of Mother Teresa. I guess thunderbolts will do that to you, but it is almost breathtaking how quickly she jettisons her own beliefs for her new religion. The supporting players are mostly awful, lacking credible emotion and timing. The usual big battle scenes, what passed for lascivious dances in 1959, and an orchestra blasting out plenty of trumpet calls behind a huge chorus singing lots of "Ah's", but none of it quite of topnotch Hollywood quality. The final swordfight between Brynner and Sanders is at the laughingly low skill level of a junior high school play. The film is one big piece of eye candy but not much more.»

    LEAST (20517, -0.40096956491470337): «Any Batman fan will know just how great the films are, they've been a major success. Batman Returns however is by far the best film in the series. A combination of excellent directing, brilliant acting and settings makes this worthy of watching on a night in.<br /><br />Tim Burton, who directed this movie, has specifically made sure that this film gives a realistic atmosphere and he's done a great job. Danny Devito (Penguin man) is a man who has inherited penguin characteristics as a baby, and grown up to become a hideous and ugly...thing! Michelle Pfiffer plays the sleek and very seducing 'Catwoman' after cats had given her there genes from being bitten. The result in both the character changes is excellent and both Catwoman and Penguin man play a very important role in this excellent film. The mysterious Catwoman is great fun to watch - her classic sayings and a funny part in which skips with her whip in a jewelry shop adds such fun to the film. Danny Devito also does well, his ability to impersonate some strange creature was vital, and he adds a great atmosphere to the film that takes us back to the dull sewers where he lives.<br /><br />You can't forget Batman though. Micheal Keaton once again pulls of a comfortable performance, and shows us a different side to Batman. His affection is let loose when he confronts Catwoman at the end of the film, and his meetings with her when she's a normal person, Selina Kyle, result in him being seduced badly in his own home. There's a clever part after this when they leave, and the film is full of great scenes. Its worth noting that Bruce Wayne's Bat mobile is not used as much as in the other Batman films, as close combat and story telling scenes make up this film.<br /><br />The winter setting is created perfectly in Gotham City with most of the scenes being set at night, and with the town being filled with snow. Therefore, if you watch this film during the summer like I have, it doesn't feel the same. Best watch it during the winter.<br /><br />Overall, its an amazing movie. All the credit goes to Tom Burton and the cast, they've done an incredible job.»




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

    target_word: 'yet?' model: Doc2Vec(dbow,d100,n5,mc2,t8) similar words:
        1. 0.43 'crestfallen'
        2. 0.43 "babies'"
        3. 0.42 'earth)'
        4. 0.41 'meh,'
        5. 0.41 'Pryor.<br'
        6. 0.40 'hole"'
        7. 0.40 'Trumpy.'
        8. 0.40 '"hold'
        9. 0.40 'CHAMPIONSHIP.'
        10. 0.40 'antagonists'

    target_word: 'yet?' model: Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8) similar words:
        1. 0.61 'already.'
        2. 0.59 'yet.'
        3. 0.58 "can't!"
        4. 0.57 'loose?'
        5. 0.57 '?'
        6. 0.57 'One.'
        7. 0.57 'Tube.'
        8. 0.56 'whatsoever!'
        9. 0.56 'afterwords.'
        10. 0.56 'anyway!'

    target_word: 'yet?' model: Doc2Vec(dm/c,d100,n5,w5,mc2,t8) similar words:
        1. 0.61 'already.'
        2. 0.59 'anyway!'
        3. 0.59 'tremendously.'
        4. 0.59 'anyways.'
        5. 0.58 'sequence).'
        6. 0.58 'too!!!'
        7. 0.57 'already!'
        8. 0.57 'indeed...'
        9. 0.57 'before....'
        10. 0.55 'BRAVO.'




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
    Doc2Vec("alpha=0.05",dm/m,d100,n5,w10,mc2,t8): 18.58% correct (2530 of 13617)
    Doc2Vec(dm/c,d100,n5,w5,mc2,t8): 17.58% correct (2394 of 13617)



Even though this is a tiny, domain-specific dataset, it shows some meager
capability on the general word analogies – at least for the DM/mean and
DM/concat models which actually train word vectors. (The untrained
random-initialized words of the DBOW model of course fail miserably.)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 40 minutes  26.360 seconds)

**Estimated memory usage:**  4146 MB


.. _sphx_glr_download_auto_examples_howtos_run_doc2vec_imdb.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_doc2vec_imdb.py <run_doc2vec_imdb.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_doc2vec_imdb.ipynb <run_doc2vec_imdb.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
