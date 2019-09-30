.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_030_fasttext.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_030_fasttext.py:


FastText Model
==============

Introduces Gensim's fastText model and demonstrates its use on the Lee Corpus.

.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)







Here, we'll learn to work with fastText library for training word-embedding
models, saving & loading them and performing similarity operations & vector
lookups analogous to Word2Vec.

When to use FastText?
---------------------

The main principle behind `fastText <https://github.com/facebookresearch/fastText>`_ is that the morphological structure of a word carries important information about the meaning of the word, which is not taken into account by traditional word embeddings, which train a unique word embedding for every individual word. This is especially significant for morphologically rich languages (German, Turkish) in which a single word can have a large number of morphological forms, each of which might occur rarely, thus making it hard to train good word embeddings.


fastText attempts to solve this by treating each word as the aggregation of its subwords. For the sake of simplicity and language-independence, subwords are taken to be the character ngrams of the word. The vector for a word is simply taken to be the sum of all vectors of its component char-ngrams.


According to a detailed comparison of Word2Vec and FastText in `this notebook <Word2Vec_FastText_Comparison.ipynb>`_\ , fastText does significantly better on syntactic tasks as compared to the original Word2Vec, especially when the size of the training corpus is small. Word2Vec slightly outperforms FastText on semantic tasks though. The differences grow smaller as the size of training corpus increases.


Training time for fastText is significantly higher than the Gensim version of Word2Vec (\ ``15min 42s`` vs ``6min 42s`` on text8, 17 mil tokens, 5 epochs, and a vector size of 100).


fastText can be used to obtain vectors for out-of-vocabulary (OOV) words, by summing up vectors for its component char-ngrams, provided at least one of the char-ngrams was present in the training data.


Training models
---------------


For the following examples, we'll use the Lee Corpus (which you already have if you've installed gensim) for training our model.





.. code-block:: default

    from pprint import pprint as print
    from gensim.models.fasttext import FastText as FT_gensim
    from gensim.test.utils import datapath

    # Set file names for train and test data
    corpus_file = datapath('lee_background.cor')

    model = FT_gensim(size=100)

    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)

    # train the model
    model.train(
        corpus_file=corpus_file, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words
    )

    print(model)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastText object at 0x7f5760e60278>


Training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^


Hyperparameters for training the model follow the same pattern as Word2Vec. FastText supports the following parameters from the original word2vec:

- model: Training architecture. Allowed values: `cbow`, `skipgram` (Default `cbow`)
- size: Size of embeddings to be learnt (Default 100)
- alpha: Initial learning rate (Default 0.025)
- window: Context window size (Default 5)
- min_count: Ignore words with number of occurrences below this (Default 5)
- loss: Training objective. Allowed values: `ns`, `hs`, `softmax` (Default `ns`)
- sample: Threshold for downsampling higher-frequency words (Default 0.001)
- negative: Number of negative words to sample, for `ns` (Default 5)
- iter: Number of epochs (Default 5)
- sorted_vocab: Sort vocab by descending frequency (Default 1)
- threads: Number of threads to use (Default 12)


In addition, FastText has three additional parameters:

- min_n: min length of char ngrams (Default 3)
- max_n: max length of char ngrams (Default 6)
- bucket: number of buckets used for hashing ngrams (Default 2000000)


Parameters ``min_n`` and ``max_n`` control the lengths of character ngrams that each word is broken down into while training and looking up embeddings. If ``max_n`` is set to 0, or to be lesser than ``min_n``\ , no character ngrams are used, and the model effectively reduces to Word2Vec.



To bound the memory requirements of the model being trained, a hashing function is used that maps ngrams to integers in 1 to K. For hashing these character sequences, the `Fowler-Noll-Vo hashing function <http://www.isthe.com/chongo/tech/comp/fnv>`_ (FNV-1a variant) is employed.


**Note:** As in the case of Word2Vec, you can continue to train your model while using Gensim's native implementation of fastText.


Saving/loading models
---------------------


Models can be saved and loaded via the ``load`` and ``save`` methods.



.. code-block:: default



    # saving a model trained via Gensim's fastText implementation
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(prefix='saved_model_gensim-', delete=False) as tmp:
        model.save(tmp.name)

    loaded_model = FT_gensim.load(tmp.name)
    print(loaded_model)

    os.unlink(tmp.name)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastText object at 0x7f575fa3d630>


The ``save_word2vec_method`` causes the vectors for ngrams to be lost. As a result, a model loaded in this way will behave as a regular word2vec model.


Word vector lookup
------------------


**Note:** Operations like word vector lookups and similarity queries can be performed in exactly the same manner for both the implementations of fastText so they have been demonstrated using only the native fastText implementation here.



FastText models support vector lookups for out-of-vocabulary words by summing up character ngrams belonging to the word.



.. code-block:: default

    print('night' in model.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True



.. code-block:: default

    print('nights' in model.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False



.. code-block:: default

    print(model['night'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 9.51775461e-02,  6.89189089e-03, -5.75633347e-01,  4.84484404e-01,
            5.98646402e-01, -3.22955698e-01, -1.88203231e-01, -3.48487310e-02,
            4.31219041e-01,  3.35425019e-01, -6.36286318e-01, -1.59965567e-02,
           -6.67633653e-01,  4.13802326e-01,  2.80708164e-01, -5.81725016e-02,
           -1.89673513e-01,  1.83778614e-01,  2.39189222e-01, -3.78011256e-01,
           -2.39169523e-01,  2.73841798e-01, -3.78472775e-01,  2.11551264e-02,
           -8.25319648e-01,  7.29921758e-01,  1.21649534e-01,  1.76731527e-01,
            4.07538623e-01,  5.84270060e-03, -6.50230169e-01,  2.28860602e-01,
            8.25605169e-02, -4.79251295e-01,  4.70403671e-01,  1.10500589e-01,
           -1.63527742e-01, -7.36760795e-02,  4.24491972e-01,  2.20531091e-01,
           -1.05110975e-02, -7.41144270e-02,  3.92821461e-01, -6.24631047e-02,
            1.27662361e-01,  1.85713992e-01, -1.31524295e-01,  2.17991263e-01,
           -1.74904689e-02, -3.83495450e-01, -5.69042623e-01, -5.53412855e-01,
            7.40526468e-02,  1.03890281e-02,  4.07311559e-01, -8.03362131e-01,
           -1.08180597e-01, -1.78428411e-01,  9.29761119e-03, -2.72015128e-02,
            2.11776942e-01, -1.24705046e-01, -4.90403205e-01, -9.45862085e-02,
           -5.35887361e-01,  3.53770256e-01, -1.20761819e-04,  1.54261395e-01,
            3.11418567e-02,  4.76123035e-01, -5.65737009e-01, -5.14414787e-01,
           -1.19198393e-02, -1.10536508e-01, -3.42910647e-01,  3.17953080e-02,
            2.77650893e-01,  1.07256003e-01, -1.41108185e-01,  1.81120232e-01,
            4.52760220e-01,  5.86952008e-02, -1.71264693e-01,  4.52266842e-01,
           -3.99318129e-01, -2.59079546e-01,  1.26494303e-01,  2.93788224e-01,
            3.23280275e-01,  2.96236515e-01, -1.72860056e-01,  1.21067993e-01,
           -3.56455632e-02, -1.90687954e-01, -2.67633319e-01,  5.57876825e-01,
            2.82045633e-01,  4.40662771e-01,  5.51508486e-01,  4.76593047e-01],
          dtype=float32)



.. code-block:: default

    print(model['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 8.36547464e-02,  6.55087968e-03, -5.00685155e-01,  4.20664519e-01,
            5.19802272e-01, -2.82329440e-01, -1.64370134e-01, -2.95282304e-02,
            3.74739796e-01,  2.92407662e-01, -5.55526197e-01, -1.53528349e-02,
           -5.80950856e-01,  3.61315876e-01,  2.40151674e-01, -5.03144711e-02,
           -1.67107195e-01,  1.58128008e-01,  2.06411198e-01, -3.29228520e-01,
           -2.08010316e-01,  2.39102423e-01, -3.29903662e-01,  1.90601517e-02,
           -7.18589365e-01,  6.34692907e-01,  1.05780549e-01,  1.54197395e-01,
            3.54656041e-01,  4.67368308e-03, -5.63514590e-01,  2.00302228e-01,
            7.05042407e-02, -4.18988466e-01,  4.08986926e-01,  9.58395377e-02,
           -1.40793711e-01, -6.53786659e-02,  3.69218737e-01,  1.92644417e-01,
           -8.25192221e-03, -6.46147057e-02,  3.41238201e-01, -5.26223630e-02,
            1.09977126e-01,  1.61804438e-01, -1.17992245e-01,  1.91429988e-01,
           -1.58510040e-02, -3.32186550e-01, -4.97555315e-01, -4.80619788e-01,
            6.55840039e-02,  1.00020058e-02,  3.54951859e-01, -6.99018478e-01,
           -9.42101181e-02, -1.54089138e-01,  5.75800799e-03, -2.33614929e-02,
            1.83812991e-01, -1.09346680e-01, -4.26444769e-01, -8.28935578e-02,
           -4.66715872e-01,  3.07202607e-01,  1.00058842e-05,  1.34095371e-01,
            2.88459435e-02,  4.15012509e-01, -4.92204189e-01, -4.47121948e-01,
           -1.04272002e-02, -9.70624313e-02, -2.98134476e-01,  2.68633105e-02,
            2.41655767e-01,  9.35228318e-02, -1.20855518e-01,  1.56785622e-01,
            3.94739091e-01,  5.08996844e-02, -1.49854109e-01,  3.92645687e-01,
           -3.46374542e-01, -2.24778414e-01,  1.12099610e-01,  2.56878823e-01,
            2.80495703e-01,  2.58337140e-01, -1.50668636e-01,  1.05068199e-01,
           -3.23841572e-02, -1.65567130e-01, -2.32554063e-01,  4.85449374e-01,
            2.44345963e-01,  3.81803930e-01,  4.79236662e-01,  4.14590687e-01],
          dtype=float32)


The ``in`` operation works slightly differently from the original word2vec. It tests whether a vector for the given word exists or not, not whether the word is present in the word vocabulary. To test whether a word is present in the training word vocabulary -


Tests if word present in vocab


.. code-block:: default

    print("word" in model.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False


Tests if vector present for word


.. code-block:: default

    print("word" in model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True


Similarity operations
---------------------


Similarity operations work the same way as word2vec. **Out-of-vocabulary words can also be used, provided they have at least one character ngram present in the training data.**



.. code-block:: default



    print("nights" in model.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False



.. code-block:: default

    print("night" in model.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True



.. code-block:: default

    print(model.similarity("night", "nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999928


Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.


Other similarity operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only


.. code-block:: default

    print(model.most_similar("nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('study', 0.9982819557189941),
     ('Arafat', 0.9982815980911255),
     ('Arafat,', 0.9982744455337524),
     ('boat', 0.998274028301239),
     ('"That', 0.9982696175575256),
     ('often', 0.9982571601867676),
     ('north.', 0.9982531666755676),
     ("Arafat's", 0.9982520341873169),
     ('Endeavour', 0.9982511401176453),
     ('heard', 0.998244047164917)]



.. code-block:: default

    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999509



.. code-block:: default

    print(model.doesnt_match("breakfast cereal dinner lunch".split()))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    'breakfast'



.. code-block:: default

    print(model.most_similar(positive=['baghdad', 'england'], negative=['london']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('1', 0.24240879714488983),
     ('40', 0.23784878849983215),
     ('2', 0.23489072918891907),
     ('26', 0.23309852182865143),
     ('20', 0.2329929769039154),
     ('UN', 0.2326236069202423),
     ('blaze', 0.23229902982711792),
     ('keep', 0.23141512274742126),
     ('...', 0.2313539981842041),
     ('As', 0.23093350231647491)]



.. code-block:: default

    print(model.accuracy(questions=datapath('questions-words.txt')))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [{'correct': [], 'incorrect': [], 'section': 'capital-common-countries'},
     {'correct': [], 'incorrect': [], 'section': 'capital-world'},
     {'correct': [], 'incorrect': [], 'section': 'currency'},
     {'correct': [], 'incorrect': [], 'section': 'city-in-state'},
     {'correct': [],
      'incorrect': [('HE', 'SHE', 'HIS', 'HER'), ('HIS', 'HER', 'HE', 'SHE')],
      'section': 'family'},
     {'correct': [], 'incorrect': [], 'section': 'gram1-adjective-to-adverb'},
     {'correct': [], 'incorrect': [], 'section': 'gram2-opposite'},
     {'correct': [('GOOD', 'BETTER', 'GREAT', 'GREATER'),
                  ('LONG', 'LONGER', 'GREAT', 'GREATER')],
      'incorrect': [('GOOD', 'BETTER', 'LONG', 'LONGER'),
                    ('GOOD', 'BETTER', 'LOW', 'LOWER'),
                    ('GREAT', 'GREATER', 'LONG', 'LONGER'),
                    ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                    ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                    ('LONG', 'LONGER', 'LOW', 'LOWER'),
                    ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                    ('LOW', 'LOWER', 'LONG', 'LONGER')],
      'section': 'gram3-comparative'},
     {'correct': [('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                  ('GREAT', 'GREATEST', 'LARGE', 'LARGEST')],
      'incorrect': [('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                    ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                    ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                    ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'GREAT', 'GREATEST')],
      'section': 'gram4-superlative'},
     {'correct': [('GO', 'GOING', 'PLAY', 'PLAYING'),
                  ('PLAY', 'PLAYING', 'SAY', 'SAYING'),
                  ('PLAY', 'PLAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'PLAY', 'PLAYING')],
      'incorrect': [('GO', 'GOING', 'LOOK', 'LOOKING'),
                    ('GO', 'GOING', 'RUN', 'RUNNING'),
                    ('GO', 'GOING', 'SAY', 'SAYING'),
                    ('LOOK', 'LOOKING', 'PLAY', 'PLAYING'),
                    ('LOOK', 'LOOKING', 'RUN', 'RUNNING'),
                    ('LOOK', 'LOOKING', 'SAY', 'SAYING'),
                    ('LOOK', 'LOOKING', 'GO', 'GOING'),
                    ('PLAY', 'PLAYING', 'RUN', 'RUNNING'),
                    ('PLAY', 'PLAYING', 'GO', 'GOING'),
                    ('RUN', 'RUNNING', 'SAY', 'SAYING'),
                    ('RUN', 'RUNNING', 'GO', 'GOING'),
                    ('RUN', 'RUNNING', 'LOOK', 'LOOKING'),
                    ('RUN', 'RUNNING', 'PLAY', 'PLAYING'),
                    ('SAY', 'SAYING', 'GO', 'GOING'),
                    ('SAY', 'SAYING', 'RUN', 'RUNNING')],
      'section': 'gram5-present-participle'},
     {'correct': [('AUSTRALIA', 'AUSTRALIAN', 'INDIA', 'INDIAN'),
                  ('AUSTRALIA', 'AUSTRALIAN', 'ISRAEL', 'ISRAELI'),
                  ('FRANCE', 'FRENCH', 'INDIA', 'INDIAN'),
                  ('FRANCE', 'FRENCH', 'ISRAEL', 'ISRAELI'),
                  ('INDIA', 'INDIAN', 'ISRAEL', 'ISRAELI'),
                  ('INDIA', 'INDIAN', 'AUSTRALIA', 'AUSTRALIAN'),
                  ('ISRAEL', 'ISRAELI', 'INDIA', 'INDIAN'),
                  ('SWITZERLAND', 'SWISS', 'INDIA', 'INDIAN')],
      'incorrect': [('AUSTRALIA', 'AUSTRALIAN', 'FRANCE', 'FRENCH'),
                    ('AUSTRALIA', 'AUSTRALIAN', 'SWITZERLAND', 'SWISS'),
                    ('FRANCE', 'FRENCH', 'SWITZERLAND', 'SWISS'),
                    ('FRANCE', 'FRENCH', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('INDIA', 'INDIAN', 'SWITZERLAND', 'SWISS'),
                    ('INDIA', 'INDIAN', 'FRANCE', 'FRENCH'),
                    ('ISRAEL', 'ISRAELI', 'SWITZERLAND', 'SWISS'),
                    ('ISRAEL', 'ISRAELI', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('ISRAEL', 'ISRAELI', 'FRANCE', 'FRENCH'),
                    ('SWITZERLAND', 'SWISS', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('SWITZERLAND', 'SWISS', 'FRANCE', 'FRENCH'),
                    ('SWITZERLAND', 'SWISS', 'ISRAEL', 'ISRAELI')],
      'section': 'gram6-nationality-adjective'},
     {'correct': [('PAYING', 'PAID', 'SAYING', 'SAID')],
      'incorrect': [('GOING', 'WENT', 'PAYING', 'PAID'),
                    ('GOING', 'WENT', 'PLAYING', 'PLAYED'),
                    ('GOING', 'WENT', 'SAYING', 'SAID'),
                    ('GOING', 'WENT', 'TAKING', 'TOOK'),
                    ('PAYING', 'PAID', 'PLAYING', 'PLAYED'),
                    ('PAYING', 'PAID', 'TAKING', 'TOOK'),
                    ('PAYING', 'PAID', 'GOING', 'WENT'),
                    ('PLAYING', 'PLAYED', 'SAYING', 'SAID'),
                    ('PLAYING', 'PLAYED', 'TAKING', 'TOOK'),
                    ('PLAYING', 'PLAYED', 'GOING', 'WENT'),
                    ('PLAYING', 'PLAYED', 'PAYING', 'PAID'),
                    ('SAYING', 'SAID', 'TAKING', 'TOOK'),
                    ('SAYING', 'SAID', 'GOING', 'WENT'),
                    ('SAYING', 'SAID', 'PAYING', 'PAID'),
                    ('SAYING', 'SAID', 'PLAYING', 'PLAYED'),
                    ('TAKING', 'TOOK', 'GOING', 'WENT'),
                    ('TAKING', 'TOOK', 'PAYING', 'PAID'),
                    ('TAKING', 'TOOK', 'PLAYING', 'PLAYED'),
                    ('TAKING', 'TOOK', 'SAYING', 'SAID')],
      'section': 'gram7-past-tense'},
     {'correct': [('MAN', 'MEN', 'CHILD', 'CHILDREN')],
      'incorrect': [('BUILDING', 'BUILDINGS', 'CAR', 'CARS'),
                    ('BUILDING', 'BUILDINGS', 'CHILD', 'CHILDREN'),
                    ('BUILDING', 'BUILDINGS', 'MAN', 'MEN'),
                    ('CAR', 'CARS', 'CHILD', 'CHILDREN'),
                    ('CAR', 'CARS', 'MAN', 'MEN'),
                    ('CAR', 'CARS', 'BUILDING', 'BUILDINGS'),
                    ('CHILD', 'CHILDREN', 'MAN', 'MEN'),
                    ('CHILD', 'CHILDREN', 'BUILDING', 'BUILDINGS'),
                    ('CHILD', 'CHILDREN', 'CAR', 'CARS'),
                    ('MAN', 'MEN', 'BUILDING', 'BUILDINGS'),
                    ('MAN', 'MEN', 'CAR', 'CARS')],
      'section': 'gram8-plural'},
     {'correct': [], 'incorrect': [], 'section': 'gram9-plural-verbs'},
     {'correct': [('GOOD', 'BETTER', 'GREAT', 'GREATER'),
                  ('LONG', 'LONGER', 'GREAT', 'GREATER'),
                  ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                  ('GREAT', 'GREATEST', 'LARGE', 'LARGEST'),
                  ('GO', 'GOING', 'PLAY', 'PLAYING'),
                  ('PLAY', 'PLAYING', 'SAY', 'SAYING'),
                  ('PLAY', 'PLAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'PLAY', 'PLAYING'),
                  ('AUSTRALIA', 'AUSTRALIAN', 'INDIA', 'INDIAN'),
                  ('AUSTRALIA', 'AUSTRALIAN', 'ISRAEL', 'ISRAELI'),
                  ('FRANCE', 'FRENCH', 'INDIA', 'INDIAN'),
                  ('FRANCE', 'FRENCH', 'ISRAEL', 'ISRAELI'),
                  ('INDIA', 'INDIAN', 'ISRAEL', 'ISRAELI'),
                  ('INDIA', 'INDIAN', 'AUSTRALIA', 'AUSTRALIAN'),
                  ('ISRAEL', 'ISRAELI', 'INDIA', 'INDIAN'),
                  ('SWITZERLAND', 'SWISS', 'INDIA', 'INDIAN'),
                  ('PAYING', 'PAID', 'SAYING', 'SAID'),
                  ('MAN', 'MEN', 'CHILD', 'CHILDREN')],
      'incorrect': [('HE', 'SHE', 'HIS', 'HER'),
                    ('HIS', 'HER', 'HE', 'SHE'),
                    ('GOOD', 'BETTER', 'LONG', 'LONGER'),
                    ('GOOD', 'BETTER', 'LOW', 'LOWER'),
                    ('GREAT', 'GREATER', 'LONG', 'LONGER'),
                    ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                    ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                    ('LONG', 'LONGER', 'LOW', 'LOWER'),
                    ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                    ('LOW', 'LOWER', 'LONG', 'LONGER'),
                    ('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                    ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                    ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                    ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'GREAT', 'GREATEST'),
                    ('GO', 'GOING', 'LOOK', 'LOOKING'),
                    ('GO', 'GOING', 'RUN', 'RUNNING'),
                    ('GO', 'GOING', 'SAY', 'SAYING'),
                    ('LOOK', 'LOOKING', 'PLAY', 'PLAYING'),
                    ('LOOK', 'LOOKING', 'RUN', 'RUNNING'),
                    ('LOOK', 'LOOKING', 'SAY', 'SAYING'),
                    ('LOOK', 'LOOKING', 'GO', 'GOING'),
                    ('PLAY', 'PLAYING', 'RUN', 'RUNNING'),
                    ('PLAY', 'PLAYING', 'GO', 'GOING'),
                    ('RUN', 'RUNNING', 'SAY', 'SAYING'),
                    ('RUN', 'RUNNING', 'GO', 'GOING'),
                    ('RUN', 'RUNNING', 'LOOK', 'LOOKING'),
                    ('RUN', 'RUNNING', 'PLAY', 'PLAYING'),
                    ('SAY', 'SAYING', 'GO', 'GOING'),
                    ('SAY', 'SAYING', 'RUN', 'RUNNING'),
                    ('AUSTRALIA', 'AUSTRALIAN', 'FRANCE', 'FRENCH'),
                    ('AUSTRALIA', 'AUSTRALIAN', 'SWITZERLAND', 'SWISS'),
                    ('FRANCE', 'FRENCH', 'SWITZERLAND', 'SWISS'),
                    ('FRANCE', 'FRENCH', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('INDIA', 'INDIAN', 'SWITZERLAND', 'SWISS'),
                    ('INDIA', 'INDIAN', 'FRANCE', 'FRENCH'),
                    ('ISRAEL', 'ISRAELI', 'SWITZERLAND', 'SWISS'),
                    ('ISRAEL', 'ISRAELI', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('ISRAEL', 'ISRAELI', 'FRANCE', 'FRENCH'),
                    ('SWITZERLAND', 'SWISS', 'AUSTRALIA', 'AUSTRALIAN'),
                    ('SWITZERLAND', 'SWISS', 'FRANCE', 'FRENCH'),
                    ('SWITZERLAND', 'SWISS', 'ISRAEL', 'ISRAELI'),
                    ('GOING', 'WENT', 'PAYING', 'PAID'),
                    ('GOING', 'WENT', 'PLAYING', 'PLAYED'),
                    ('GOING', 'WENT', 'SAYING', 'SAID'),
                    ('GOING', 'WENT', 'TAKING', 'TOOK'),
                    ('PAYING', 'PAID', 'PLAYING', 'PLAYED'),
                    ('PAYING', 'PAID', 'TAKING', 'TOOK'),
                    ('PAYING', 'PAID', 'GOING', 'WENT'),
                    ('PLAYING', 'PLAYED', 'SAYING', 'SAID'),
                    ('PLAYING', 'PLAYED', 'TAKING', 'TOOK'),
                    ('PLAYING', 'PLAYED', 'GOING', 'WENT'),
                    ('PLAYING', 'PLAYED', 'PAYING', 'PAID'),
                    ('SAYING', 'SAID', 'TAKING', 'TOOK'),
                    ('SAYING', 'SAID', 'GOING', 'WENT'),
                    ('SAYING', 'SAID', 'PAYING', 'PAID'),
                    ('SAYING', 'SAID', 'PLAYING', 'PLAYED'),
                    ('TAKING', 'TOOK', 'GOING', 'WENT'),
                    ('TAKING', 'TOOK', 'PAYING', 'PAID'),
                    ('TAKING', 'TOOK', 'PLAYING', 'PLAYED'),
                    ('TAKING', 'TOOK', 'SAYING', 'SAID'),
                    ('BUILDING', 'BUILDINGS', 'CAR', 'CARS'),
                    ('BUILDING', 'BUILDINGS', 'CHILD', 'CHILDREN'),
                    ('BUILDING', 'BUILDINGS', 'MAN', 'MEN'),
                    ('CAR', 'CARS', 'CHILD', 'CHILDREN'),
                    ('CAR', 'CARS', 'MAN', 'MEN'),
                    ('CAR', 'CARS', 'BUILDING', 'BUILDINGS'),
                    ('CHILD', 'CHILDREN', 'MAN', 'MEN'),
                    ('CHILD', 'CHILDREN', 'BUILDING', 'BUILDINGS'),
                    ('CHILD', 'CHILDREN', 'CAR', 'CARS'),
                    ('MAN', 'MEN', 'BUILDING', 'BUILDINGS'),
                    ('MAN', 'MEN', 'CAR', 'CARS')],
      'section': 'total'}]


Word Movers distance
^^^^^^^^^^^^^^^^^^^^

Let's start with two sentences:


.. code-block:: default

    sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    sentence_president = 'The president greets the press in Chicago'.lower().split()








Remove their stopwords.



.. code-block:: default

    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    sentence_obama = [w for w in sentence_obama if w not in stopwords]
    sentence_president = [w for w in sentence_president if w not in stopwords]







Compute WMD.


.. code-block:: default

    distance = model.wmdistance(sentence_obama, sentence_president)
    print(distance)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.3895121413147449


That's all! You've made it to the end of this tutorial.

Here we show a pretty fastText logo so that our gallery picks it up as a thumbnail.



.. code-block:: default

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('fasttext-logo-color-web.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/010_tutorials/images/sphx_glr_run_030_fasttext_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  22.652 seconds)

**Estimated memory usage:**  4554 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_030_fasttext.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_030_fasttext.py <run_030_fasttext.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_030_fasttext.ipynb <run_030_fasttext.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
