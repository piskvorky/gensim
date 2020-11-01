.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_fasttext.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_tutorials_run_fasttext.py:


FastText Model
==============

Introduces Gensim's fastText model and demonstrates its use on the Lee Corpus.


.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








Here, we'll learn to work with fastText library for training word-embedding
models, saving & loading them and performing similarity operations & vector
lookups analogous to Word2Vec.

When to use fastText?
---------------------

The main principle behind `fastText <https://github.com/facebookresearch/fastText>`_ is that the
morphological structure of a word carries important information about the meaning of the word.
Such structure is not taken into account by traditional word embeddings like Word2Vec, which
train a unique word embedding for every individual word.
This is especially significant for morphologically rich languages (German, Turkish) in which a
single word can have a large number of morphological forms, each of which might occur rarely,
thus making it hard to train good word embeddings.


fastText attempts to solve this by treating each word as the aggregation of its subwords.
For the sake of simplicity and language-independence, subwords are taken to be the character ngrams
of the word. The vector for a word is simply taken to be the sum of all vectors of its component char-ngrams.


According to a detailed comparison of Word2Vec and fastText in
`this notebook <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Word2Vec_FastText_Comparison.ipynb>`__,
fastText does significantly better on syntactic tasks as compared to the original Word2Vec,
especially when the size of the training corpus is small. Word2Vec slightly outperforms fastText
on semantic tasks though. The differences grow smaller as the size of the training corpus increases.


fastText can obtain vectors even for out-of-vocabulary (OOV) words, by summing up vectors for its
component char-ngrams, provided at least one of the char-ngrams was present in the training data.


Training models
---------------


For the following examples, we'll use the Lee Corpus (which you already have if you've installed Gensim) for training our model.





.. code-block:: default

    from pprint import pprint as print
    from gensim.models.fasttext import FastText
    from gensim.test.utils import datapath

    # Set file names for train and test data
    corpus_file = datapath('lee_background.cor')

    model = FastText(vector_size=100)

    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)

    # train the model
    model.train(
        corpus_file=corpus_file, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words,
    )

    print(model)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastText object at 0x20ce0d390>




Training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^


Hyperparameters for training the model follow the same pattern as Word2Vec. FastText supports the following parameters from the original word2vec:

- model: Training architecture. Allowed values: `cbow`, `skipgram` (Default `cbow`)
- vector_size: Dimensionality of vector embeddings to be learnt (Default 100)
- alpha: Initial learning rate (Default 0.025)
- window: Context window size (Default 5)
- min_count: Ignore words with number of occurrences below this (Default 5)
- loss: Training objective. Allowed values: `ns`, `hs`, `softmax` (Default `ns`)
- sample: Threshold for downsampling higher-frequency words (Default 0.001)
- negative: Number of negative words to sample, for `ns` (Default 5)
- epochs: Number of epochs (Default 5)
- sorted_vocab: Sort vocab by descending frequency (Default 1)
- threads: Number of threads to use (Default 12)


In addition, fastText has three additional parameters:

- min_n: min length of char ngrams (Default 3)
- max_n: max length of char ngrams (Default 6)
- bucket: number of buckets used for hashing ngrams (Default 2000000)


Parameters ``min_n`` and ``max_n`` control the lengths of character ngrams that each word is broken down into while training and looking up embeddings. If ``max_n`` is set to 0, or to be lesser than ``min_n``\ , no character ngrams are used, and the model effectively reduces to Word2Vec.



To bound the memory requirements of the model being trained, a hashing function is used that maps ngrams to integers in 1 to K. For hashing these character sequences, the `Fowler-Noll-Vo hashing function <http://www.isthe.com/chongo/tech/comp/fnv>`_ (FNV-1a variant) is employed.


**Note:** You can continue to train your model while using Gensim's native implementation of fastText.


Saving/loading models
---------------------


Models can be saved and loaded via the ``load`` and ``save`` methods, just like
any other model in Gensim.



.. code-block:: default



    # Save a model trained via Gensim's fastText implementation to temp.
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(prefix='saved_model_gensim-', delete=False) as tmp:
        model.save(tmp.name, separately=[])

    # Load back the same model.
    loaded_model = FastText.load(tmp.name)
    print(loaded_model)

    os.unlink(tmp.name)  # demonstration complete, don't need the temp file anymore





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastText object at 0x20cc99d30>




The ``save_word2vec_format`` is also available for fastText models, but will
cause all vectors for ngrams to be lost.
As a result, a model loaded in this way will behave as a regular word2vec model.


Word vector lookup
------------------


All information necessary for looking up fastText words (incl. OOV words) is
contained in its ``model.wv`` attribute.

If you don't need to continue training your model, you can export & save this `.wv`
attribute and discard `model`, to save space and RAM.



.. code-block:: default

    wv = model.wv
    print(wv)

    #
    # FastText models support vector lookups for out-of-vocabulary words by summing up character ngrams belonging to the word.
    #
    print('night' in wv.key_to_index)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastTextKeyedVectors object at 0x20ce0d828>
    True





.. code-block:: default

    print('nights' in wv.key_to_index)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False





.. code-block:: default

    print(wv['night'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 0.12453239, -0.26018462, -0.04087191,  0.2563215 ,  0.31401935,
            0.16155584,  0.39527607,  0.27404118, -0.45236284,  0.06942682,
            0.36584955,  0.51162827, -0.51161295, -0.192019  , -0.5068029 ,
           -0.07426998, -0.6276584 ,  0.22271585,  0.19990133,  0.2582401 ,
            0.14329399, -0.01959469, -0.45576197, -0.06447829,  0.1493489 ,
            0.17261286, -0.13472046,  0.26546794, -0.34596932,  0.5626187 ,
           -0.7038802 ,  0.15603925, -0.03104019, -0.06228801, -0.13480644,
           -0.0684596 ,  0.24728075,  0.55081636,  0.07330963,  0.32814154,
            0.1574982 ,  0.56742406, -0.31233737,  0.14195296,  0.0540203 ,
            0.01718009,  0.05519052, -0.04002226,  0.16157456, -0.5134223 ,
           -0.01033936,  0.05745083, -0.39208183,  0.52553374, -1.0542839 ,
            0.2145304 , -0.15234643, -0.35197273, -0.6215585 ,  0.01796502,
            0.21242104,  0.30762967,  0.2787644 , -0.19908747,  0.7144409 ,
            0.45586124, -0.21344525,  0.26920903, -0.651759  , -0.37096855,
           -0.16243419, -0.3085725 , -0.70485127, -0.04926324, -0.80278563,
           -0.24352737,  0.6427129 , -0.3530421 , -0.29960123,  0.01466726,
           -0.18253349, -0.2489397 ,  0.00648343,  0.18057272, -0.11812428,
           -0.49044088,  0.1847386 , -0.27946883,  0.3941279 , -0.39211616,
            0.26847798,  0.41468227, -0.3953728 , -0.25371104,  0.3390468 ,
           -0.16447693, -0.18722224,  0.2782088 , -0.0696249 ,  0.4313547 ],
          dtype=float32)





.. code-block:: default

    print(wv['nights'])






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 0.10586783, -0.22489995, -0.03636307,  0.22263278,  0.27037606,
            0.1394871 ,  0.3411114 ,  0.2369042 , -0.38989475,  0.05935   ,
            0.31713557,  0.44301754, -0.44249156, -0.16652377, -0.4388366 ,
           -0.06266895, -0.5436303 ,  0.19294666,  0.17363031,  0.22459263,
            0.12532061, -0.01866964, -0.3936521 , -0.05507145,  0.12905194,
            0.14942174, -0.11657442,  0.22935589, -0.29934618,  0.4859668 ,
           -0.6073519 ,  0.13433163, -0.02491274, -0.05468523, -0.11884545,
           -0.06117092,  0.21444008,  0.4775469 ,  0.06227469,  0.28350767,
            0.13580805,  0.48993143, -0.27067345,  0.1252003 ,  0.04606731,
            0.01598426,  0.04640368, -0.03456376,  0.14138013, -0.44429192,
           -0.00865329,  0.05027836, -0.341311  ,  0.45402458, -0.91097856,
            0.1868968 , -0.13116683, -0.30361563, -0.5364188 ,  0.01603454,
            0.18146741,  0.26708448,  0.24074472, -0.17163375,  0.61906886,
            0.39530373, -0.18259627,  0.23319626, -0.5634787 , -0.31959867,
           -0.13945322, -0.269441  , -0.60941464, -0.0403638 , -0.69563633,
           -0.2098089 ,  0.5569868 , -0.30320194, -0.25840232,  0.01436759,
           -0.15632603, -0.21624804,  0.00434287,  0.15566474, -0.10228094,
           -0.4249678 ,  0.16197811, -0.24147548,  0.34205705, -0.3391568 ,
            0.23235887,  0.35860622, -0.34247142, -0.21777524,  0.29318404,
           -0.1407287 , -0.16115218,  0.24247572, -0.06217333,  0.37221798],
          dtype=float32)




Similarity operations
---------------------


Similarity operations work the same way as word2vec. **Out-of-vocabulary words can also be used, provided they have at least one character ngram present in the training data.**



.. code-block:: default



    print("nights" in wv.key_to_index)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False





.. code-block:: default

    print("night" in wv.key_to_index)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True





.. code-block:: default

    print(wv.similarity("night", "nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999929




Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.


Other similarity operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only


.. code-block:: default

    print(wv.most_similar("nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('night', 0.9999929070472717),
     ('night.', 0.9999895095825195),
     ('flights', 0.999988853931427),
     ('rights', 0.9999886751174927),
     ('residents', 0.9999884366989136),
     ('overnight', 0.9999883770942688),
     ('commanders', 0.999988317489624),
     ('reached', 0.9999881386756897),
     ('commander', 0.9999880790710449),
     ('leading', 0.999987781047821)]





.. code-block:: default

    print(wv.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999402





.. code-block:: default

    print(wv.doesnt_match("breakfast cereal dinner lunch".split()))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    'lunch'





.. code-block:: default

    print(wv.most_similar(positive=['baghdad', 'england'], negative=['london']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('attempt', 0.999660074710846),
     ('biggest', 0.9996545314788818),
     ('again', 0.9996527433395386),
     ('against', 0.9996523857116699),
     ('doubles', 0.9996522068977356),
     ('Royal', 0.9996512532234192),
     ('Airlines', 0.9996494054794312),
     ('forced', 0.9996494054794312),
     ('arrest', 0.9996492266654968),
     ('follows', 0.999649167060852)]





.. code-block:: default

    print(wv.evaluate_word_analogies(datapath('questions-words.txt')))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.24489795918367346,
     [{'correct': [], 'incorrect': [], 'section': 'capital-common-countries'},
      {'correct': [], 'incorrect': [], 'section': 'capital-world'},
      {'correct': [], 'incorrect': [], 'section': 'currency'},
      {'correct': [], 'incorrect': [], 'section': 'city-in-state'},
      {'correct': [],
       'incorrect': [('HE', 'SHE', 'HIS', 'HER'), ('HIS', 'HER', 'HE', 'SHE')],
       'section': 'family'},
      {'correct': [], 'incorrect': [], 'section': 'gram1-adjective-to-adverb'},
      {'correct': [], 'incorrect': [], 'section': 'gram2-opposite'},
      {'correct': [('GOOD', 'BETTER', 'LOW', 'LOWER'),
                   ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                   ('LONG', 'LONGER', 'LOW', 'LOWER')],
       'incorrect': [('GOOD', 'BETTER', 'GREAT', 'GREATER'),
                     ('GOOD', 'BETTER', 'LONG', 'LONGER'),
                     ('GREAT', 'GREATER', 'LONG', 'LONGER'),
                     ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                     ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                     ('LONG', 'LONGER', 'GREAT', 'GREATER'),
                     ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                     ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                     ('LOW', 'LOWER', 'LONG', 'LONGER')],
       'section': 'gram3-comparative'},
      {'correct': [('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                   ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                   ('GREAT', 'GREATEST', 'LARGE', 'LARGEST')],
       'incorrect': [('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                     ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                     ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                     ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                     ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                     ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                     ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                     ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                     ('LARGE', 'LARGEST', 'GREAT', 'GREATEST')],
       'section': 'gram4-superlative'},
      {'correct': [('GO', 'GOING', 'SAY', 'SAYING'),
                   ('LOOK', 'LOOKING', 'PLAY', 'PLAYING'),
                   ('LOOK', 'LOOKING', 'SAY', 'SAYING'),
                   ('LOOK', 'LOOKING', 'GO', 'GOING'),
                   ('PLAY', 'PLAYING', 'SAY', 'SAYING'),
                   ('PLAY', 'PLAYING', 'GO', 'GOING'),
                   ('SAY', 'SAYING', 'GO', 'GOING')],
       'incorrect': [('GO', 'GOING', 'LOOK', 'LOOKING'),
                     ('GO', 'GOING', 'PLAY', 'PLAYING'),
                     ('GO', 'GOING', 'RUN', 'RUNNING'),
                     ('LOOK', 'LOOKING', 'RUN', 'RUNNING'),
                     ('PLAY', 'PLAYING', 'RUN', 'RUNNING'),
                     ('PLAY', 'PLAYING', 'LOOK', 'LOOKING'),
                     ('RUN', 'RUNNING', 'SAY', 'SAYING'),
                     ('RUN', 'RUNNING', 'GO', 'GOING'),
                     ('RUN', 'RUNNING', 'LOOK', 'LOOKING'),
                     ('RUN', 'RUNNING', 'PLAY', 'PLAYING'),
                     ('SAY', 'SAYING', 'LOOK', 'LOOKING'),
                     ('SAY', 'SAYING', 'PLAY', 'PLAYING'),
                     ('SAY', 'SAYING', 'RUN', 'RUNNING')],
       'section': 'gram5-present-participle'},
      {'correct': [('AUSTRALIA', 'AUSTRALIAN', 'INDIA', 'INDIAN'),
                   ('AUSTRALIA', 'AUSTRALIAN', 'ISRAEL', 'ISRAELI'),
                   ('FRANCE', 'FRENCH', 'INDIA', 'INDIAN'),
                   ('INDIA', 'INDIAN', 'ISRAEL', 'ISRAELI'),
                   ('ISRAEL', 'ISRAELI', 'INDIA', 'INDIAN'),
                   ('SWITZERLAND', 'SWISS', 'INDIA', 'INDIAN')],
       'incorrect': [('AUSTRALIA', 'AUSTRALIAN', 'FRANCE', 'FRENCH'),
                     ('AUSTRALIA', 'AUSTRALIAN', 'SWITZERLAND', 'SWISS'),
                     ('FRANCE', 'FRENCH', 'ISRAEL', 'ISRAELI'),
                     ('FRANCE', 'FRENCH', 'SWITZERLAND', 'SWISS'),
                     ('FRANCE', 'FRENCH', 'AUSTRALIA', 'AUSTRALIAN'),
                     ('INDIA', 'INDIAN', 'SWITZERLAND', 'SWISS'),
                     ('INDIA', 'INDIAN', 'AUSTRALIA', 'AUSTRALIAN'),
                     ('INDIA', 'INDIAN', 'FRANCE', 'FRENCH'),
                     ('ISRAEL', 'ISRAELI', 'SWITZERLAND', 'SWISS'),
                     ('ISRAEL', 'ISRAELI', 'AUSTRALIA', 'AUSTRALIAN'),
                     ('ISRAEL', 'ISRAELI', 'FRANCE', 'FRENCH'),
                     ('SWITZERLAND', 'SWISS', 'AUSTRALIA', 'AUSTRALIAN'),
                     ('SWITZERLAND', 'SWISS', 'FRANCE', 'FRENCH'),
                     ('SWITZERLAND', 'SWISS', 'ISRAEL', 'ISRAELI')],
       'section': 'gram6-nationality-adjective'},
      {'correct': [],
       'incorrect': [('GOING', 'WENT', 'PAYING', 'PAID'),
                     ('GOING', 'WENT', 'PLAYING', 'PLAYED'),
                     ('GOING', 'WENT', 'SAYING', 'SAID'),
                     ('GOING', 'WENT', 'TAKING', 'TOOK'),
                     ('PAYING', 'PAID', 'PLAYING', 'PLAYED'),
                     ('PAYING', 'PAID', 'SAYING', 'SAID'),
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
      {'correct': [('BUILDING', 'BUILDINGS', 'CAR', 'CARS'),
                   ('BUILDING', 'BUILDINGS', 'CHILD', 'CHILDREN'),
                   ('CAR', 'CARS', 'BUILDING', 'BUILDINGS'),
                   ('CHILD', 'CHILDREN', 'CAR', 'CARS'),
                   ('MAN', 'MEN', 'CAR', 'CARS')],
       'incorrect': [('BUILDING', 'BUILDINGS', 'MAN', 'MEN'),
                     ('CAR', 'CARS', 'CHILD', 'CHILDREN'),
                     ('CAR', 'CARS', 'MAN', 'MEN'),
                     ('CHILD', 'CHILDREN', 'MAN', 'MEN'),
                     ('CHILD', 'CHILDREN', 'BUILDING', 'BUILDINGS'),
                     ('MAN', 'MEN', 'BUILDING', 'BUILDINGS'),
                     ('MAN', 'MEN', 'CHILD', 'CHILDREN')],
       'section': 'gram8-plural'},
      {'correct': [], 'incorrect': [], 'section': 'gram9-plural-verbs'},
      {'correct': [('GOOD', 'BETTER', 'LOW', 'LOWER'),
                   ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                   ('LONG', 'LONGER', 'LOW', 'LOWER'),
                   ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                   ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                   ('GREAT', 'GREATEST', 'LARGE', 'LARGEST'),
                   ('GO', 'GOING', 'SAY', 'SAYING'),
                   ('LOOK', 'LOOKING', 'PLAY', 'PLAYING'),
                   ('LOOK', 'LOOKING', 'SAY', 'SAYING'),
                   ('LOOK', 'LOOKING', 'GO', 'GOING'),
                   ('PLAY', 'PLAYING', 'SAY', 'SAYING'),
                   ('PLAY', 'PLAYING', 'GO', 'GOING'),
                   ('SAY', 'SAYING', 'GO', 'GOING'),
                   ('AUSTRALIA', 'AUSTRALIAN', 'INDIA', 'INDIAN'),
                   ('AUSTRALIA', 'AUSTRALIAN', 'ISRAEL', 'ISRAELI'),
                   ('FRANCE', 'FRENCH', 'INDIA', 'INDIAN'),
                   ('INDIA', 'INDIAN', 'ISRAEL', 'ISRAELI'),
                   ('ISRAEL', 'ISRAELI', 'INDIA', 'INDIAN'),
                   ('SWITZERLAND', 'SWISS', 'INDIA', 'INDIAN'),
                   ('BUILDING', 'BUILDINGS', 'CAR', 'CARS'),
                   ('BUILDING', 'BUILDINGS', 'CHILD', 'CHILDREN'),
                   ('CAR', 'CARS', 'BUILDING', 'BUILDINGS'),
                   ('CHILD', 'CHILDREN', 'CAR', 'CARS'),
                   ('MAN', 'MEN', 'CAR', 'CARS')],
       'incorrect': [('HE', 'SHE', 'HIS', 'HER'),
                     ('HIS', 'HER', 'HE', 'SHE'),
                     ('GOOD', 'BETTER', 'GREAT', 'GREATER'),
                     ('GOOD', 'BETTER', 'LONG', 'LONGER'),
                     ('GREAT', 'GREATER', 'LONG', 'LONGER'),
                     ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                     ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                     ('LONG', 'LONGER', 'GREAT', 'GREATER'),
                     ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                     ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                     ('LOW', 'LOWER', 'LONG', 'LONGER'),
                     ('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                     ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                     ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                     ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                     ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                     ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                     ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                     ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                     ('LARGE', 'LARGEST', 'GREAT', 'GREATEST'),
                     ('GO', 'GOING', 'LOOK', 'LOOKING'),
                     ('GO', 'GOING', 'PLAY', 'PLAYING'),
                     ('GO', 'GOING', 'RUN', 'RUNNING'),
                     ('LOOK', 'LOOKING', 'RUN', 'RUNNING'),
                     ('PLAY', 'PLAYING', 'RUN', 'RUNNING'),
                     ('PLAY', 'PLAYING', 'LOOK', 'LOOKING'),
                     ('RUN', 'RUNNING', 'SAY', 'SAYING'),
                     ('RUN', 'RUNNING', 'GO', 'GOING'),
                     ('RUN', 'RUNNING', 'LOOK', 'LOOKING'),
                     ('RUN', 'RUNNING', 'PLAY', 'PLAYING'),
                     ('SAY', 'SAYING', 'LOOK', 'LOOKING'),
                     ('SAY', 'SAYING', 'PLAY', 'PLAYING'),
                     ('SAY', 'SAYING', 'RUN', 'RUNNING'),
                     ('AUSTRALIA', 'AUSTRALIAN', 'FRANCE', 'FRENCH'),
                     ('AUSTRALIA', 'AUSTRALIAN', 'SWITZERLAND', 'SWISS'),
                     ('FRANCE', 'FRENCH', 'ISRAEL', 'ISRAELI'),
                     ('FRANCE', 'FRENCH', 'SWITZERLAND', 'SWISS'),
                     ('FRANCE', 'FRENCH', 'AUSTRALIA', 'AUSTRALIAN'),
                     ('INDIA', 'INDIAN', 'SWITZERLAND', 'SWISS'),
                     ('INDIA', 'INDIAN', 'AUSTRALIA', 'AUSTRALIAN'),
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
                     ('PAYING', 'PAID', 'SAYING', 'SAID'),
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
                     ('BUILDING', 'BUILDINGS', 'MAN', 'MEN'),
                     ('CAR', 'CARS', 'CHILD', 'CHILDREN'),
                     ('CAR', 'CARS', 'MAN', 'MEN'),
                     ('CHILD', 'CHILDREN', 'MAN', 'MEN'),
                     ('CHILD', 'CHILDREN', 'BUILDING', 'BUILDINGS'),
                     ('MAN', 'MEN', 'BUILDING', 'BUILDINGS'),
                     ('MAN', 'MEN', 'CHILD', 'CHILDREN')],
       'section': 'Total accuracy'}])




Word Movers distance
^^^^^^^^^^^^^^^^^^^^

You'll need the optional ``pyemd`` library for this section, ``pip install pyemd``.

Let's start with two sentences:


.. code-block:: default

    sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    sentence_president = 'The president greets the press in Chicago'.lower().split()









Remove their stopwords.



.. code-block:: default

    from gensim.parsing.preprocessing import STOPWORDS
    sentence_obama = [w for w in sentence_obama if w not in STOPWORDS]
    sentence_president = [w for w in sentence_president if w not in STOPWORDS]








Compute the Word Movers Distance between the two sentences.


.. code-block:: default

    distance = wv.wmdistance(sentence_obama, sentence_president)
    print(f"Word Movers Distance is {distance} (lower means closer)")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    'Word Movers Distance is 0.015923231075180694 (lower means closer)'




That's all! You've made it to the end of this tutorial.



.. code-block:: default

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('fasttext-logo-color-web.png')
    imgplot = plt.imshow(img)
    _ = plt.axis('off')



.. image:: /auto_examples/tutorials/images/sphx_glr_run_fasttext_001.png
    :alt: run fasttext
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  28.645 seconds)

**Estimated memory usage:**  2975 MB


.. _sphx_glr_download_auto_examples_tutorials_run_fasttext.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_fasttext.py <run_fasttext.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_fasttext.ipynb <run_fasttext.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
