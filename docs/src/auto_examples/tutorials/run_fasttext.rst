.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_fasttext.py>` to download the full example code
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

    <gensim.models.fasttext.FastText object at 0x7facf7a1f780>


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

    <gensim.models.fasttext.FastText object at 0x7facfa03fdd8>


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

    array([ 0.0901682 ,  0.00294852, -0.58242214,  0.48336455,  0.60207903,
           -0.3259916 , -0.18890728, -0.04535883,  0.427281  ,  0.3354491 ,
           -0.6369988 , -0.01205245, -0.6746157 ,  0.41458675,  0.28214553,
           -0.06317813, -0.19516222,  0.18986242,  0.24284747, -0.37776226,
           -0.24019355,  0.2740346 , -0.37726364,  0.02587009, -0.82512945,
            0.72943956,  0.11839548,  0.17816432,  0.40805727,  0.00937917,
           -0.64387745,  0.23085909,  0.08261522, -0.4865105 ,  0.47281662,
            0.10547248, -0.15922274, -0.0661319 ,  0.41995323,  0.2191626 ,
           -0.0115916 , -0.06945843,  0.3894297 , -0.05846132,  0.1273995 ,
            0.1828662 , -0.13160151,  0.21973433, -0.0196108 , -0.3849172 ,
           -0.57192934, -0.5500985 ,  0.07157315,  0.01165739,  0.40098342,
           -0.8021547 , -0.10948794, -0.18269393,  0.00565537, -0.02862216,
            0.21497464, -0.11382702, -0.49290702, -0.10057615, -0.5354987 ,
            0.3436918 ,  0.0081019 ,  0.15149334,  0.02953579,  0.46941382,
           -0.56941664, -0.509449  , -0.00840828, -0.12042598, -0.33464944,
            0.02867694,  0.27383512,  0.11274044, -0.14110038,  0.17511652,
            0.45666447,  0.05277738, -0.17294791,  0.45916864, -0.39082173,
           -0.25972942,  0.12153289,  0.28688145,  0.323132  ,  0.3001046 ,
           -0.17770444,  0.12484344, -0.03607898, -0.1898243 , -0.2756515 ,
            0.5639621 ,  0.28384823,  0.44537458,  0.5499145 ,  0.4810851 ],
          dtype=float32)



.. code-block:: default

    print(model['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 0.07931967,  0.00311344, -0.50677073,  0.4198445 ,  0.5229806 ,
           -0.28508183, -0.16503437, -0.03866487,  0.3714626 ,  0.2925421 ,
           -0.5563491 , -0.01190763, -0.58725286,  0.36214375,  0.24150082,
           -0.05469211, -0.17193176,  0.16345991,  0.2096496 , -0.32910353,
           -0.20896138,  0.23933496, -0.32894656,  0.02317971, -0.71860677,
            0.63443   ,  0.10297966,  0.15547429,  0.35519433,  0.00773287,
           -0.55812263,  0.20209104,  0.07059211, -0.42544958,  0.4112376 ,
            0.09150278, -0.1371011 , -0.05883201,  0.3654037 ,  0.1915147 ,
           -0.0091759 , -0.06058947,  0.33839184, -0.04917968,  0.10979534,
            0.15938555, -0.11809663,  0.19301064, -0.01769728, -0.3335203 ,
           -0.5002038 , -0.47790018,  0.06343117,  0.01110996,  0.34957272,
           -0.69819194, -0.09538081, -0.15783711,  0.00260112, -0.02460477,
            0.18662988, -0.09990683, -0.4287245 , -0.08811973, -0.46654388,
            0.29852834,  0.00716597,  0.13173096,  0.02747719,  0.40931824,
           -0.49555174, -0.44295332, -0.00736047, -0.10571075, -0.29105732,
            0.02416036,  0.23839389,  0.0983296 , -0.12089454,  0.15162577,
            0.3982688 ,  0.04576813, -0.15136947,  0.39875692, -0.33911076,
           -0.22541183,  0.10782126,  0.2509758 ,  0.2805652 ,  0.26189   ,
           -0.15499805,  0.10841402, -0.03278403, -0.16494651, -0.23972231,
            0.49111483,  0.24609719,  0.3861827 ,  0.4782151 ,  0.41880232],
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

    [('study', 0.9982776641845703),
     ('Arafat', 0.9982735514640808),
     ('boat', 0.9982701539993286),
     ('Arafat,', 0.9982678890228271),
     ('"That', 0.9982609748840332),
     ('often', 0.9982587099075317),
     ('Endeavour', 0.9982519745826721),
     ('north.', 0.9982508420944214),
     ("Arafat's", 0.9982452392578125),
     ('details', 0.998244047164917)]



.. code-block:: default

    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999515



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

    [('1', 0.24167776107788086),
     ('40', 0.23674552142620087),
     ('2', 0.23436526954174042),
     ('20', 0.2326813042163849),
     ('26', 0.23191922903060913),
     ('blaze', 0.23159129917621613),
     ('UN', 0.23136889934539795),
     ('keep', 0.23088328540325165),
     ('...', 0.23050493001937866),
     ('As', 0.23039135336875916)]



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
                  ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                  ('LONG', 'LONGER', 'GREAT', 'GREATER')],
      'incorrect': [('GOOD', 'BETTER', 'LONG', 'LONGER'),
                    ('GOOD', 'BETTER', 'LOW', 'LOWER'),
                    ('GREAT', 'GREATER', 'LONG', 'LONGER'),
                    ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                    ('LONG', 'LONGER', 'LOW', 'LOWER'),
                    ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                    ('LOW', 'LOWER', 'LONG', 'LONGER')],
      'section': 'gram3-comparative'},
     {'correct': [('GREAT', 'GREATEST', 'LARGE', 'LARGEST')],
      'incorrect': [('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                    ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                    ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                    ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                    ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'GREAT', 'GREATEST')],
      'section': 'gram4-superlative'},
     {'correct': [('PLAY', 'PLAYING', 'SAY', 'SAYING'),
                  ('PLAY', 'PLAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'LOOK', 'LOOKING'),
                  ('SAY', 'SAYING', 'PLAY', 'PLAYING')],
      'incorrect': [('GO', 'GOING', 'LOOK', 'LOOKING'),
                    ('GO', 'GOING', 'PLAY', 'PLAYING'),
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
                  ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                  ('LONG', 'LONGER', 'GREAT', 'GREATER'),
                  ('GREAT', 'GREATEST', 'LARGE', 'LARGEST'),
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
                    ('GREAT', 'GREATER', 'GOOD', 'BETTER'),
                    ('LONG', 'LONGER', 'LOW', 'LOWER'),
                    ('LONG', 'LONGER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GOOD', 'BETTER'),
                    ('LOW', 'LOWER', 'GREAT', 'GREATER'),
                    ('LOW', 'LOWER', 'LONG', 'LONGER'),
                    ('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                    ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                    ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
                    ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'BIG', 'BIGGEST'),
                    ('GREAT', 'GREATEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'BIG', 'BIGGEST'),
                    ('LARGE', 'LARGEST', 'GOOD', 'BEST'),
                    ('LARGE', 'LARGEST', 'GREAT', 'GREATEST'),
                    ('GO', 'GOING', 'LOOK', 'LOOKING'),
                    ('GO', 'GOING', 'PLAY', 'PLAYING'),
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

    1.3930025392232537


That's all! You've made it to the end of this tutorial.

Here we show a pretty fastText logo so that our gallery picks it up as a thumbnail.



.. code-block:: default

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('fasttext-logo-color-web.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/tutorials/images/sphx_glr_run_fasttext_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  23.237 seconds)

**Estimated memory usage:**  4482 MB


.. _sphx_glr_download_auto_examples_tutorials_run_fasttext.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_fasttext.py <run_fasttext.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_fasttext.ipynb <run_fasttext.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
