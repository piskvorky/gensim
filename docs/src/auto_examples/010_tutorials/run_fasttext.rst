.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_fasttext.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_fasttext.py:


.. _fasttext_tutorial_py:

FastText Tutorial
=================

This tutorial is about using `fastText <https://github.com/facebookresearch/fastText>`_ model in Gensim.
Here, we'll learn to work with fastText library for training word-embedding models, saving & loading them and performing similarity operations & vector lookups analogous to Word2Vec.

When to use FastText?
---------------------


The main principle behind fastText is that the morphological structure of a word carries important information about the meaning of the word, which is not taken into account by traditional word embeddings, which train a unique word embedding for every individual word. This is especially significant for morphologically rich languages (German, Turkish) in which a single word can have a large number of morphological forms, each of which might occur rarely, thus making it hard to train good word embeddings.


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

    <gensim.models.fasttext.FastText object at 0x7f02528fda58>


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
    model.save('saved_model_gensim')
    loaded_model = FT_gensim.load('saved_model_gensim')
    print(loaded_model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <gensim.models.fasttext.FastText object at 0x7f0252c7b160>


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

    array([ 9.68532935e-02,  7.53772864e-03, -5.85787535e-01,  5.01900017e-01,
            5.97002208e-01, -3.22997630e-01, -1.89923018e-01, -3.03466413e-02,
            4.35077220e-01,  3.41016859e-01, -6.40107334e-01, -1.82615593e-02,
           -6.73842490e-01,  4.12545979e-01,  2.96752930e-01, -5.60669079e-02,
           -1.83477640e-01,  1.81586832e-01,  2.40116790e-01, -3.78429711e-01,
           -2.29927719e-01,  2.80625165e-01, -3.82913530e-01,  1.92540623e-02,
           -8.28781426e-01,  7.46151507e-01,  1.21208429e-01,  1.72570542e-01,
            4.13714051e-01,  1.34509010e-03, -6.58230484e-01,  2.36849174e-01,
            9.31602493e-02, -4.86029893e-01,  4.70224082e-01,  1.13173991e-01,
           -1.57694548e-01, -7.43469074e-02,  4.30460751e-01,  2.25712672e-01,
            6.96012634e-04, -7.92741925e-02,  3.97076160e-01, -6.33269921e-02,
            1.31146774e-01,  1.84730470e-01, -1.29139245e-01,  2.22790375e-01,
           -2.07373183e-02, -3.83532912e-01, -5.73759258e-01, -5.61627388e-01,
            6.55515864e-02,  5.86396223e-03,  4.10754532e-01, -8.07851672e-01,
           -1.12943970e-01, -1.82702392e-01,  9.85942781e-03, -2.53139529e-02,
            2.14411363e-01, -1.20995745e-01, -4.91106421e-01, -9.31469351e-02,
           -5.42534471e-01,  3.62293780e-01, -6.55674003e-03,  1.56759426e-01,
            3.19635458e-02,  4.90581453e-01, -5.66761255e-01, -5.23110867e-01,
           -1.03973057e-02, -9.87688825e-02, -3.50620300e-01,  3.72875929e-02,
            2.78797328e-01,  1.00131914e-01, -1.42450854e-01,  1.84168115e-01,
            4.59850043e-01,  6.29884228e-02, -1.68175057e-01,  4.47754353e-01,
           -4.06847298e-01, -2.64195889e-01,  1.35069609e-01,  2.95037359e-01,
            3.24666053e-01,  2.98116088e-01, -1.73260227e-01,  1.27341181e-01,
           -3.79600078e-02, -1.90464646e-01, -2.63515115e-01,  5.54983020e-01,
            2.82217324e-01,  4.46204334e-01,  5.53820074e-01,  4.70821410e-01],
          dtype=float32)



.. code-block:: default

    print(model['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 0.08485256,  0.00709199, -0.50795406,  0.43446752,  0.51676315,
           -0.2815003 , -0.1653465 , -0.02552212,  0.376937  ,  0.296335  ,
           -0.55712414, -0.01726196, -0.5845559 ,  0.3591265 ,  0.25334477,
           -0.04835646, -0.16121016,  0.15573291,  0.20659533, -0.32859027,
           -0.19935194,  0.24424045, -0.33273613,  0.01735642, -0.7193255 ,
            0.6467734 ,  0.10506127,  0.1501022 ,  0.35888463,  0.0007384 ,
           -0.56865793,  0.20659256,  0.0794794 , -0.4235615 ,  0.40752798,
            0.09785194, -0.13529222, -0.06574824,  0.37323502,  0.19653909,
            0.00149382, -0.06888723,  0.34383968, -0.05322546,  0.11262683,
            0.16043907, -0.11555009,  0.19497375, -0.01862768, -0.3311437 ,
           -0.50003594, -0.48619714,  0.05800197,  0.00603414,  0.35679218,
           -0.70067173, -0.09802678, -0.15729271,  0.00622192, -0.02165668,
            0.185478  , -0.10577381, -0.42565048, -0.08135959, -0.47099203,
            0.3135938 , -0.00556303,  0.13582142,  0.02946983,  0.42623445,
           -0.4914924 , -0.45323166, -0.00905497, -0.08654751, -0.30384213,
            0.03154093,  0.24186537,  0.08704651, -0.12163028,  0.15891603,
            0.39960673,  0.05445196, -0.14669728,  0.38747063, -0.35179204,
           -0.22850554,  0.11917583,  0.25715488,  0.28089544,  0.25922963,
           -0.15060836,  0.1101789 , -0.03429657, -0.16494407, -0.22833037,
            0.48158392,  0.24381368,  0.3855345 ,  0.47989166,  0.4084021 ],
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

    0.9999929


Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.


Other similarity operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only


.. code-block:: default

    print(model.most_similar("nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('study', 0.9982758164405823),
     ('often', 0.9982642531394958),
     ('"That', 0.9982632398605347),
     ('Arafat', 0.9982630610466003),
     ('Arafat,', 0.9982588887214661),
     ('boat', 0.9982563257217407),
     ('north.', 0.9982514381408691),
     ('Endeavour', 0.9982463121414185),
     ('details', 0.9982396960258484),
     ("Arafat's", 0.9982390999794006)]



.. code-block:: default

    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.99995095



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

    [('1', 0.2429654449224472),
     ('40', 0.23754820227622986),
     ('2', 0.23473285138607025),
     ('20', 0.23319712281227112),
     ('26', 0.23305919766426086),
     ('UN', 0.23269976675510406),
     ('blaze', 0.23214034736156464),
     ('keep', 0.2312297224998474),
     ('As', 0.23102840781211853),
     ('...', 0.2306765913963318)]



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

    1.3932733855659962


That's all! You've made it to the end of this tutorial.

Here we show a pretty fastText logo so that our gallery picks it up as a thumbnail.



.. code-block:: default

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('fasttext-logo-color-web.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/010_tutorials/images/sphx_glr_run_fasttext_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  30.269 seconds)

**Estimated memory usage:**  4506 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_fasttext.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/010_tutorials/run_fasttext.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_fasttext.py <run_fasttext.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_fasttext.ipynb <run_fasttext.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
