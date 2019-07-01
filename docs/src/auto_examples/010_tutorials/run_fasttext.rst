.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_fasttext.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_fasttext.py:


.. _fasttext_tutorial_py:

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

    <gensim.models.fasttext.FastText object at 0x7f2b1b63f6d8>


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

    <gensim.models.fasttext.FastText object at 0x7f2b1b63fb38>


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

    array([ 0.08809454,  0.00462394, -0.5864776 ,  0.48764512,  0.59266067,
           -0.3289348 , -0.18769875, -0.05185637,  0.42779785,  0.32606098,
           -0.63238895, -0.01023355, -0.66825646,  0.4079376 ,  0.28898352,
           -0.05819847, -0.20956703,  0.19218387,  0.22385162, -0.37211645,
           -0.24080205,  0.28925216, -0.3845128 ,  0.01912642, -0.82059234,
            0.7312545 ,  0.13218074,  0.17801882,  0.40161476,  0.0055988 ,
           -0.645143  ,  0.24397795,  0.09603377, -0.48711702,  0.45796207,
            0.10446949, -0.15425271, -0.06821961,  0.41876414,  0.21820873,
           -0.01122862, -0.07099853,  0.39802697, -0.06133444,  0.12586786,
            0.17753069, -0.13177377,  0.21863978, -0.01259975, -0.38519612,
           -0.56651664, -0.5438191 ,  0.05280963,  0.01127269,  0.39372838,
           -0.7988662 , -0.10466399, -0.18153362,  0.00976756, -0.03204089,
            0.20417729, -0.10242397, -0.49522698, -0.10040525, -0.53820163,
            0.3599945 ,  0.01341753,  0.14818543,  0.02709586,  0.48140046,
           -0.55834526, -0.509605  ,  0.00542481, -0.10311265, -0.33714786,
            0.03041851,  0.27081314,  0.10661431, -0.14005469,  0.1771436 ,
            0.45091954,  0.0537632 , -0.16273336,  0.45720157, -0.39692032,
           -0.25611022,  0.12993893,  0.27449307,  0.3198574 ,  0.314403  ,
           -0.17170118,  0.12574679, -0.02804552, -0.18872   , -0.27759033,
            0.5663356 ,  0.2748947 ,  0.4453878 ,  0.54980594,  0.47006914],
          dtype=float32)



.. code-block:: default

    print(model['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 0.07766417,  0.00458002, -0.51123786,  0.4243402 ,  0.5157236 ,
           -0.288177  , -0.16427347, -0.04439441,  0.37260023,  0.28485677,
           -0.5533233 , -0.01036947, -0.58277214,  0.3569826 ,  0.24792239,
           -0.05046317, -0.18477666,  0.16579315,  0.19349647, -0.32478225,
           -0.20988104,  0.25304878, -0.3358584 ,  0.01734176, -0.71593636,
            0.63718253,  0.11516301,  0.15562919,  0.35023203,  0.00446516,
           -0.5602484 ,  0.21387918,  0.08240761, -0.42676434,  0.39907047,
            0.09078669, -0.1330423 , -0.06077712,  0.3650524 ,  0.19104657,
           -0.00888137, -0.06201624,  0.34642476, -0.05174574,  0.10860538,
            0.15498231, -0.1184248 ,  0.19236074, -0.01163285, -0.33424512,
           -0.49623632, -0.473113  ,  0.04720827,  0.01079329,  0.34376985,
           -0.6963661 , -0.09131003, -0.15706775,  0.00616824, -0.0276337 ,
            0.17751509, -0.09014308, -0.4314059 , -0.08809985, -0.4695263 ,
            0.3131413 ,  0.0118192 ,  0.12902623,  0.02537876,  0.42031896,
           -0.48662105, -0.44367906,  0.00466951, -0.09073911, -0.2936068 ,
            0.02572422,  0.23610575,  0.09312797, -0.12016112,  0.15358655,
            0.39381853,  0.04669925, -0.14271255,  0.39762706, -0.3448826 ,
           -0.2225944 ,  0.11527011,  0.2405121 ,  0.2781124 ,  0.2747289 ,
           -0.14999147,  0.10934623, -0.02584376, -0.16420597, -0.24172637,
            0.49384513,  0.23865801,  0.3867399 ,  0.47874016,  0.4097537 ],
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

    0.99999267


Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.


Other similarity operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only


.. code-block:: default

    print(model.most_similar("nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('Arafat', 0.9982844591140747),
     ('boat', 0.9982783198356628),
     ('Arafat,', 0.9982778429985046),
     ('"That', 0.9982759356498718),
     ('study', 0.9982714056968689),
     ('often', 0.9982637763023376),
     ('north.', 0.9982573986053467),
     ("Arafat's", 0.998254656791687),
     ('Endeavour', 0.9982540011405945),
     ('details', 0.998242974281311)]



.. code-block:: default

    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999513



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

    [('1', 0.24139031767845154),
     ('40', 0.23604942858219147),
     ('2', 0.23340445756912231),
     ('26', 0.23200523853302002),
     ('20', 0.23162022233009338),
     ('UN', 0.23103418946266174),
     ('blaze', 0.23080500960350037),
     ('keep', 0.2298724502325058),
     ('As', 0.22917315363883972),
     ('...', 0.22909186780452728)]



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

    1.3949753428325056


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

   **Total running time of the script:** ( 0 minutes  21.915 seconds)

**Estimated memory usage:**  4477 MB


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
