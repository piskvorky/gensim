.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_030_fasttext.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_030_fasttext.py:


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

    <gensim.models.fasttext.FastText object at 0x7fe399b9fdd8>


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

    <gensim.models.fasttext.FastText object at 0x7fe260091cc0>


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

    array([ 8.4455185e-02, -1.0038717e-02, -5.9108943e-01,  4.5057020e-01,
            6.1291409e-01, -2.9829705e-01, -1.9161057e-01, -5.8460418e-02,
            4.2386848e-01,  3.4023973e-01, -6.3242573e-01,  3.8265304e-03,
           -7.0242476e-01,  4.0764502e-01,  2.6526663e-01, -7.3893599e-02,
           -2.1616289e-01,  1.8904778e-01,  2.3743360e-01, -3.7548923e-01,
           -2.4184871e-01,  2.6657289e-01, -3.5883111e-01,  4.5346294e-02,
           -8.3400190e-01,  7.4467582e-01,  1.3188230e-01,  1.9738491e-01,
            4.0142837e-01,  1.9379098e-02, -6.5326267e-01,  2.4698308e-01,
            8.3173893e-02, -4.9540538e-01,  4.7354230e-01,  7.7310435e-02,
           -1.5998441e-01, -3.8665496e-02,  4.0121624e-01,  1.8975993e-01,
           -3.8958719e-04, -7.5674988e-02,  3.8030967e-01, -5.9949864e-02,
            1.2742083e-01,  1.8134865e-01, -1.0703909e-01,  2.0883927e-01,
           -2.0948178e-03, -3.9545634e-01, -5.7543284e-01, -5.4236752e-01,
            3.5078954e-02,  2.1431992e-02,  3.7261876e-01, -8.1508273e-01,
           -1.0877564e-01, -2.0154855e-01,  1.3842276e-02, -4.1202016e-02,
            2.2705761e-01, -9.6409880e-02, -4.9242309e-01, -1.3487200e-01,
           -5.4028577e-01,  3.0330792e-01,  1.9712528e-02,  1.3594845e-01,
            2.6747780e-02,  4.6500212e-01, -5.7465625e-01, -5.3054690e-01,
           -9.4801508e-04, -1.4894724e-01, -3.2207409e-01,  3.5728578e-02,
            2.5958005e-01,  1.3595429e-01, -1.2496351e-01,  1.8443903e-01,
            4.6904901e-01,  3.6963750e-02, -1.4681615e-01,  4.6917006e-01,
           -3.6232165e-01, -2.5221449e-01,  1.0284642e-01,  2.8069499e-01,
            3.0935711e-01,  2.8910854e-01, -1.9946001e-01,  1.2434962e-01,
           -2.9544644e-02, -1.7803410e-01, -2.9119951e-01,  5.7917684e-01,
            2.8827086e-01,  4.9247938e-01,  5.5208868e-01,  4.5207450e-01],
          dtype=float32)



.. code-block:: default

    print(model['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([ 7.4382693e-02, -8.1921108e-03, -5.1449597e-01,  3.9142546e-01,
            5.3269345e-01, -2.6111686e-01, -1.6746593e-01, -5.0075538e-02,
            3.6867455e-01,  2.9684237e-01, -5.5263603e-01,  1.9028945e-03,
           -6.1175287e-01,  3.5626999e-01,  2.2692643e-01, -6.4051077e-02,
           -1.9029701e-01,  1.6284081e-01,  2.0505545e-01, -3.2731465e-01,
           -2.1052462e-01,  2.3295945e-01, -3.1307852e-01,  4.0144339e-02,
           -7.2678447e-01,  6.4808446e-01,  1.1478644e-01,  1.7230231e-01,
            3.4964937e-01,  1.6442286e-02, -5.6662863e-01,  2.1622521e-01,
            7.1130097e-02, -4.3346009e-01,  4.1215119e-01,  6.7024469e-02,
           -1.3786827e-01, -3.4958296e-02,  3.4931681e-01,  1.6603869e-01,
            5.6221359e-04, -6.6077292e-02,  3.3085573e-01, -5.0529730e-02,
            1.0994045e-01,  1.5825827e-01, -9.6847363e-02,  1.8375373e-01,
           -2.4567798e-03, -3.4309617e-01, -5.0386149e-01, -4.7173393e-01,
            3.1731639e-02,  1.9644525e-02,  3.2527378e-01, -7.1030229e-01,
           -9.4889559e-02, -1.7446031e-01,  9.7353803e-03, -3.5604339e-02,
            1.9738355e-01, -8.4844828e-02, -4.2885214e-01, -1.1809799e-01,
           -4.7122967e-01,  2.6369980e-01,  1.7286493e-02,  1.1833906e-01,
            2.5069227e-02,  4.0593657e-01, -5.0070614e-01, -4.6183729e-01,
           -8.6749974e-04, -1.3066567e-01, -2.8041545e-01,  3.0339070e-02,
            2.2625467e-01,  1.1867213e-01, -1.0698760e-01,  1.5990424e-01,
            4.0952358e-01,  3.2027539e-02, -1.2878175e-01,  4.0795246e-01,
           -3.1465945e-01, -2.1913464e-01,  9.1661632e-02,  2.4584998e-01,
            2.6889491e-01,  2.5260991e-01, -1.7412896e-01,  1.0811622e-01,
           -2.7129170e-02, -1.5485901e-01, -2.5353333e-01,  5.0491500e-01,
            2.5022638e-01,  4.2763287e-01,  4.8061603e-01,  3.9399058e-01],
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

    0.99999285


Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided `here <Word2Vec_FastText_Comparison.ipynb>`_.


Other similarity operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only


.. code-block:: default

    print(model.most_similar("nights"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('boat', 0.9982773065567017),
     ('study', 0.9982753992080688),
     ('"That', 0.9982752799987793),
     ('often', 0.9982641935348511),
     ('north.', 0.9982575178146362),
     ('Endeavour', 0.9982515573501587),
     ('beyond', 0.9982446432113647),
     ('stage', 0.9982429146766663),
     ('accept', 0.9982421398162842),
     ('Commonwealth', 0.9982419013977051)]



.. code-block:: default

    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9999508



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

    [('1', 0.242139533162117),
     ('40', 0.23757682740688324),
     ('2', 0.23549434542655945),
     ('20', 0.2334955781698227),
     ('26', 0.23319362103939056),
     ('UN', 0.232555091381073),
     ('blaze', 0.23230190575122833),
     ('keep', 0.2315463423728943),
     ('As', 0.23101860284805298),
     ('...', 0.23101305961608887)]



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
     {'correct': [('GOOD', 'BEST', 'LARGE', 'LARGEST'),
                  ('GREAT', 'GREATEST', 'LARGE', 'LARGEST')],
      'incorrect': [('BIG', 'BIGGEST', 'GOOD', 'BEST'),
                    ('BIG', 'BIGGEST', 'GREAT', 'GREATEST'),
                    ('BIG', 'BIGGEST', 'LARGE', 'LARGEST'),
                    ('GOOD', 'BEST', 'GREAT', 'GREATEST'),
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
                  ('GREAT', 'GREATER', 'LOW', 'LOWER'),
                  ('LONG', 'LONGER', 'GREAT', 'GREATER'),
                  ('GOOD', 'BEST', 'LARGE', 'LARGEST'),
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

    1.3962668012305497


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

   **Total running time of the script:** ( 0 minutes  25.365 seconds)

**Estimated memory usage:**  4530 MB


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
