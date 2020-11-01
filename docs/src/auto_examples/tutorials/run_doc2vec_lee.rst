.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_doc2vec_lee.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_tutorials_run_doc2vec_lee.py:


Doc2Vec Model
=============

Introduces Gensim's Doc2Vec model and demonstrates its use on the
`Lee Corpus <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`__.



.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








Doc2Vec is a :ref:`core_concepts_model` that represents each
:ref:`core_concepts_document` as a :ref:`core_concepts_vector`.  This
tutorial introduces the model and demonstrates how to train and assess it.

Here's a list of what we'll be doing:

0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
4. Assess the model
5. Test the model on the test corpus

Review: Bag-of-words
--------------------

.. Note:: Feel free to skip these review sections if you're already familiar with the models.

You may be familiar with the `bag-of-words model
<https://en.wikipedia.org/wiki/Bag-of-words_model>`_ from the
:ref:`core_concepts_vector` section.
This model transforms each document to a fixed-length vector of integers.
For example, given the sentences:

- ``John likes to watch movies. Mary likes movies too.``
- ``John also likes to watch football games. Mary hates football.``

The model outputs the vectors:

- ``[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]``
- ``[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]``

Each vector has 10 elements, where each element counts the number of times a
particular word occurred in the document.
The order of elements is arbitrary.
In the example above, the order of the elements corresponds to the words:
``["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]``.

Bag-of-words models are surprisingly effective, but have several weaknesses.

First, they lose all information about word order: "John likes Mary" and
"Mary likes John" correspond to identical vectors. There is a solution: bag
of `n-grams <https://en.wikipedia.org/wiki/N-gram>`__
models consider word phrases of length n to represent documents as
fixed-length vectors to capture local word order but suffer from data
sparsity and high dimensionality.

Second, the model does not attempt to learn the meaning of the underlying
words, and as a consequence, the distance between vectors doesn't always
reflect the difference in meaning.  The ``Word2Vec`` model addresses this
second problem.

Review: ``Word2Vec`` Model
--------------------------

``Word2Vec`` is a more recent model that embeds words in a lower-dimensional
vector space using a shallow neural network. The result is a set of
word-vectors where vectors close together in vector space have similar
meanings based on context, and word-vectors distant to each other have
differing meanings. For example, ``strong`` and ``powerful`` would be close
together and ``strong`` and ``Paris`` would be relatively far.

Gensim's :py:class:`~gensim.models.word2vec.Word2Vec` class implements this model.

With the ``Word2Vec`` model, we can calculate the vectors for each **word** in a document.
But what if we want to calculate a vector for the **entire document**\ ?
We could average the vectors for each word in the document - while this is quick and crude, it can often be useful.
However, there is a better way...

Introducing: Paragraph Vector
-----------------------------

.. Important:: In Gensim, we refer to the Paragraph Vector model as ``Doc2Vec``.

Le and Mikolov in 2014 introduced the `Doc2Vec algorithm <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__,
which usually outperforms such simple-averaging of ``Word2Vec`` vectors.

The basic idea is: act as if a document has another floating word-like
vector, which contributes to all training predictions, and is updated like
other word-vectors, but we will call it a doc-vector. Gensim's
:py:class:`~gensim.models.doc2vec.Doc2Vec` class implements this algorithm.

There are two implementations:

1. Paragraph Vector - Distributed Memory (PV-DM)
2. Paragraph Vector - Distributed Bag of Words (PV-DBOW)

.. Important::
  Don't let the implementation details below scare you.
  They're advanced material: if it's too much, then move on to the next section.

PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training
a neural network on the synthetic task of predicting a center word based an
average of both context word-vectors and the full document's doc-vector.

PV-DBOW is analogous to Word2Vec SG. The doc-vectors are obtained by training
a neural network on the synthetic task of predicting a target word just from
the full document's doc-vector. (It is also common to combine this with
skip-gram testing, using both the doc-vector and nearby word-vectors to
predict a single target word, but only one at a time.)

Prepare the Training and Test Data
----------------------------------

For this tutorial, we'll be training our model using the `Lee Background
Corpus
<https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
included in gensim. This corpus contains 314 documents selected from the
Australian Broadcasting Corporation’s news mail service, which provides text
e-mails of headline stories and covers a number of broad topics.

And we'll test our model by eye using the much shorter `Lee Corpus
<https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
which contains 50 documents.



.. code-block:: default


    import os
    import gensim
    # Set file names for train and test data
    test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
    lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
    lee_test_file = os.path.join(test_data_dir, 'lee.cor')








Define a Function to Read and Preprocess Text
---------------------------------------------

Below, we define a function to:

- open the train/test file (with latin encoding)
- read the file line-by-line
- pre-process each line (tokenize text into individual words, remove punctuation, set to lowercase, etc)

The file we're reading is a **corpus**.
Each line of the file is a **document**.

.. Important::
  To train the model, we'll need to associate a tag/number with each document
  of the training corpus. In our case, the tag is simply the zero-based line
  number.



.. code-block:: default

    import smart_open

    def read_corpus(fname, tokens_only=False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    train_corpus = list(read_corpus(lee_train_file))
    test_corpus = list(read_corpus(lee_test_file, tokens_only=True))








Let's take a look at the training corpus



.. code-block:: default

    print(train_corpus[:2])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [TaggedDocument(words=['hundreds', 'of', 'people', 'have', 'been', 'forced', 'to', 'vacate', 'their', 'homes', 'in', 'the', 'southern', 'highlands', 'of', 'new', 'south', 'wales', 'as', 'strong', 'winds', 'today', 'pushed', 'huge', 'bushfire', 'towards', 'the', 'town', 'of', 'hill', 'top', 'new', 'blaze', 'near', 'goulburn', 'south', 'west', 'of', 'sydney', 'has', 'forced', 'the', 'closure', 'of', 'the', 'hume', 'highway', 'at', 'about', 'pm', 'aedt', 'marked', 'deterioration', 'in', 'the', 'weather', 'as', 'storm', 'cell', 'moved', 'east', 'across', 'the', 'blue', 'mountains', 'forced', 'authorities', 'to', 'make', 'decision', 'to', 'evacuate', 'people', 'from', 'homes', 'in', 'outlying', 'streets', 'at', 'hill', 'top', 'in', 'the', 'new', 'south', 'wales', 'southern', 'highlands', 'an', 'estimated', 'residents', 'have', 'left', 'their', 'homes', 'for', 'nearby', 'mittagong', 'the', 'new', 'south', 'wales', 'rural', 'fire', 'service', 'says', 'the', 'weather', 'conditions', 'which', 'caused', 'the', 'fire', 'to', 'burn', 'in', 'finger', 'formation', 'have', 'now', 'eased', 'and', 'about', 'fire', 'units', 'in', 'and', 'around', 'hill', 'top', 'are', 'optimistic', 'of', 'defending', 'all', 'properties', 'as', 'more', 'than', 'blazes', 'burn', 'on', 'new', 'year', 'eve', 'in', 'new', 'south', 'wales', 'fire', 'crews', 'have', 'been', 'called', 'to', 'new', 'fire', 'at', 'gunning', 'south', 'of', 'goulburn', 'while', 'few', 'details', 'are', 'available', 'at', 'this', 'stage', 'fire', 'authorities', 'says', 'it', 'has', 'closed', 'the', 'hume', 'highway', 'in', 'both', 'directions', 'meanwhile', 'new', 'fire', 'in', 'sydney', 'west', 'is', 'no', 'longer', 'threatening', 'properties', 'in', 'the', 'cranebrook', 'area', 'rain', 'has', 'fallen', 'in', 'some', 'parts', 'of', 'the', 'illawarra', 'sydney', 'the', 'hunter', 'valley', 'and', 'the', 'north', 'coast', 'but', 'the', 'bureau', 'of', 'meteorology', 'claire', 'richards', 'says', 'the', 'rain', 'has', 'done', 'little', 'to', 'ease', 'any', 'of', 'the', 'hundred', 'fires', 'still', 'burning', 'across', 'the', 'state', 'the', 'falls', 'have', 'been', 'quite', 'isolated', 'in', 'those', 'areas', 'and', 'generally', 'the', 'falls', 'have', 'been', 'less', 'than', 'about', 'five', 'millimetres', 'she', 'said', 'in', 'some', 'places', 'really', 'not', 'significant', 'at', 'all', 'less', 'than', 'millimetre', 'so', 'there', 'hasn', 'been', 'much', 'relief', 'as', 'far', 'as', 'rain', 'is', 'concerned', 'in', 'fact', 'they', 've', 'probably', 'hampered', 'the', 'efforts', 'of', 'the', 'firefighters', 'more', 'because', 'of', 'the', 'wind', 'gusts', 'that', 'are', 'associated', 'with', 'those', 'thunderstorms'], tags=[0]), TaggedDocument(words=['indian', 'security', 'forces', 'have', 'shot', 'dead', 'eight', 'suspected', 'militants', 'in', 'night', 'long', 'encounter', 'in', 'southern', 'kashmir', 'the', 'shootout', 'took', 'place', 'at', 'dora', 'village', 'some', 'kilometers', 'south', 'of', 'the', 'kashmiri', 'summer', 'capital', 'srinagar', 'the', 'deaths', 'came', 'as', 'pakistani', 'police', 'arrested', 'more', 'than', 'two', 'dozen', 'militants', 'from', 'extremist', 'groups', 'accused', 'of', 'staging', 'an', 'attack', 'on', 'india', 'parliament', 'india', 'has', 'accused', 'pakistan', 'based', 'lashkar', 'taiba', 'and', 'jaish', 'mohammad', 'of', 'carrying', 'out', 'the', 'attack', 'on', 'december', 'at', 'the', 'behest', 'of', 'pakistani', 'military', 'intelligence', 'military', 'tensions', 'have', 'soared', 'since', 'the', 'raid', 'with', 'both', 'sides', 'massing', 'troops', 'along', 'their', 'border', 'and', 'trading', 'tit', 'for', 'tat', 'diplomatic', 'sanctions', 'yesterday', 'pakistan', 'announced', 'it', 'had', 'arrested', 'lashkar', 'taiba', 'chief', 'hafiz', 'mohammed', 'saeed', 'police', 'in', 'karachi', 'say', 'it', 'is', 'likely', 'more', 'raids', 'will', 'be', 'launched', 'against', 'the', 'two', 'groups', 'as', 'well', 'as', 'other', 'militant', 'organisations', 'accused', 'of', 'targetting', 'india', 'military', 'tensions', 'between', 'india', 'and', 'pakistan', 'have', 'escalated', 'to', 'level', 'not', 'seen', 'since', 'their', 'war'], tags=[1])]




And the testing corpus looks like this:



.. code-block:: default

    print(test_corpus[:2])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [['the', 'national', 'executive', 'of', 'the', 'strife', 'torn', 'democrats', 'last', 'night', 'appointed', 'little', 'known', 'west', 'australian', 'senator', 'brian', 'greig', 'as', 'interim', 'leader', 'shock', 'move', 'likely', 'to', 'provoke', 'further', 'conflict', 'between', 'the', 'party', 'senators', 'and', 'its', 'organisation', 'in', 'move', 'to', 'reassert', 'control', 'over', 'the', 'party', 'seven', 'senators', 'the', 'national', 'executive', 'last', 'night', 'rejected', 'aden', 'ridgeway', 'bid', 'to', 'become', 'interim', 'leader', 'in', 'favour', 'of', 'senator', 'greig', 'supporter', 'of', 'deposed', 'leader', 'natasha', 'stott', 'despoja', 'and', 'an', 'outspoken', 'gay', 'rights', 'activist'], ['cash', 'strapped', 'financial', 'services', 'group', 'amp', 'has', 'shelved', 'million', 'plan', 'to', 'buy', 'shares', 'back', 'from', 'investors', 'and', 'will', 'raise', 'million', 'in', 'fresh', 'capital', 'after', 'profits', 'crashed', 'in', 'the', 'six', 'months', 'to', 'june', 'chief', 'executive', 'paul', 'batchelor', 'said', 'the', 'result', 'was', 'solid', 'in', 'what', 'he', 'described', 'as', 'the', 'worst', 'conditions', 'for', 'stock', 'markets', 'in', 'years', 'amp', 'half', 'year', 'profit', 'sank', 'per', 'cent', 'to', 'million', 'or', 'share', 'as', 'australia', 'largest', 'investor', 'and', 'fund', 'manager', 'failed', 'to', 'hit', 'projected', 'per', 'cent', 'earnings', 'growth', 'targets', 'and', 'was', 'battered', 'by', 'falling', 'returns', 'on', 'share', 'markets']]




Notice that the testing corpus is just a list of lists and does not contain
any tags.


Training the Model
------------------

Now, we'll instantiate a Doc2Vec model with a vector size with 50 dimensions and
iterating over the training corpus 40 times. We set the minimum word count to
2 in order to discard words with very few occurrences. (Without a variety of
representative examples, retaining such infrequent words can often make a
model worse!) Typical iteration counts in the published `Paragraph Vector paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__
results, using 10s-of-thousands to millions of docs, are 10-20. More
iterations take more time and eventually reach a point of diminishing
returns.

However, this is a very very small dataset (300 documents) with shortish
documents (a few hundred words). Adding training passes can sometimes help
with such small datasets.



.. code-block:: default

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)








Build a vocabulary


.. code-block:: default

    model.build_vocab(train_corpus)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    2020-09-30 21:08:55,026 : INFO : collecting all words and their counts
    2020-09-30 21:08:55,027 : INFO : PROGRESS: at example #0, processed 0 words (0/s), 0 word types, 0 tags
    2020-09-30 21:08:55,043 : INFO : collected 6981 word types and 300 unique tags from a corpus of 300 examples and 58152 words
    2020-09-30 21:08:55,043 : INFO : Loading a fresh vocabulary
    2020-09-30 21:08:55,064 : INFO : effective_min_count=2 retains 3955 unique words (56% of original 6981, drops 3026)
    2020-09-30 21:08:55,064 : INFO : effective_min_count=2 leaves 55126 word corpus (94% of original 58152, drops 3026)
    2020-09-30 21:08:55,098 : INFO : deleting the raw counts dictionary of 6981 items
    2020-09-30 21:08:55,100 : INFO : sample=0.001 downsamples 46 most-common words
    2020-09-30 21:08:55,100 : INFO : downsampling leaves estimated 42390 word corpus (76.9% of prior 55126)
    2020-09-30 21:08:55,149 : INFO : estimated required memory for 3955 words and 50 dimensions: 3679500 bytes
    2020-09-30 21:08:55,149 : INFO : resetting layer weights




Essentially, the vocabulary is a list (accessible via
``model.wv.index_to_key``) of all of the unique words extracted from the training corpus.
Additional attributes for each word are available using the ``model.wv.get_vecattr()`` method,
For example, to see how many times ``penalty`` appeared in the training corpus:



.. code-block:: default

    print(f"Word 'penalty' appeared {model.wv.get_vecattr('penalty', 'count')} times in the training corpus.")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Word 'penalty' appeared 4 times in the training corpus.




Next, train the model on the corpus.
If optimized Gensim (with BLAS library) is being used, this should take no more than 3 seconds.
If the BLAS library is not being used, this should take no more than 2
minutes, so use optimized Gensim with BLAS if you value your time.



.. code-block:: default

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    2020-09-30 21:08:55,553 : INFO : training model with 3 workers on 3955 vocabulary and 50 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2020-09-30 21:08:55,613 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,614 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,614 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,614 : INFO : EPOCH - 1 : training on 58152 raw words (42784 effective words) took 0.1s, 751479 effective words/s
    2020-09-30 21:08:55,664 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,666 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,666 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,666 : INFO : EPOCH - 2 : training on 58152 raw words (42745 effective words) took 0.1s, 845101 effective words/s
    2020-09-30 21:08:55,718 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,719 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,720 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,720 : INFO : EPOCH - 3 : training on 58152 raw words (42605 effective words) took 0.1s, 810845 effective words/s
    2020-09-30 21:08:55,781 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,783 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,784 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,784 : INFO : EPOCH - 4 : training on 58152 raw words (42723 effective words) took 0.1s, 677810 effective words/s
    2020-09-30 21:08:55,846 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,847 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,848 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,848 : INFO : EPOCH - 5 : training on 58152 raw words (42641 effective words) took 0.1s, 682513 effective words/s
    2020-09-30 21:08:55,903 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,905 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,905 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,905 : INFO : EPOCH - 6 : training on 58152 raw words (42654 effective words) took 0.1s, 760381 effective words/s
    2020-09-30 21:08:55,960 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:55,962 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:55,964 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:55,964 : INFO : EPOCH - 7 : training on 58152 raw words (42751 effective words) took 0.1s, 741994 effective words/s
    2020-09-30 21:08:56,018 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,020 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,020 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,020 : INFO : EPOCH - 8 : training on 58152 raw words (42692 effective words) took 0.1s, 773631 effective words/s
    2020-09-30 21:08:56,076 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,078 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,081 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,081 : INFO : EPOCH - 9 : training on 58152 raw words (42745 effective words) took 0.1s, 719453 effective words/s
    2020-09-30 21:08:56,137 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,137 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,137 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,138 : INFO : EPOCH - 10 : training on 58152 raw words (42733 effective words) took 0.1s, 770082 effective words/s
    2020-09-30 21:08:56,195 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,196 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,197 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,197 : INFO : EPOCH - 11 : training on 58152 raw words (42791 effective words) took 0.1s, 734171 effective words/s
    2020-09-30 21:08:56,253 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,255 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,255 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,255 : INFO : EPOCH - 12 : training on 58152 raw words (42773 effective words) took 0.1s, 745248 effective words/s
    2020-09-30 21:08:56,316 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,318 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,318 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,318 : INFO : EPOCH - 13 : training on 58152 raw words (42793 effective words) took 0.1s, 702300 effective words/s
    2020-09-30 21:08:56,369 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,371 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,373 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,373 : INFO : EPOCH - 14 : training on 58152 raw words (42637 effective words) took 0.1s, 802259 effective words/s
    2020-09-30 21:08:56,421 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,425 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,426 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,426 : INFO : EPOCH - 15 : training on 58152 raw words (42686 effective words) took 0.1s, 820787 effective words/s
    2020-09-30 21:08:56,475 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,478 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,479 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,479 : INFO : EPOCH - 16 : training on 58152 raw words (42799 effective words) took 0.1s, 829690 effective words/s
    2020-09-30 21:08:56,530 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,530 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,533 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,534 : INFO : EPOCH - 17 : training on 58152 raw words (42733 effective words) took 0.1s, 794744 effective words/s
    2020-09-30 21:08:56,583 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,585 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,587 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,587 : INFO : EPOCH - 18 : training on 58152 raw words (42703 effective words) took 0.1s, 813146 effective words/s
    2020-09-30 21:08:56,638 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,640 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,640 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,641 : INFO : EPOCH - 19 : training on 58152 raw words (42763 effective words) took 0.1s, 822300 effective words/s
    2020-09-30 21:08:56,696 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,700 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,700 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,700 : INFO : EPOCH - 20 : training on 58152 raw words (42649 effective words) took 0.1s, 733047 effective words/s
    2020-09-30 21:08:56,752 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,753 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,754 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,754 : INFO : EPOCH - 21 : training on 58152 raw words (42701 effective words) took 0.1s, 822006 effective words/s
    2020-09-30 21:08:56,803 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,805 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,805 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,805 : INFO : EPOCH - 22 : training on 58152 raw words (42714 effective words) took 0.1s, 848390 effective words/s
    2020-09-30 21:08:56,857 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,857 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,859 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,860 : INFO : EPOCH - 23 : training on 58152 raw words (42740 effective words) took 0.1s, 811758 effective words/s
    2020-09-30 21:08:56,907 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,909 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,910 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,910 : INFO : EPOCH - 24 : training on 58152 raw words (42754 effective words) took 0.0s, 873741 effective words/s
    2020-09-30 21:08:56,959 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:56,960 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:56,960 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:56,960 : INFO : EPOCH - 25 : training on 58152 raw words (42704 effective words) took 0.0s, 862291 effective words/s
    2020-09-30 21:08:57,009 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,010 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,011 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,011 : INFO : EPOCH - 26 : training on 58152 raw words (42741 effective words) took 0.0s, 868076 effective words/s
    2020-09-30 21:08:57,059 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,062 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,063 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,063 : INFO : EPOCH - 27 : training on 58152 raw words (42610 effective words) took 0.1s, 830699 effective words/s
    2020-09-30 21:08:57,112 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,114 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,115 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,116 : INFO : EPOCH - 28 : training on 58152 raw words (42747 effective words) took 0.1s, 835959 effective words/s
    2020-09-30 21:08:57,164 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,169 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,170 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,170 : INFO : EPOCH - 29 : training on 58152 raw words (42755 effective words) took 0.1s, 804348 effective words/s
    2020-09-30 21:08:57,219 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,222 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,224 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,224 : INFO : EPOCH - 30 : training on 58152 raw words (42760 effective words) took 0.1s, 808636 effective words/s
    2020-09-30 21:08:57,271 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,273 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,273 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,273 : INFO : EPOCH - 31 : training on 58152 raw words (42727 effective words) took 0.0s, 889118 effective words/s
    2020-09-30 21:08:57,323 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,326 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,327 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,327 : INFO : EPOCH - 32 : training on 58152 raw words (42786 effective words) took 0.1s, 819149 effective words/s
    2020-09-30 21:08:57,377 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,378 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,379 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,379 : INFO : EPOCH - 33 : training on 58152 raw words (42614 effective words) took 0.1s, 828217 effective words/s
    2020-09-30 21:08:57,427 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,430 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,431 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,431 : INFO : EPOCH - 34 : training on 58152 raw words (42757 effective words) took 0.1s, 848700 effective words/s
    2020-09-30 21:08:57,476 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,479 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,481 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,481 : INFO : EPOCH - 35 : training on 58152 raw words (42713 effective words) took 0.0s, 881912 effective words/s
    2020-09-30 21:08:57,530 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,530 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,532 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,532 : INFO : EPOCH - 36 : training on 58152 raw words (42632 effective words) took 0.1s, 843930 effective words/s
    2020-09-30 21:08:57,580 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,583 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,584 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,584 : INFO : EPOCH - 37 : training on 58152 raw words (42691 effective words) took 0.1s, 851268 effective words/s
    2020-09-30 21:08:57,632 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,634 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,635 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,635 : INFO : EPOCH - 38 : training on 58152 raw words (42667 effective words) took 0.1s, 850589 effective words/s
    2020-09-30 21:08:57,685 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,686 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,687 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,687 : INFO : EPOCH - 39 : training on 58152 raw words (42641 effective words) took 0.1s, 843857 effective words/s
    2020-09-30 21:08:57,736 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-09-30 21:08:57,737 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-09-30 21:08:57,741 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-09-30 21:08:57,741 : INFO : EPOCH - 40 : training on 58152 raw words (42721 effective words) took 0.1s, 807691 effective words/s
    2020-09-30 21:08:57,741 : INFO : training on a 2326080 raw words (1708575 effective words) took 2.2s, 781245 effective words/s




Now, we can use the trained model to infer a vector for any piece of text
by passing a list of words to the ``model.infer_vector`` function. This
vector can then be compared with other vectors via cosine similarity.



.. code-block:: default

    vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    print(vector)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-0.08478509  0.05011684  0.0675064  -0.19926868 -0.1235586   0.01768214
     -0.12645927  0.01062329  0.06113973  0.35424358  0.01320948  0.07561274
     -0.01645093  0.0692549   0.08346193 -0.01599065  0.08287009 -0.0139379
     -0.17772709 -0.26271465  0.0442089  -0.04659882 -0.12873884  0.28799203
     -0.13040264  0.12478471 -0.14091878 -0.09698066 -0.07903259 -0.10124907
     -0.28239366  0.13270256  0.04445919 -0.24210942 -0.1907376  -0.07264525
     -0.14167067 -0.22816683 -0.00663796  0.23165748 -0.10436232 -0.01028251
     -0.04064698  0.08813146  0.01072008 -0.149789    0.05923386  0.16301566
      0.05815683  0.1258063 ]




Note that ``infer_vector()`` does *not* take a string, but rather a list of
string tokens, which should have already been tokenized the same way as the
``words`` property of original training document objects.

Also note that because the underlying training/inference algorithms are an
iterative approximation problem that makes use of internal randomization,
repeated inferences of the same text will return slightly different vectors.


Assessing the Model
-------------------

To assess our new model, we'll first infer new vectors for each document of
the training corpus, compare the inferred vectors with the training corpus,
and then returning the rank of the document based on self-similarity.
Basically, we're pretending as if the training corpus is some new unseen data
and then seeing how they compare with the trained model. The expectation is
that we've likely overfit our model (i.e., all of the ranks will be less than
2) and so we should be able to find similar documents very easily.
Additionally, we'll keep track of the second ranks for a comparison of less
similar documents.



.. code-block:: default

    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])








Let's count how each document ranks with respect to the training corpus

NB. Results vary between runs due to random seeding and very small corpus


.. code-block:: default

    import collections

    counter = collections.Counter(ranks)
    print(counter)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Counter({0: 292, 1: 8})




Basically, greater than 95% of the inferred documents are found to be most
similar to itself and about 5% of the time it is mistakenly most similar to
another document. Checking the inferred-vector against a
training-vector is a sort of 'sanity check' as to whether the model is
behaving in a usefully consistent manner, though not a real 'accuracy' value.

This is great and not entirely surprising. We can take a look at an example:



.. code-block:: default

    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Document (299): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (299, 0.9482713341712952): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (104, 0.8029672503471375): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (238, 0.2635717988014221): «centrelink is urging people affected by job cuts at regional pay tv operator austar and travel company traveland to seek information about their income support options traveland has announced it is shedding more than jobs around australia and austar is letting employees go centrelink finance information officer peter murray says those facing uncertain futures should head to centrelink in the next few days centrelink is the shopfront now for commonwealth services for income support and the employment network so that it is important if people haven been to us before they might get pleasant surprise at the range of services that we do offer to try and help them through situations where things might have changed for them mr murray said»

    LEAST (243, -0.13247375190258026): «four afghan factions have reached agreement on an interim cabinet during talks in germany the united nations says the administration which will take over from december will be headed by the royalist anti taliban commander hamed karzai it concludes more than week of negotiations outside bonn and is aimed at restoring peace and stability to the war ravaged country the year old former deputy foreign minister who is currently battling the taliban around the southern city of kandahar is an ally of the exiled afghan king mohammed zahir shah he will serve as chairman of an interim authority that will govern afghanistan for six month period before loya jirga or grand traditional assembly of elders in turn appoints an month transitional government meanwhile united states marines are now reported to have been deployed in eastern afghanistan where opposition forces are closing in on al qaeda soldiers reports from the area say there has been gun battle between the opposition and al qaeda close to the tora bora cave complex where osama bin laden is thought to be hiding in the south of the country american marines are taking part in patrols around the air base they have secured near kandahar but are unlikely to take part in any assault on the city however the chairman of the joint chiefs of staff general richard myers says they are prepared for anything they are prepared for engagements they re robust fighting force and they re absolutely ready to engage if that required he said»





Notice above that the most similar document (usually the same text) is has a
similarity score approaching 1.0. However, the similarity score for the
second-ranked documents should be significantly lower (assuming the documents
are in fact different) and the reasoning becomes obvious when we examine the
text itself.

We can run the next cell repeatedly to see a sampling other target-document
comparisons.



.. code-block:: default


    # Pick a random document from the corpus and infer a vector from the model
    import random
    doc_id = random.randint(0, len(train_corpus) - 1)

    # Compare and print the second-most-similar document
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Train Document (292): «rival afghan factions are deadlocked over the shape of future government the northern alliance has demanded day adjournment of power sharing talks in germany after its president burhanuddin rabbani objected to the appointment system for an interim administration president rabbani has objected to the plans for an interim government to be drawn up by appointment as discussed in bonn saying the interim leaders should be voted in by afghans themselves he also says there is no real need for sizeable international security force president rabbani says he would prefer local afghan factions drew up their own internal security forces of around personnel but if the world insisted there should be an international security presence there should be no more than or personnel in their security forces he says president rabbani objections are likely to cast doubt on his delegation ability to commit the northern alliance to any course of action decided upon in bonn he now threatens to undermine the very process he claims to support in the quest for stable government in afghanistan»

    Similar Document (13, 0.7867921590805054): «talks between afghan and british officials in kabul have ended without final agreement on the deployment of international security force the lack of suitable translation of the document meant further delay authorities in kabul have been giving conflicting signals for weeks now over the number of peacekeepers they would allow and the role the international force would play the foreign minister dr abdullah appeared to be ending the confusion saying an agreement was about to be signed there is already the agreement so it was finalised he said but spokesman for the interior minister yunis kanooni emerged soon after to say there was no agreement and nothing to sign scores of british peacekeepers are already patrolling the streets of kabul in tandem with afghan police but proposals to enlarge the force to as many as international peacekeepers have been criticised by some commanders as tantamount to foreign occupation»





Testing the Model
-----------------

Using the same approach above, we'll infer the vector for a randomly chosen
test document, and compare the document to our model by eye.



.. code-block:: default


    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Test Document (49): «labor needed to distinguish itself from the government on the issue of asylum seekers greens leader bob brown has said his senate colleague kerry nettle intends to move motion today on the first anniversary of the tampa crisis condemning the government over its refugee policy and calling for an end to mandatory detention we greens want to bring the government to book over its serial breach of international obligations as far as asylum seekers in this country are concerned senator brown said today»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (218, 0.8016394376754761): «refugee support groups are strongly critical of federal government claims that the pacific solution program is working well the immigration minister philip ruddock says he is pleased with the program which uses pacific island nations to process asylum seekers wanting to come to australia president of the hazara ethnic society of australia hassan ghulam says the australian government is bullying smaller nations into accepting asylum seekers if the pacific countries wanted refugees they can clearly raise their voice in the united nations and say yes we are accepting refugees and why australia who gives this authority to the australian government to force the pacific countries to accept refugees in this form or in the other form he asked»

    MEDIAN (204, 0.3319269120693207): «an iraqi doctor being held at sydney villawood detention centre claims he was prevented from receiving human rights award dr aamer sultan had been awarded special commendation at yesterday human rights and equal opportunity commission awards in sydney but was not able to receive the honour in person dr sultan says he had been hoping to attend the ceremony but says the management at villawood stopped him from going submitted formal request to the centre manager who promised me that he will present the matter to migration management here who are the main authority here they also came back that unfortunately we can not fulfill this request for you but they didn give any explanation dr sultan says he was disappointed by the decision the immigration minister philip ruddock has written letter of complaint to the medical journal of australia about an article penned by dr sultan on the psychological state of detainees at villawood the journal has published research dr sultan conducted with former visiting psychologist to the centre kevin sullivan their survey of detainees over nine months found all but one displayed symptoms of psychological distress at some time the article says per cent acknowledged chronic depressive symptoms and close to half of the group had reached severe stages of depression»

    LEAST (157, -0.10524928569793701): «british man has been found guilty by unanimous verdict of the kidnap and murder of an eight year old schoolgirl whose death in july shocked britain and set off rampage of anti paedophile vigilantes roy whiting was sentenced to life imprisonment for the abduction and murder of eight year old sarah payne with recommendation by trial judge justice richard curtis that he never be released you are indeed an evil man you are in no way mentally unwell have seen you for month and in my view you are glib and cunning liar justice curtis said there were cheers of delight as the verdicts were read out by the foreman at lewes crown court the jury of nine men and three women had been deliberating for nine hours as soon as the verdicts were declared the court heard details of whiting previous conviction for the kidnap and indecent assault of nine year old girl in prosecutor timothy langdale told the jury how the defendant threw the child into the back of his dirty red ford sierra and locked the doors he had driven her somewhere she didn know where when she asked where they were going he said shut up because he had knife mr langdale said the defendant told the girl to take off her clothes when she refused he produced rope from his pocket and threatened to tie her up what he actually threatened was that he would tie her mouth up she took her clothes off as he had ordered her to do mr langdale then gave graphic details of the abuse to which whiting subjected the terrified child whiting was given four year jail sentence in june after admitting carrying out the attack in march that year but he was released in november despite warnings from probation officers who were convinced there was danger he would attack another child they set out their warnings in pre sentence report prepared after the first assault and in the parole report before he was released from prison he was kept under supervision for four months after his release but was not being monitored by july last year when eight year old sarah was abducted and killed whiting has been arrested three times in connection with the case but the first and second times was released without being charged sarah disappeared on july last year prompting massive police search her partially buried naked body was found days later in field and police believe she was strangled or suffocated»





Conclusion
----------

Let's review what we've seen in this tutorial:

0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
4. Assess the model
5. Test the model on the test corpus

That's it! Doc2Vec is a great way to explore relationships between documents.

Additional Resources
--------------------

If you'd like to know more about the subject matter of this tutorial, check out the links below.

* `Word2Vec Paper <https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_
* `Doc2Vec Paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_
* `Dr. Michael D. Lee's Website <http://faculty.sites.uci.edu/mdlee>`_
* `Lee Corpus <http://faculty.sites.uci.edu/mdlee/similarity-data/>`__
* `IMDB Doc2Vec Tutorial <doc2vec-IMDB.ipynb>`_



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  7.863 seconds)

**Estimated memory usage:**  37 MB


.. _sphx_glr_download_auto_examples_tutorials_run_doc2vec_lee.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_doc2vec_lee.py <run_doc2vec_lee.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_doc2vec_lee.ipynb <run_doc2vec_lee.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
