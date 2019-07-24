.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_doc2vec_lee.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_doc2vec_lee.py:


.. _doc2vec_lee_py:

Doc2Vec Model
=============

Introduces Gensim's Doc2Vec model and demonstrates its use on the Lee Corpus.

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
:ref:`core_concepts_py` section.
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

Le and Mikolov in 2014 introduced the *Paragraph Vector*, which usually outperforms such simple-averaging of ``Word2Vec`` vectors.

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
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    lee_test_file = test_data_dir + os.sep + 'lee.cor'







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
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

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

Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and
iterating over the training corpus 40 times. We set the minimum word count to
2 in order to discard words with very few occurrences. (Without a variety of
representative examples, retaining such infrequent words can often make a
model worse!) Typical iteration counts in published 'Paragraph Vectors'
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







Essentially, the vocabulary is a dictionary (accessible via
``model.wv.vocab``\ ) of all of the unique words extracted from the training
corpus along with the count (e.g., ``model.wv.vocab['penalty'].count`` for
counts for the word ``penalty``\ ).


Next, train the model on the corpus.
If the BLAS library is being used, this should take no more than 3 seconds.
If the BLAS library is not being used, this should take no more than 2
minutes, so use BLAS if you value your time.



.. code-block:: default

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)







Now, we can use the trained model to infer a vector for any piece of text
by passing a list of words to the ``model.infer_vector`` function. This
vector can then be compared with other vectors via cosine similarity.



.. code-block:: default

    vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    print(vector)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.17541483 -0.49196827  0.08779179 -0.00909387  0.06893727  0.12022585
     -0.0408328  -0.09224472 -0.18015207  0.12408012  0.02713208 -0.04907346
      0.18109988  0.04060466 -0.16465215 -0.07176629  0.04047617  0.09858649
     -0.05062681  0.20516384 -0.12163664  0.00814463  0.03010288  0.05426239
     -0.02301049 -0.01393805  0.01991753  0.16094668 -0.19610573 -0.1707981
      0.08435371 -0.03093486  0.16544315  0.05434167  0.13624124 -0.10247199
      0.09367518  0.230475   -0.06811217 -0.10978779 -0.16073455  0.08391835
      0.0385895   0.08876099 -0.10157748 -0.17650042  0.17826441 -0.00666687
      0.03084892  0.10378202]


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
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
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

    Counter({0: 291, 1: 9})


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

    MOST (299, 0.9471086263656616): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (112, 0.8182107210159302): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (151, 0.275578111410141): «senior construction forestry mining and energy union cfmeu officials giving evidence at the royal commission into the building industry have been overwhelmed by support from union members about construction workers have walked off the job for the third day to demonstrate outside the commission venue mounted police escorted the protesters from melbourne city square to collins place morning traffic ground to halt at the intersection of russell and collins streets when the crowd stopped to chant union slogans cfmeu victorian secretary martin kingham says he has been astounded by the strong support shown by union members on each day of the hearings he maintains the union has been treated unfairly as it faces allegations of intimidation and using standover tactics on work sites mr kingham is currently giving evidence before the commission it is the last day of hearings before the christmas break the labor leader simon crean says senior labor figures bob hawke and neville wran will be used to help modernise the party labor national executive is meeting in canberra mr crean will put his views on the changes labor needs to make the executive is expected to ask mr hawke and mr wran to oversee the process mr crean says they know what needs to be done bob hawke and neville wran understood the importance of modernising the party and that why they were successful leaders of the country sure we don need to teach them to suck eggs what want them to do is to give us guidance as to how we can bring the new approach to labor in to enable us to properly present and gain the confidence of the majority of the australian people the opposition leader said»

    LEAST (233, -0.10959328711032867): «three us troops and five members of the afghan opposition were killed by stray us bomb near kandahar in afghanistan the pentagon said the pentagon had earlier confirmed that two us special forces soldiers were killed and others wounded north of kandahar when bomber dropped pound bomb too close to them the was flying in support of opposition forces north of kandahar said pentagon spokeswoman victori clark we have an update since this morning and unfortunately the number of us forces killed is now three rival afghan factions signed an historic power sharing agreement to form post taliban government and set the country on the road to recovery and democracy after two decades of war the accord was sealed after nine days of exhausting negotiations and paves the way for six month interim administration headed by moderate muslim hamid karzai from the dominant pashtun ethnic group the deal gives the northern alliance control of three key portfolios in the member cabinet which includes two women and is due to be up and running by december it also gives symbolic role to the former king and provides for un security force for kabul the agreement was signed in the german city of bonn by the leaders of the four delegations and un special envoy for afghanistan lakhdar brahimi to applause from an audience which included german chancellor gerhard schroeder we were the champions of resistance and will be proud to be the champions of peace said yunus qanooni the northern alliance chief negotiator and the interim government interior minister delegate from the so called peshawar group sayed hamed gailani summed up the atmosphere in single phrase there are two things evident today yesterday rain does not have the courage to cry and the sun cannot hide its smile he said the appointment of karzai year old tribal pashtun tribal leader currently fighting the taliban near their last stronghold of kandahar was seen as an attempt to balance afghanistan delicate ethnic mix it cements whirlwind transformation in afghanistan fate since the september attacks on new york and washington the trigger for massive us air strikes that have dislodged the taliban militia from most of the country and put the northern alliance back on top showing the strain from nine days of frantic diplomacy brahimi recognised the accord was far from perfect and that its signatories were not fully representative of the afghan people»


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

    Train Document (110): «the radical palestinian group hamas has reportedly shifted the focus in its guerrilla war against israel senior israeli defence official has told israel army radio the palestinian organisation is now planning to attack strategic targets hamas has carried out numerous suicide bombings in israel but its targets have tended to be what are known as soft ones such as public buses or crowded shopping areas but now the israel official says hamas focus will be on attacking strategic buildings and senior israeli officials and he admits that stopping the group will be next to impossible hamas has rejected call by palestinian leader yasser arafat for an end to all military operations against israel and return to peace negotiations»

    Similar Document (93, 0.8593460917472839): «senior hamas official has said the radical palestinian movement has decided to stop suicide bombings against israel he told the agence france presse news agency that hamas had taken an internal decision to end what it calls martyrdom operations but it was not going to make an official declaration the official refused to say what tactics the radical group would use in its avowed struggle to end the israeli occupation of palestinian land or if it would revert to grisly attacks if israel carries out more targeted killings which have taken heavy toll on hamas operatives we can predict anything the official said an official from palestinian leader yasser arafat fatah movement also said there had been meeting of various palestinian factions in the west bank on wednesday and that hamas had informed them they were halting their operations the suicide attacks triggered massive israeli armed response and sweeping international condemnation the fatah official who likewise asked to remain anonymous said hamas had said it did not want to damage palestinian national unity by carrying on with such operations hamas suicide bombers blew themselves up in jerusalem and haifa in northern israel at the beginning of this month killing people and provoking the heaviest israeli air raids against mr arafat administration to date israel prime minister ariel sharon severed all ties with mr arafat last week after another hamas attack which used gunmen and roadside explosives rather than suicide bombings mr arafat in turn ordered the closure of dozens of hamas offices and has started arresting its militants senior israeli defence official was quoted by army radio on tuesday as saying hamas had shifted the focus in its guerrilla war against israel and now planned to attack strategic targets rather than soft ones such as public buses or crowded shopping areas the defence source said the focus will be on attacking strategic buildings and senior israeli officials»


Testing the Model
-----------------

Using the same approach above, we'll infer the vector for a randomly chosen
test document, and compare the document to our model by eye.



.. code-block:: default


    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Test Document (35): «the spectre of osama bin laden rose again today urging afghans to launch new jihad or holy war and predicting the fall of the united states in hand written letter posted on an islamic website there was no hard proof that the scruffy missive was genuine but islamonline net said it had been received by their correspondent in jalalabad eastern afghanistan from an afghan source who asked to remain anonymous the source claimed it was the most recent letter from the world most wanted man»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (143, 0.8098287582397461): «kashmiri militant groups denied involvement in thursday attack on the indian parliament accusing indian intelligence instead we want to make it clear that kashmiris have no connection with this attack said the muttahida jihad council mjc an alliance of groups fighting indian rule in kashmir we believe it was carried out by indian intelligence agencies to achieve their motives about the kashmir issue the groups added in statement the attack on the parliament building in new delhi left at least dead the indian authorities have not said who they believe was behind the killings but the kashmiri groups accused the indian government of masterminding the attack in bid to divert attention from what they called increasing international pressure over kashmir»

    MEDIAN (2, 0.1597033143043518): «the national road toll for the christmas new year holiday period stands at eight fewer than for the same time last year people have died on new south wales roads with eight fatalities in both queensland and victoria western australia the northern territory and south australia have each recorded three deaths while the act and tasmania remain fatality free»

    LEAST (203, -0.3237740099430084): «qantas management and unions representing the airline maintenance workers will meet again today after marathon talks last night failed to resolve wage dispute unions are fighting proposed to month wage freeze and to secure better career structure for their employees bill shorten of the australian workers union awu says the unions will not rest until satisfactory outcome is reached after eight hours the awu and amwu are still talking to qantas we will resume tomorrow morning at am aedt in the industrial relations commission to see if we can work through this position our members now find ourselves in he said last night meanwhile ansett workers will sing christmas carols in front of the prime minister kirribilli residence in sydney this morning to remind john howard about their owed entitlements the transport workers union twu will hold hour picket outside kirribilli house and says about workers and their families are facing bleak christmas the news south wales secretary of the twu tony sheldon says the government promised to deliver about million in entitlements mr sheldon says the financial situation for many workers has reached crisis point there been very little delivered by this government lot of promises lot of noise was made before the federal election but very little in substance it important john howard delivers for the tourism community for the ansett workers and for the australian community generally mr sheldon said»


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

   **Total running time of the script:** ( 0 minutes  12.068 seconds)

**Estimated memory usage:**  17 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_doc2vec_lee.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/010_tutorials/run_doc2vec_lee.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_doc2vec_lee.py <run_doc2vec_lee.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_doc2vec_lee.ipynb <run_doc2vec_lee.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
