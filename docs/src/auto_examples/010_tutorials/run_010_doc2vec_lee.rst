.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_010_doc2vec_lee.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_010_doc2vec_lee.py:


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

    [ 0.22109026  0.12945214 -0.04987111 -0.06666645 -0.03005451  0.07812942
     -0.0468159   0.05266683 -0.02142708  0.11049211 -0.26119208 -0.04469399
      0.11371834 -0.2370929  -0.07897254 -0.05506096 -0.03876496 -0.09452289
     -0.00322254  0.0846746   0.09233897 -0.07097694 -0.16656673  0.01237435
      0.00558276  0.04805247 -0.32035804 -0.08328743 -0.03138952  0.22360589
      0.14839473  0.02690092  0.04741399 -0.24609153  0.06968495  0.29817858
     -0.10294496  0.06082038 -0.08411389 -0.14609648 -0.17304227  0.1022751
      0.18142411  0.03966308  0.09084661 -0.03894547  0.08898773 -0.12771595
      0.04623495  0.09869438]


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

    MOST (299, 0.9480094313621521): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (112, 0.8142329454421997): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (236, 0.2697334885597229): «two swiss guides who survived the interlaken canyoning accident in which australians died have denied any wrongdoing in leading the trip into the saxet river court in switzerland has been told the huge wall of water which swept down the river was an unpredictable natural catastrophe both simon wiget and stefan abegglen told the court of their sorrow about the accident but denied making any mistakes nor did they think weather conditions were inappropriate asked if they had received adequate training in weather analysis from their adventure world supervisors they said they had but when asked what specific training they received they were unable to give detail they also acknowledged their salaries as guides were dependent on the number of trips they took and the number of people who accompanied them for some of the australians the day ended with an emotional visit to the river where their children died»

    LEAST (261, -0.12024208903312683): «afghan opposition leaders meeting in germany have reached an agreement after seven days of talks on the structure of an interim post taliban government for afghanistan the agreement calls for the immediate assembly of temporary group of multi national peacekeepers in kabul and possibly other areas the four afghan factions have approved plan for member ruling council composed of chairman five deputy chairmen and other members the council would govern afghanistan for six months at which time traditional afghan assembly called loya jirga would be convened to decide on more permanent structure the agreement calls for elections within two years»


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

    Train Document (108): «the international monetary fund imf has described economic conditions in australia as reasonably robust in the release of its latest world economic outlook the imf says prospects globally remain subject to considerable uncertainty but number of factors should help support recovery in including the stimulus still in the pipeline from policy decisions by central banks and governments around the world sharp weakening in oil prices should also contribute the imf says growth in australia is expected to reach per cent in stronger than this year but half percentage point lower than projected in its october outlook australia exports while sustained so far by weak currency could come under pressure if global growth and commodity prices remain weak»

    Similar Document (136, 0.7728472352027893): «new report suggests the costs of an aging australian population have been exaggerated the report issued by the australia institute says detailed examination of population and health data shows an aging population will not create an unsustainable burden on shrinking workforce far from being an economic and social burden it found the majority of older people enjoyed healthy and independent lives many making financial contributions to their families and participating in voluntary community activities the paper challenges the assumption an older population will see health costs rise to unsustainable levels it says rising health costs are caused mainly by factors other than aging such as the growth of medical technology rising consumer demand and escalating prices»


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

    Test Document (22): «the biowarfare expert under scrutiny in the anthrax attacks declared am not the anthrax killer and lashed out today against attorney general john ashcroft for calling him person of interest in the investigation for the second time in two weeks the scientist went before throng of reporters outside his lawyer office to profess his innocence and decry the attention from law enforcers that he contends has destroyed his life»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (23, 0.7439348697662354): «americans fears about airplane security continue to increase after man made it through two separate flights with loaded gun in his carry on luggage the man was finally stopped before boarding third plane in memphis the man had travelled from florida to atlanta and then atlanta to memphis he was attempting to board his return flight last night when he was stopped by security personnel for random check they discovered loaded mm beretta semi automatic pistol in his hand luggage the man acknowledged the gun was his and was released on bail there is no suggestion he was planning any sort of terrorist attack but his ability to complete two flights while carrying the weapon has again highlighted airline security problems the incident follows last week drama when man was able to board plane from paris to miami with explosives in his shoes»

    MEDIAN (273, 0.3715556263923645): «the royal commission into the collapse of insurance giant hih will resume in sydney this morning on the first day of public hearings yesterday royal commissioner justice neville owen warned all parties not to break confidentiality agreements following the possible leak of report to the media last month he told the commission one of the single biggest factors in the company collapse was that it did not plan for future claims and did not have prudential margins over the estimate of future liabilities he says all parties involved in the crash will have chance to put their views of course fundamental principles of fairness require that before that stage of the inquiry is reached any person whose interest might be adversely affected by finding of fact or recommendation must be made clearly and unequivocally aware of that risk and be given ample opportunity to present any evidence and submissions relevant to that issue he said»

    LEAST (8, -0.07617580890655518): «there has been welcome relief for firefighters in new south wales overnight with milder weather allowing them to strengthen containment lines around the most severe fires but fire authorities are not getting overly optimistic as dry and hot weather is forecast to continue the weather bureau is forecasting temperatures in the high and westerly winds until at least friday which means fire authorities are reluctant to get too excited about last night favourable conditions marks sullivan from the rural fire service says fire fighters are remaining on guard lot of fires that have been burning in the areas around sydney and the north coast and further south have been burning within areas that are known and are contained he said however that not to say that these fires won pose threat given the weather conditions that are coming up over the next few days despite the caution the rural fire service says most of the state fires that threaten property are burning within containment lines greater sydney is ringed by fires to the north west and south two of those flared overnight one at appin in the southern highlands was quickly brought under control another flare up at spencer north of the city is not contained on its north western flank but is not threatening property in the lower blue mountains west of sydney firefighters have spent the night setting up kilometre containment line to protect communities along the great western highway from glenbrook to bulaburra two fires burning near cessnock west of newcastle are still within containment lines in the state north aircraft will this morning check if lightning from large electrical storm overnight has sparked any new fires above grafton aircraft have also been used in the shoalhaven area in the state south to drop incendiary devices that start fire control lines in inaccessible areas the rural fire service commissioner phil koperberg says if fire activity increases hundreds of new year eve fireworks celebrations in new south wales will be cancelled»


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

   **Total running time of the script:** ( 0 minutes  7.590 seconds)

**Estimated memory usage:**  46 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_010_doc2vec_lee.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_010_doc2vec_lee.py <run_010_doc2vec_lee.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_010_doc2vec_lee.ipynb <run_010_doc2vec_lee.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
