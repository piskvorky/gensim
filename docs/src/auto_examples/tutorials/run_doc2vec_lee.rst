.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_doc2vec_lee.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tutorials_run_doc2vec_lee.py:


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

    [-0.08798727 -0.13129461  0.15840417  0.17654417 -0.11176015 -0.0564478
     -0.25220573  0.08728343 -0.05107945 -0.01988564  0.17926392  0.01253001
     -0.15548548 -0.08958613  0.13900393  0.00482653  0.09774715  0.05284704
      0.01435522 -0.06572937  0.24613744 -0.13968733  0.21480554 -0.08533692
      0.15711235  0.03944711  0.05400683 -0.01113842 -0.06421786  0.03654933
     -0.01246722  0.02566761 -0.03354044 -0.17895083  0.08384037 -0.02384334
     -0.07023416 -0.27627805 -0.01872351 -0.14052777  0.09896303 -0.22885887
     -0.16809027 -0.22808242 -0.05857427  0.1042163  -0.00800784 -0.0564643
     -0.10638058  0.24180691]


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

    MOST (299, 0.946539044380188): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (104, 0.8230295181274414): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (121, 0.25729361176490784): «the latest business expectations survey is raising hopes of solid economic start to the new year the dun and bradstreet survey has identified positive projections for sales profits investment and employment of the organisations questioned by dun bradstreet during november per cent were looking for an increase in sales in the first quarter of the new year twenty per cent were planning to take on extra staff compared with per cent who said they had been shedding labour profit projections are well up while investment intentions have improved slightly the survey has also found companies are finding it significantly easier to comply with the goods and services tax gst»

    LEAST (243, -0.10224725306034088): «four afghan factions have reached agreement on an interim cabinet during talks in germany the united nations says the administration which will take over from december will be headed by the royalist anti taliban commander hamed karzai it concludes more than week of negotiations outside bonn and is aimed at restoring peace and stability to the war ravaged country the year old former deputy foreign minister who is currently battling the taliban around the southern city of kandahar is an ally of the exiled afghan king mohammed zahir shah he will serve as chairman of an interim authority that will govern afghanistan for six month period before loya jirga or grand traditional assembly of elders in turn appoints an month transitional government meanwhile united states marines are now reported to have been deployed in eastern afghanistan where opposition forces are closing in on al qaeda soldiers reports from the area say there has been gun battle between the opposition and al qaeda close to the tora bora cave complex where osama bin laden is thought to be hiding in the south of the country american marines are taking part in patrols around the air base they have secured near kandahar but are unlikely to take part in any assault on the city however the chairman of the joint chiefs of staff general richard myers says they are prepared for anything they are prepared for engagements they re robust fighting force and they re absolutely ready to engage if that required he said»


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

    Train Document (143): «kashmiri militant groups denied involvement in thursday attack on the indian parliament accusing indian intelligence instead we want to make it clear that kashmiris have no connection with this attack said the muttahida jihad council mjc an alliance of groups fighting indian rule in kashmir we believe it was carried out by indian intelligence agencies to achieve their motives about the kashmir issue the groups added in statement the attack on the parliament building in new delhi left at least dead the indian authorities have not said who they believe was behind the killings but the kashmiri groups accused the indian government of masterminding the attack in bid to divert attention from what they called increasing international pressure over kashmir»

    Similar Document (1, 0.717074453830719): «indian security forces have shot dead eight suspected militants in night long encounter in southern kashmir the shootout took place at dora village some kilometers south of the kashmiri summer capital srinagar the deaths came as pakistani police arrested more than two dozen militants from extremist groups accused of staging an attack on india parliament india has accused pakistan based lashkar taiba and jaish mohammad of carrying out the attack on december at the behest of pakistani military intelligence military tensions have soared since the raid with both sides massing troops along their border and trading tit for tat diplomatic sanctions yesterday pakistan announced it had arrested lashkar taiba chief hafiz mohammed saeed police in karachi say it is likely more raids will be launched against the two groups as well as other militant organisations accused of targetting india military tensions between india and pakistan have escalated to level not seen since their war»


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

    Test Document (9): «british air raid in southern iraq left eight civilians dead and nine wounded the iraqi military said sunday the military told the official iraqi news agency that the warplanes bombed areas in basra province miles south of baghdad the central command in florida said coalition aircraft used precision guided weapons to strike two air defense radar systems near basra in response to recent iraqi hostile acts against coalition aircraft monitoring the southern no fly zone»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (223, 0.8073922395706177): «indonesian troop re enforcements have started arriving in central sulawesi as the government attempts to end days of deadly clashes between christians and muslims violence in the last week has claimed at least eight lives and left thousands of people homeless more than police and soldiers are being sent in to disarm rival groups and restore calm there have been no new reports of violence but residents in the christian town of ten tena say they fear further attacks by muslim militiamen taking up positions in the hills around the town in region where fighting between muslims and christians has claimed hundreds of lives in the last two years many blame the latest upsurge in violence on the arrival of members of the laskar jihad muslim militia from training camps in java and from the neighbouring maluka islands»

    MEDIAN (60, 0.25128552317619324): «israel has rejected palestinian leader yasser arafat bid to make his annual visit to bethlehem for christmas eve during an security cabinet meeting early today the security cabinet made its decision based on the fact that arafat is not working to dismantle terror organizations and to foil terror attacks against israel and to arrest and punish terrorists including the murderers of tourism minister rhavam zeevi statement from prime minister ariel sharon office said earlier mr arafat declared he would walk to bethlehem for christmas eve mass if he has to if israeli authorities refused him access to the biblical town mr arafat statement comes as palestinians in gaza buried six teenagers killed in the worst internal palestinian violence in seven years the funerals in gaza were peaceful with palestinian police staying away and mourners agreeing no weapons were to be carried this has been difficult week for yasser arafat he put his reputation on the line by ordering the arrest of some key palestinian militants in the most important radical group hamas then declared an end to its campaign of suicide bombings and other attacks against israel another smaller but important radical group islamic jihad might follow hamas lead but the key to all this is israel reaction if it eases its blockade of towns in the west bank then yasser arafat will have something to show for his efforts»

    LEAST (17, -0.2976718246936798): «spain has begun its hopman cup campaign in perth with victory over argentina arantxa sanchez vicario and tommy robredoboth won their singles matches and then teamed to win the mixed doubles sanchez vicario says she is hoping to win her second hopman cup title after winning the tournament with her brother emilio in it would be very nice to start the year off and as say it always tough but it very good start for me and looking forward with tommy to see if we can be the champions again she said today the united states will play france meanwhile world number one lleyton hewitt says he will not be putting pressure on himself to win next month australian tennis open in melbourne hewitt yesterday teamed with fellow australian alicia molik to beat switzerland in their opening tie at the hopman cup in perth hewitt says his first objective will be to reach the second week of the grand slam event think if play my best tennis and give per cent no matter who play think in with good chance of getting through to the second week and if that happens then most times in grand slam it sort of anyone tournament from there he said»


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

   **Total running time of the script:** ( 0 minutes  7.197 seconds)

**Estimated memory usage:**  16 MB


.. _sphx_glr_download_auto_examples_tutorials_run_doc2vec_lee.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_doc2vec_lee.py <run_doc2vec_lee.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_doc2vec_lee.ipynb <run_doc2vec_lee.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
