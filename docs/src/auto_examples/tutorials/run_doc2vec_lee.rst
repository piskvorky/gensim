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

Le and Mikolov in 2014 introduced the `Doc2Vec algorithm <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__, which usually outperforms such simple-averaging of ``Word2Vec`` vectors.

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

    [-0.0014455  -0.03838259  0.03199863  0.01624313  0.04130909  0.20024535
     -0.09749083  0.00597675 -0.0498415  -0.04540551  0.01723257 -0.20151177
      0.08523481 -0.08950453  0.00380471  0.10519169 -0.11385646 -0.12259311
      0.05124485  0.13983724  0.12103602 -0.2321382  -0.07852937 -0.24980102
      0.08878644 -0.1038101   0.22263823 -0.21950239 -0.31584352  0.11648487
      0.18644053 -0.08014616 -0.11723718 -0.22560167 -0.04025911  0.05705469
      0.20113727  0.12674493  0.07401953 -0.01472244  0.13031979 -0.19944443
      0.16314563 -0.05472009  0.01138415  0.09830751 -0.11751664  0.00259685
      0.11373404  0.03917272]



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

    MOST (299, 0.9392883777618408): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (104, 0.8041712045669556): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (171, 0.268340528011322): «drug education campaigns appear to be paying dividends with new figures showing per cent drop in drug related deaths last year according to the australian bureau of statistics people died from drug related causes in the year that figure is substantial drop from when australians died of drug related causes across the states and territories new south wales recorded the biggest decrease the bureau david payne attributes the decline of drug deaths to the heroin drought in some parts of the country better equipped ambulances and emergency wards and above all effective federal and state drug education campaigns they have put lot of money into the program there has been fall and while you can discern trend from that the figures are going in the right way right direction mr payne said»

    LEAST (223, -0.05577106401324272): «indonesian troop re enforcements have started arriving in central sulawesi as the government attempts to end days of deadly clashes between christians and muslims violence in the last week has claimed at least eight lives and left thousands of people homeless more than police and soldiers are being sent in to disarm rival groups and restore calm there have been no new reports of violence but residents in the christian town of ten tena say they fear further attacks by muslim militiamen taking up positions in the hills around the town in region where fighting between muslims and christians has claimed hundreds of lives in the last two years many blame the latest upsurge in violence on the arrival of members of the laskar jihad muslim militia from training camps in java and from the neighbouring maluka islands»




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

    Train Document (272): «the storm clean up in sydney will resume in earnest this morning as fresh crews are brought in to replace state emergency service ses personnel who worked through the night the storm hit sydney early yesterday afternoon and two schoolgirls died when tree fell on them at reserve at hornsby heights in the city north number of other people were injured as the storm brought down trees and power poles and lifted roofs new south wales emergency services minister bob debus says welfare and emergency funding arrangements have been put in place with the declaration of natural disaster areas in campbeltown hornsby warringah and kurringai welfare services become available if they are needed local government is refunded any money it spends on the clean up or that it spends on repairing its own infrastructure low interest loans if they are needed are available to small business to help them get back on their feet again mr debus said energy australia says power has been restored to customers and work will continue today to reconnect those still without electricity energy australia peter leete says work will concentrate around the worst hit areas the worst of the problems we have still got are in sydney northern suburbs which seem to be the worst hit of all and that around hornsby st ives turramurra and frenches forest mr leete said four hundred ses volunteers are responding to more than calls for assistance the volunteers have worked throughout the night to remove trees from homes and roads the ses laura goodin says it will take several days before the damage is cleared up while the ses has received fewer calls for help than in the storm two weeks ago many of the jobs in this storm are actually quite complicated involving large trees or extensively damaged homes and businesses we re estimating that most of the tasks will be completed by friday if no new storms develop ms goodin said outside sydney the storms caused damage in north east of the state and the lower hunter scores of homes and farm buildings have been damaged and literally hundreds of trees have been brought down the storms accompanied by gale force winds and hail left large areas around tamworth gunnedah and quirindi without electricity and telephone services»

    Similar Document (40, 0.8816476464271545): «firefighters across new south wales are gearing up for wind change that may bring further property losses today more than fires now ring two thirds of the greater sydney area the blazes stretch south of the royal national park and north of wollongong all the way to the blue mountains and up towards the edge of the baulkham hills shire fires are also burning around huskisson on the far south coast and as far inland as mudgee narromine and kempsey and the richmond valley in the north however the major areas of concern today are the southern sydney suburbs of heathcote and engadine thousands of residents in those suburbs were evacuated overnight senior forecaster with the sydney weather bureau ian robertson says the greatest risk will come when winds change direction this afternoon we re looking at another dry day ahead throughout the state particularly along the coast more average sort of temperatures but the trick will be the winds mr robertson said we re looking at south west winds this morning an east to south east sea breeze along the coast which is going to make things quite challenging for firefighting between and firefighters are currently battling the blazes crews have already been brought in from victoria but the rural fire service says it expects to call on other states for help service spokesman john winter says property losses have been high we are estimating that around homes have been lost obviously there are areas we re yet to confirm property losses mr winter said»




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

    Test Document (45): «batasuna political party that campaigns for an independent basque state faces double blow today the spanish parliament is expected to vote overwhelmingly in favour of banning the radical group while senior investigative judge is poised to suspend batasuna activities on the grounds that they benefit eta the outlawed basque separatist group»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (76, 0.7851859331130981): «the death toll in argentina food riots has risen to local media reports say four more people died this morning in clashes between police and protesters near the presidential palace in the capital buenos aires president fernando de la rua has called on the opposition to take part in government of national unity and apparently will resign if it does not looting and rioting has generally given way to more peaceful demonstrations against the faltering government blamed for month recession heavily armed police using powers under day state of siege decree are attempting to prevent large public gatherings but union leaders say workers and the unemployed will not stop until the government is removed and living standards restored with argentina discredited economy minister now gone the government hopes to approve new budget acceptable to the international monetary fund imf to avoid default on the billion foreign debt the presidents of neighbouring brazil and chile say they fear the social unrest could infect their own nations unless argentina and its leaders can resolve the crisis quickly»

    MEDIAN (142, 0.42160844802856445): «the defence minister robert hill says the australian government is still trying to interview suspected taliban fighter david hicks senator hill says the government does not know much more than what is on the public record about the year old background he says he was not aware he had applied to join the australian defence force or that australian authorities have known about him for some time senator hill has told channel seven the government does not know what motivated the man to fight alongside taliban forces in rare circumstances this does happen as we know there one american who has been captured after fighting for the taliban occasionally people decide to exercise violent option in pursuing particular political or religious belief and think you ought to probably address the questions to the psychologists or the psychiatrists senator hill said»

    LEAST (6, -0.0833040177822113): «the united states team of monica seles and jan michael gambill scored decisive victory over unseeded france in their first hopman cup match at burswood dome in perth the pair runners up in the million dollar mixed teams event last year both won their singles encounters to give the us an unbeatable lead the year old seles currently ranked eighth recovered from shaky start to overpower virginie razzano who is ranked nd seles had to fight hard to get home in straight sets winning in minutes then the year old gambill ranked st wore down determined arnaud clement th to win in minutes the americans are aiming to go one better than last year when they were beaten by swiss pair martina hingis and roger federer in the final of the eight nation contest gambill said the win was great way to start the tennis year got little tentative at the end but it was great start to my year he said arnaud is great scrapper and am delighted to beat him even though am frankly bit out of shape that is one of the reasons am here will be in shape by the end of the tournament just aim to keep improving in the new year and if do think have chance to beat anyone when am playing well gambill was pressed hard by clement before taking the first set in minutes but the american gained the ascendancy in the second set breaking in the third and fifth games seles said she had expected her clash with razzano to be tough she was top junior player in the world so it was no surprise that she fought so well she said seles said she still had the hunger to strive to regain her position at the top of her sport this is why you play she said but want to try not to peak too early this season seles slow into her stride slipped to in her opening set against razzano but recovered quickly claiming the set after snatching four games in row in the second set seles broke her opponent in the opening game and completed victory with relative ease despite razzano tenacious efforts»




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

   **Total running time of the script:** ( 0 minutes  9.354 seconds)

**Estimated memory usage:**  8 MB


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

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
