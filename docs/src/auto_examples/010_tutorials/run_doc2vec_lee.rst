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







Doc2Vec is an NLP tool for representing documents as a vector and is a
generalizing of the Word2Vec method.  This tutorial will serve as an
introduction to Doc2Vec and present ways to train and assess a Doc2Vec model.

This tutorial will take you through the following steps:

1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
4. Assess the model
5. Test the model on the test corpus

Getting Started
---------------

To get going, we'll need to have a set of documents to train our doc2vec
model. In theory, a document could be anything from a short 140 character
tweet, a single paragraph (i.e., journal article abstract), a news article,
or a book. In NLP parlance a collection or set of documents is often referred
to as a **corpus**. 

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

Below, we define a function to open the train/test file (with latin
encoding), read the file line-by-line, pre-process each line using a simple
gensim pre-processing tool (i.e., tokenize text into individual words, remove
punctuation, set to lowercase, etc), and return a list of words. Note that,
for a given file (aka corpus), each continuous line constitutes a single
document and the length of each line (i.e., document) can vary. Also, to
train the model, we'll need to associate a tag/number with each document of
the training corpus. In our case, the tag is simply the zero-based line
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

    [-0.18575795 -0.13511986 -0.076375   -0.09754043  0.1818997  -0.0625411
      0.11482348 -0.26765174 -0.00440135 -0.12531066 -0.066892    0.09797165
     -0.2682665  -0.16709955  0.09132338  0.14093246  0.361435    0.1316588
     -0.04919421 -0.06542347 -0.10133749  0.00798248  0.03723055 -0.22058137
     -0.3045081   0.10888425 -0.12888427 -0.12382892  0.0957042   0.19151324
      0.04488073 -0.05343053 -0.43255424  0.05910409 -0.06140405 -0.15290727
     -0.0347239   0.0077343   0.17252333 -0.10788079  0.06345859 -0.04193899
     -0.06547964 -0.05680009 -0.06289701 -0.07419236 -0.08074679  0.06059996
     -0.09367087  0.02488928]


Note that ``infer_vector()`` does *not* take a string, but rather a list of
string tokens, which should have already been tokenized the same way as the
``words`` property of original training document objects. 

Also note that because the underlying training/inference algorithms are an
iterative approximation problem that makes use of internal randomization,
repeated inferences of the same text will return slightly different vectors.


Assessing Model
---------------

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
another document. the checking of an inferred-vector against a
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

    MOST (299, 0.9336286783218384): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (146, 0.8106157183647156): «the australian and south african sides for the first cricket test starting at the adelaide oval today are not expected to be finalised until just before the start of play australian captain steve waugh and his south african counterpart shaun pollock will decide on their lineups after an inspection of the pitch shortly before the start of play the match holds special significance for waugh and his twin brother mark who play their th test together steve waugh is not placing too much relevance on the milestone don want to read too much into it guess and then get too carried away but later on when we retire and look back on it it will be significant it nice for the family mum and dad all the sacrifices they made you know with us growing up and also our brothers so you know it nice for the family he said»

    MEDIAN (129, 0.2583906054496765): «the governor general will issue statement this week to answer allegations about his response to alleged sexual abuse at queensland school dr peter hollingworth was the anglican archbishop of brisbane when teacher at toowoomba anglican school allegedly abused students there more than decade ago pressure has been mounting on dr hollingworth to speak out after public criticism of his role in responding to the claims of abuse spokeswoman says dr hollingworth is becoming concerned that if he does not respond publicly to the allegations he may jeopardise the standing of the position of governor general the spokeswoman says dr hollingworth will issue written statement in the next few days after obtaining legal advice four people were killed and eight others injured when fire broke out overnight at hotel in central paris fire service spokesperson says the fire which was brought under control within two hours could have been an act of arson the number of people staying in the hotel du palais at the time the fire was not immediately known the inferno began at around am in the elevator shaft of the six storey hotel next to the theatre du chatelet in paris first arrondissement the centre of the french capital the flames spread quickly via the shaft to the building roof firemen helped several hotel guests to safety through the windows of their rooms two of the victims were found asphyxiated on the fifth floor one of the injured was said to be in serious condition in hospital according to police one man was arrested at the scene and an inquiry has been opened the theatre was undamaged»

    LEAST (87, -0.06319954991340637): «the australian transport safety bureau has called for pilots to be better trained on the risks of air turbulence it is response to helicopter crash last august which claimed the life of media personality shirley strachan mr strachan was on solo navigation training flight on august when he crashed into mt archer on queensland sunshine coast witnesses told of seeing mr strachan apparently struggling to control his aircraft just prior to the crash safety bureau director alan stray says the helicopter was struck by severe air turbulence phenomena known as mountain wave it caused one of the helicopter rotors to flap and strike the tail boom while reluctant to attribute blame mr stray says mountain waves are not uncommon and mr strachan could have been better advised of local weather conditions prior to the flight he says the accident is wake up call to flight trainers to ensure students are fully educated on the dangers of weather phenomena the helicopter training company which owned the aircraft mr strachan died in has declined to comment in detail on the findings blue tongue helicopters owner helen gillies says the company respects the findings of the australian transport safety bureau mrs gillies says the investigation was thorough one but says that the incident is still too painful to discuss the former chief financial officer of retailer harris scarfe will face court on charges following inquiries by the australian securities and investment commission asic the charges to be faced by alan hodgson from beaumont in adelaide eastern suburbs include counts of acting dishonestly as an officer of harris scarfe six counts of acting dishonestly as an employee of the company and eight counts of giving false information to the australian stock exchange the matter has been brought by the commonwealth director of public prosecutions following asic investigation of the company the original harris scarfe business went into receivership in april with debts of about million management buyout by executives not connected with the original company was finalised last month the buyout saw the closure of stores around australia and the retention of others in south australia victoria and tasmania»


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

    Train Document (288): «eight people are to appear in swiss court tomorrow charged with the manslaughter of tourists and three guides after the interlaken canyoning tragedy the first three defendants are managers of the now defunctoperator adventure world twenty one people including australians were killed when thunderstorm struck when they were canyoning down the saxeten river gorge near interlaken massive wall of water hit the group and swept them to their deaths it will be alleged the company adventure world allowed the trip to proceed with no safety provisions in place that they employed inexperienced staff and guides who had lack of knowledge about the violent weather changes which can occur in the mountains if convicted they face one year jail sentence»

    Similar Document (288, 0.955746054649353): «eight people are to appear in swiss court tomorrow charged with the manslaughter of tourists and three guides after the interlaken canyoning tragedy the first three defendants are managers of the now defunctoperator adventure world twenty one people including australians were killed when thunderstorm struck when they were canyoning down the saxeten river gorge near interlaken massive wall of water hit the group and swept them to their deaths it will be alleged the company adventure world allowed the trip to proceed with no safety provisions in place that they employed inexperienced staff and guides who had lack of knowledge about the violent weather changes which can occur in the mountains if convicted they face one year jail sentence»


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

    Test Document (42): «pope john paul ii urged delegates at major summit on sustainable growth on sunday to pursue development that protects the environment and social justice in comments to tourists and the faithful at his summer residence southeast of rome the pope said god had put humans on earth to be his administrators of the land to cultivate it and take care of it in world ever more interdependent peace justice and the safekeeping of creation cannot but be the fruit of joint commitment of all in pursuing the common good john paul said»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (160, 0.7607941031455994): «french moroccan man has been charged in the united states with conspiracy in the terrorist attacks of september it is the first indictment directly related to the suicide hijackings news of the charge came as president george bush delivered major foreign policy speech zaccarias moussaoui sought flying lessons month before the hijackings attorney general john ashcroft claims he was an active participant in the attacks moussaoui is charged with undergoing the same training receiving the same funding and pledging the same commitment to kill americans as the hijackers he said three months to the day since the attacks and president bush says missile defence is now more essential than ever before we must protect america and our friends against all forms of terror including the terror that could arrive on missile he said president bush says the united states now needs dramatically retooled military armed with hi tech weapons and real time intelligence»

    MEDIAN (278, 0.38609835505485535): «the royal commission looking into the collapse of insurance giant hih says the possible leak of confidential document is criminal offence royal commissioner justice neville owen has opened the public hearings into the collapse more than eight months after the company was placed into provisional liquidation in his opening statement justice owen called on all parties to adhere to the confidentiality requirements of royal commission justice owen says there could have been leak of report on the role of auditors circulated in early november it is possible that someone to whom the commission delivered copy of the report in strict confidence disclosed its contents to the author of the article if so there may have been breach of section of the royal commissions act that is criminal offence he said»

    LEAST (8, 0.032870396971702576): «there has been welcome relief for firefighters in new south wales overnight with milder weather allowing them to strengthen containment lines around the most severe fires but fire authorities are not getting overly optimistic as dry and hot weather is forecast to continue the weather bureau is forecasting temperatures in the high and westerly winds until at least friday which means fire authorities are reluctant to get too excited about last night favourable conditions marks sullivan from the rural fire service says fire fighters are remaining on guard lot of fires that have been burning in the areas around sydney and the north coast and further south have been burning within areas that are known and are contained he said however that not to say that these fires won pose threat given the weather conditions that are coming up over the next few days despite the caution the rural fire service says most of the state fires that threaten property are burning within containment lines greater sydney is ringed by fires to the north west and south two of those flared overnight one at appin in the southern highlands was quickly brought under control another flare up at spencer north of the city is not contained on its north western flank but is not threatening property in the lower blue mountains west of sydney firefighters have spent the night setting up kilometre containment line to protect communities along the great western highway from glenbrook to bulaburra two fires burning near cessnock west of newcastle are still within containment lines in the state north aircraft will this morning check if lightning from large electrical storm overnight has sparked any new fires above grafton aircraft have also been used in the shoalhaven area in the state south to drop incendiary devices that start fire control lines in inaccessible areas the rural fire service commissioner phil koperberg says if fire activity increases hundreds of new year eve fireworks celebrations in new south wales will be cancelled»


Conclusion
----------

Let's review what we've seen in this tutorial:

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

   **Total running time of the script:** ( 0 minutes  7.685 seconds)

**Estimated memory usage:**  14 MB


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
