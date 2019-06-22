.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_doc2vec_lee.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_doc2vec_lee.py:


.. _doc2vec_lee_py:

Doc2Vec Tutorial
================

Doc2Vec is an NLP tool for representing documents as a vector and is a generalizing of the Word2Vec method.
This tutorial will serve as an introduction to Doc2Vec and present ways to train and assess a Doc2Vec model.

.. code-block:: default


    import gensim
    import os
    import collections
    import smart_open
    import random








Resources
---------




* `Word2Vec Paper <https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_



* `Doc2Vec Paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_



* `Dr. Michael D. Lee's Website <http://faculty.sites.uci.edu/mdlee>`_



* `Lee Corpus <http://faculty.sites.uci.edu/mdlee/similarity-data/>`__



* `IMDB Doc2Vec Tutorial <doc2vec-IMDB.ipynb>`_


Getting Started
---------------


.. role:: raw-html-m2r(raw)
   :format: html


To get going, we'll need to have a set of documents to train our doc2vec model. In theory, a document could be anything from a short 140 character tweet, a single paragraph (i.e., journal article abstract), a news article, or a book. In NLP parlance a collection or set of documents is often referred to as a :raw-html-m2r:`<b>corpus</b>`. 



For this tutorial, we'll be training our model using the `Lee Background Corpus <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_ included in gensim. This corpus contains 314 documents selected from the Australian Broadcasting


Corporation’s news mail service, which provides text e-mails of headline stories and covers a number of broad topics.



And we'll test our model by eye using the much shorter `Lee Corpus <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_ which contains 50 documents.



.. code-block:: default



    # Set file names for train and test data
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    lee_test_file = test_data_dir + os.sep + 'lee.cor'







Define a Function to Read and Preprocess Text
---------------------------------------------


Below, we define a function to open the train/test file (with latin encoding), read the file line-by-line, pre-process each line using a simple gensim pre-processing tool (i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Note that, for a given file (aka corpus), each continuous line constitutes a single document and the length of each line (i.e., document) can vary. Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.



.. code-block:: default



    def read_corpus(fname, tokens_only=False):
        with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
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



    train_corpus[:2]







And the testing corpus looks like this:



.. code-block:: default



    print(test_corpus[:2])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [['the', 'national', 'executive', 'of', 'the', 'strife', 'torn', 'democrats', 'last', 'night', 'appointed', 'little', 'known', 'west', 'australian', 'senator', 'brian', 'greig', 'as', 'interim', 'leader', 'shock', 'move', 'likely', 'to', 'provoke', 'further', 'conflict', 'between', 'the', 'party', 'senators', 'and', 'its', 'organisation', 'in', 'move', 'to', 'reassert', 'control', 'over', 'the', 'party', 'seven', 'senators', 'the', 'national', 'executive', 'last', 'night', 'rejected', 'aden', 'ridgeway', 'bid', 'to', 'become', 'interim', 'leader', 'in', 'favour', 'of', 'senator', 'greig', 'supporter', 'of', 'deposed', 'leader', 'natasha', 'stott', 'despoja', 'and', 'an', 'outspoken', 'gay', 'rights', 'activist'], ['cash', 'strapped', 'financial', 'services', 'group', 'amp', 'has', 'shelved', 'million', 'plan', 'to', 'buy', 'shares', 'back', 'from', 'investors', 'and', 'will', 'raise', 'million', 'in', 'fresh', 'capital', 'after', 'profits', 'crashed', 'in', 'the', 'six', 'months', 'to', 'june', 'chief', 'executive', 'paul', 'batchelor', 'said', 'the', 'result', 'was', 'solid', 'in', 'what', 'he', 'described', 'as', 'the', 'worst', 'conditions', 'for', 'stock', 'markets', 'in', 'years', 'amp', 'half', 'year', 'profit', 'sank', 'per', 'cent', 'to', 'million', 'or', 'share', 'as', 'australia', 'largest', 'investor', 'and', 'fund', 'manager', 'failed', 'to', 'hit', 'projected', 'per', 'cent', 'earnings', 'growth', 'targets', 'and', 'was', 'battered', 'by', 'falling', 'returns', 'on', 'share', 'markets']]


Notice that the testing corpus is just a list of lists and does not contain any tags.


Training the Model
------------------


Instantiate a Doc2Vec Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 40 times. We set the minimum word count to 2 in order to discard words with very few occurrences. (Without a variety of representative examples, retaining such infrequent words can often make a model worse!) Typical iteration counts in published 'Paragraph Vectors' results, using 10s-of-thousands to millions of docs, are 10-20. More iterations take more time and eventually reach a point of diminishing returns.



However, this is a very very small dataset (300 documents) with shortish documents (a few hundred words). Adding training passes can sometimes help with such small datasets.



.. code-block:: default



    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)







Build a Vocabulary
^^^^^^^^^^^^^^^^^^



.. code-block:: default



    model.build_vocab(train_corpus)







Essentially, the vocabulary is a dictionary (accessible via ``model.wv.vocab``\ ) of all of the unique words extracted from the training corpus along with the count (e.g., ``model.wv.vocab['penalty'].count`` for counts for the word ``penalty``\ ).


Time to Train
^^^^^^^^^^^^^



If the BLAS library is being used, this should take no more than 3 seconds.


If the BLAS library is not being used, this should take no more than 2 minutes, so use BLAS if you value your time.



.. code-block:: default



    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)







Inferring a Vector
^^^^^^^^^^^^^^^^^^


One important thing to note is that you can now infer a vector for any piece of text without having to re-train the model by passing a list of words to the ``model.infer_vector`` function. This vector can then be compared with other vectors via cosine similarity.



.. code-block:: default



    model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])







Note that ``infer_vector()`` does *not* take a string, but rather a list of string tokens, which should have already been tokenized the same way as the ``words`` property of original training document objects. 



Also note that because the underlying training/inference algorithms are an iterative approximation problem that makes use of internal randomization, repeated inferences of the same text will return slightly different vectors.


Assessing Model
---------------


To assess our new model, we'll first infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, and then returning the rank of the document based on self-similarity. Basically, we're pretending as if the training corpus is some new unseen data and then seeing how they compare with the trained model. The expectation is that we've likely overfit our model (i.e., all of the ranks will be less than 2) and so we should be able to find similar documents very easily. Additionally, we'll keep track of the second ranks for a comparison of less similar documents. 



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



.. code-block:: default



    collections.Counter(ranks)  # Results vary between runs due to random seeding and very small corpus







Basically, greater than 95% of the inferred documents are found to be most similar to itself and about 5% of the time it is mistakenly most similar to another document. the checking of an inferred-vector against a training-vector is a sort of 'sanity check' as to whether the model is behaving in a usefully consistent manner, though not a real 'accuracy' value.



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

    MOST (299, 0.9343571066856384): «australia will take on france in the doubles rubber of the davis cup tennis final today with the tie levelled at wayne arthurs and todd woodbridge are scheduled to lead australia in the doubles against cedric pioline and fabrice santoro however changes can be made to the line up up to an hour before the match and australian team captain john fitzgerald suggested he might do just that we ll make team appraisal of the whole situation go over the pros and cons and make decision french team captain guy forget says he will not make changes but does not know what to expect from australia todd is the best doubles player in the world right now so expect him to play he said would probably use wayne arthurs but don know what to expect really pat rafter salvaged australia davis cup campaign yesterday with win in the second singles match rafter overcame an arm injury to defeat french number one sebastien grosjean in three sets the australian says he is happy with his form it not very pretty tennis there isn too many consistent bounces you are playing like said bit of classic old grass court rafter said rafter levelled the score after lleyton hewitt shock five set loss to nicholas escude in the first singles rubber but rafter says he felt no added pressure after hewitt defeat knew had good team to back me up even if we were down he said knew could win on the last day know the boys can win doubles so even if we were down still feel we are good enough team to win and vice versa they are good enough team to beat us as well»

    SECOND-MOST (104, 0.7842312455177307): «australian cricket captain steve waugh has supported fast bowler brett lee after criticism of his intimidatory bowling to the south african tailenders in the first test in adelaide earlier this month lee was fined for giving new zealand tailender shane bond an unsportsmanlike send off during the third test in perth waugh says tailenders should not be protected from short pitched bowling these days you re earning big money you ve got responsibility to learn how to bat he said mean there no times like years ago when it was not professional and sort of bowlers code these days you re professional our batsmen work very hard at their batting and expect other tailenders to do likewise meanwhile waugh says his side will need to guard against complacency after convincingly winning the first test by runs waugh says despite the dominance of his side in the first test south africa can never be taken lightly it only one test match out of three or six whichever way you want to look at it so there lot of work to go he said but it nice to win the first battle definitely it gives us lot of confidence going into melbourne you know the big crowd there we love playing in front of the boxing day crowd so that will be to our advantage as well south africa begins four day match against new south wales in sydney on thursday in the lead up to the boxing day test veteran fast bowler allan donald will play in the warm up match and is likely to take his place in the team for the second test south african captain shaun pollock expects much better performance from his side in the melbourne test we still believe that we didn play to our full potential so if we can improve on our aspects the output we put out on the field will be lot better and we still believe we have side that is good enough to beat australia on our day he said»

    MEDIAN (119, 0.2657945156097412): «australia is continuing to negotiate with the united states government in an effort to interview the australian david hicks who was captured fighting alongside taliban forces in afghanistan mr hicks is being held by the united states on board ship in the afghanistan region where the australian federal police and australian security intelligence organisation asio officials are trying to gain access foreign affairs minister alexander downer has also confirmed that the australian government is investigating reports that another australian has been fighting for taliban forces in afghanistan we often get reports of people going to different parts of the world and asking us to investigate them he said we always investigate sometimes it is impossible to find out we just don know in this case but it is not to say that we think there are lot of australians in afghanistan the only case we know is hicks mr downer says it is unclear when mr hicks will be back on australian soil but he is hopeful the americans will facilitate australian authorities interviewing him»

    LEAST (233, -0.05824951454997063): «three us troops and five members of the afghan opposition were killed by stray us bomb near kandahar in afghanistan the pentagon said the pentagon had earlier confirmed that two us special forces soldiers were killed and others wounded north of kandahar when bomber dropped pound bomb too close to them the was flying in support of opposition forces north of kandahar said pentagon spokeswoman victori clark we have an update since this morning and unfortunately the number of us forces killed is now three rival afghan factions signed an historic power sharing agreement to form post taliban government and set the country on the road to recovery and democracy after two decades of war the accord was sealed after nine days of exhausting negotiations and paves the way for six month interim administration headed by moderate muslim hamid karzai from the dominant pashtun ethnic group the deal gives the northern alliance control of three key portfolios in the member cabinet which includes two women and is due to be up and running by december it also gives symbolic role to the former king and provides for un security force for kabul the agreement was signed in the german city of bonn by the leaders of the four delegations and un special envoy for afghanistan lakhdar brahimi to applause from an audience which included german chancellor gerhard schroeder we were the champions of resistance and will be proud to be the champions of peace said yunus qanooni the northern alliance chief negotiator and the interim government interior minister delegate from the so called peshawar group sayed hamed gailani summed up the atmosphere in single phrase there are two things evident today yesterday rain does not have the courage to cry and the sun cannot hide its smile he said the appointment of karzai year old tribal pashtun tribal leader currently fighting the taliban near their last stronghold of kandahar was seen as an attempt to balance afghanistan delicate ethnic mix it cements whirlwind transformation in afghanistan fate since the september attacks on new york and washington the trigger for massive us air strikes that have dislodged the taliban militia from most of the country and put the northern alliance back on top showing the strain from nine days of frantic diplomacy brahimi recognised the accord was far from perfect and that its signatories were not fully representative of the afghan people»


Notice above that the most similar document (usually the same text) is has a similarity score approaching 1.0. However, the similarity score for the second-ranked documents should be significantly lower (assuming the documents are in fact different) and the reasoning becomes obvious when we examine the text itself.



We can run the next cell repeatedly to see a sampling other target-document comparisons. 



.. code-block:: default



    # Pick a random document from the corpus and infer a vector from the model
    doc_id = random.randint(0, len(train_corpus) - 1)

    # Compare and print the second-most-similar document
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Train Document (115): «australia is continuing to negotiate with the united states government in an effort to interview the australian david hicks who was captured fighting alongside taliban forces in afghanistan mr hicks is being held by the united states on board ship in the afghanistan region where the australian federal police and australian security intelligence organisation asio officials are trying to gain access foreign affairs minister alexander downer has also confirmed that the australian government is investigating reports that another australian has been fighting for taliban forces in afghanistan we often get reports of people going to different parts of the world and asking us to investigate them he said we always investigate sometimes it is impossible to find out we just don know in this case but it is not to say that we think there are lot of australians in afghanistan the only case we know is hicks mr downer says it is unclear when mr hicks will be back on australian soil but he is hopeful the americans will facilitate australian authorities interviewing him»

    Similar Document (119, 0.928067684173584): «australia is continuing to negotiate with the united states government in an effort to interview the australian david hicks who was captured fighting alongside taliban forces in afghanistan mr hicks is being held by the united states on board ship in the afghanistan region where the australian federal police and australian security intelligence organisation asio officials are trying to gain access foreign affairs minister alexander downer has also confirmed that the australian government is investigating reports that another australian has been fighting for taliban forces in afghanistan we often get reports of people going to different parts of the world and asking us to investigate them he said we always investigate sometimes it is impossible to find out we just don know in this case but it is not to say that we think there are lot of australians in afghanistan the only case we know is hicks mr downer says it is unclear when mr hicks will be back on australian soil but he is hopeful the americans will facilitate australian authorities interviewing him»


Testing the Model
-----------------


Using the same approach above, we'll infer the vector for a randomly chosen test document, and compare the document to our model by eye.



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

    Test Document (41): «russia defended itself against criticism of its economic ties with countries like iraq saying attempts to mix business and ideology were misguided mixing ideology with economic ties which was characteristic of the cold war that russia and the united states worked to end is thing of the past russian foreign ministry spokesman boris malakhov said saturday reacting to defense secretary donald rumsfeld statement that moscow economic relationships with such countries sends negative signal»

    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dm/m,d50,n5,w5,mc2,s0.001,t3):

    MOST (94, 0.6798781752586365): «foreign minister alexander downer says the commonwealth democracy watchdog should put zimbabwe formally on its agenda in the first step to possible suspension from the organisation mr downer says ministers from the commonwealth ministerial action group cmag should review whether the reported violence and intimidation in zimbabwe means it has violated the commonwealth code of good governance cmag ministers from australia bangladesh barbados botswana britain canada malaysia and nigeria will meet in london tomorrow for talks on zimbabwe in recent meetings they have suspended both fiji and pakistan following military coups however their talks on the violent campaign of farm occupations in zimbabwe have been restricted to informal discussions as president robert mugabe government holds power through recognised elections mr downer also says the commonwealth ministers should maintain pressure on president mugabe to allow international observers to oversee presidential elections next march»

    MEDIAN (83, 0.29019537568092346): «the opposition leader simon crean says child abuse scandal in brisbane has damaged the office of the governor general and its incumbent dr peter hollingworth child advocates have called on dr hollingworth to step down as governor general saying he did not do enough to prevent abuse of children in an anglican school when he was archbishop of brisbane mr crean says he is not calling on dr hollingworth to resign but he says there are still unanswered questions think it has tarnished the office of the governor general the fact that it took so long for this statement to come out he said many people have been calling for it me included think if we are to avoid further damage to the office we need to clear it up completely brisbane lord mayor says the governor general explanation of his handling of child sex abuse allegations at queensland school raises more questions than it answers jim soorley who is former catholic priest says the explanation does not wash within the christian tradition bishops are regarded as shepherds he said it very clear that he was not good shepherd and there are serious consequences for that think his actions are not the actions of good shepherd and think there are still questions to be answered»

    LEAST (33, -0.18467053771018982): «new south wales firefighters are hoping lighter winds will help ease their workload today but are predicting nasty conditions over the weekend while the winds are expected to ease somewhat today the weather bureau says temperatures will be higher more than fires are still burning across new south wales the rural fire service says the change may allow it to concentrate more on preventative action but there is no room for complacency mark sullivan from the rural fire service says while conditions may be little kinder to them today the outlook for the weekend has them worried it certainly appears from the weather forecast with very high temperatures and high winds that it certainly could be nasty couple of days ahead mr sullivan said one of the areas causing greatest concern today is the kilometre long blaze in the lower blue mountains firefighters are also keeping close eye on blaze at spencer north of sydney which yesterday broke through containment lines there are concerns that fire may jump the hawkesbury river backburning continues in the state central west and south of sydney in the shoalhaven in the illawarra firefighters have been able to carry out back burning operations in three areas operations were carried out in parts of mt kembla as well as an area bounded by appin road and the old princes highway at helensburgh an area west of windy gully near cataract dam was also targeted meanwhile illawarra police have arrested three teenagers in relation to bushfires at shellharbour on the south coast of new south wales spokesman says three small fires were extinguished around pm aedt yesterday short time later police arrested three year old boys from shellharbour barrack heights and shell cove all three have been interviewed but no charges have been laid»


Wrapping Up
^^^^^^^^^^^



That's it! Doc2Vec is a great way to explore relationships between documents.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  13.757 seconds)

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
