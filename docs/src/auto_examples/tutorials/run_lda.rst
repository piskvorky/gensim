.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_lda.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tutorials_run_lda.py:


LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)







The purpose of this tutorial is to demonstrate training an LDA model and
obtaining good results.

In this tutorial we will:

* Load data.
* Pre-process data.
* Transform documents to a vectorized form.
* Train an LDA model.

This tutorial will **not**:

* Explain how Latent Dirichlet Allocation works
* Explain how the LDA model performs inference
* Teach you how to use Gensim's LDA implementation in its entirety

If you are not familiar with the LDA model or how to use it in Gensim, I
suggest you read up on that before continuing with this tutorial. Basic
understanding of the LDA model should suffice. Examples:

* `Introduction to Latent Dirichlet Allocation <http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation>`_
* Gensim tutorial: :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`
* Gensim's LDA model API docs: :py:class:`gensim.models.LdaModel`

I would also encourage you to consider each step when applying the model to
your data, instead of just blindly applying my solution. The different steps
will depend on your data and possibly your goal with the model.

Data
----

I have used a corpus of NIPS papers in this tutorial, but if you're following
this tutorial just to learn about LDA I encourage you to consider picking a
corpus on a subject that you are familiar with. Qualitatively evaluating the
output of an LDA model is challenging and can require you to understand the
subject matter of your corpus (depending on your goal with the model).

NIPS (Neural Information Processing Systems) is a machine learning conference
so the subject matter should be well suited for most of the target audience
of this tutorial.  You can download the original data from Sam Roweis'
`website <http://www.cs.nyu.edu/~roweis/data.html>`_.  The code below will
also do that for you.

.. Important::
    The corpus contains 1740 documents, and not particularly long ones.
    So keep in mind that this tutorial is not geared towards efficiency, and be
    careful before applying the code to a large dataset.



.. code-block:: default


    import io
    import os.path
    import re
    import tarfile

    import smart_open

    def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
        fname = url.split('/')[-1]
    
        # Download the file to local storage first.
        # We can't read it on the fly because of 
        # https://github.com/RaRe-Technologies/smart_open/issues/331
        if not os.path.isfile(fname):
            with smart_open.open(url, "rb") as fin:
                with smart_open.open(fname, 'wb') as fout:
                    while True:
                        buf = fin.read(io.DEFAULT_BUFFER_SIZE)
                        if not buf:
                            break
                        fout.write(buf)
                         
        with tarfile.open(fname, mode='r:gz') as tar:
            # Ignore directory entries, as well as files like README, etc.
            files = [
                m for m in tar.getmembers()
                if m.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', m.name)
            ]
            for member in sorted(files, key=lambda x: x.name):
                member_bytes = tar.extractfile(member).read()
                yield member_bytes.decode('utf-8', errors='replace')

    docs = list(extract_documents())







So we have a list of 1740 documents, where each document is a Unicode string. 
If you're thinking about using your own corpus, then you need to make sure
that it's in the same format (list of Unicode strings) before proceeding
with the rest of this tutorial.



.. code-block:: default

    print(len(docs))
    print(docs[0][:500])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1740
    1 
    CONNECTIVITY VERSUS ENTROPY 
    Yaser S. Abu-Mostafa 
    California Institute of Technology 
    Pasadena, CA 91125 
    ABSTRACT 
    How does the connectivity of a neural network (number of synapses per 
    neuron) relate to the complexity of the problems it can handle (measured by 
    the entropy)? Switching theory would suggest no relation at all, since all Boolean 
    functions can be implemented using a circuit with very low connectivity (e.g., 
    using two-input NAND gates). However, for a network that learns a pr


Pre-process and vectorize the documents
---------------------------------------

As part of preprocessing, we will:

* Tokenize (split the documents into tokens).
* Lemmatize the tokens.
* Compute bigrams.
* Compute a bag-of-words representation of the data.

First we tokenize the text using a regular expression tokenizer from NLTK. We
remove numeric tokens and tokens that are only a single character, as they
don't tend to be useful, and the dataset contains a lot of them.

.. Important::

   This tutorial uses the nltk library for preprocessing, although you can
   replace it with something else if you want.



.. code-block:: default


    # Tokenize the documents.
    from nltk.tokenize import RegexpTokenizer

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]







We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
stemmer in this case because it produces more readable words. Output that is
easy to read is very desirable in topic modelling.



.. code-block:: default


    # Lemmatize the documents.
    from nltk.stem.wordnet import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]







We find bigrams in the documents. Bigrams are sets of two adjacent words.
Using bigrams we can get phrases like "machine_learning" in our output
(spaces are replaced with underscores); without bigrams we would only get
"machine" and "learning".

Note that in the code below, we find bigrams and then add them to the
original data, because we would like to keep the words "machine" and
"learning" as well as the bigram "machine_learning".

.. Important::
    Computing n-grams of large dataset can be very computationally
    and memory intensive.



.. code-block:: default



    # Compute bigrams.
    from gensim.models import Phrases

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)







We remove rare words and common words based on their *document frequency*.
Below we remove words that appear in less than 20 documents or in more than
50% of the documents. Consider trying to remove words only based on their
frequency, or maybe combining that with this approach.



.. code-block:: default


    # Remove rare and common tokens.
    from gensim.corpora import Dictionary

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)







Finally, we transform the documents to a vectorized form. We simply compute
the frequency of each word, including the bigrams.



.. code-block:: default


    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]







Let's see how many tokens and documents we have to train on.



.. code-block:: default


    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of unique tokens: 8644
    Number of documents: 1740


Training
--------

We are ready to train the LDA model. We will first discuss how to set some of
the training parameters.

First of all, the elephant in the room: how many topics do I need? There is
really no easy answer for this, it will depend on both your data and your
application. I have used 10 topics here because I wanted to have a few topics
that I could interpret and "label", and because that turned out to give me
reasonably good results. You might not need to interpret all your topics, so
you could use a large number of topics, for example 100.

``chunksize`` controls how many documents are processed at a time in the
training algorithm. Increasing chunksize will speed up training, at least as
long as the chunk of documents easily fit into memory. I've set ``chunksize =
2000``, which is more than the amount of documents, so I process all the
data in one go. Chunksize can however influence the quality of the model, as
discussed in Hoffman and co-authors [2], but the difference was not
substantial in this case.

``passes`` controls how often we train the model on the entire corpus.
Another word for passes might be "epochs". ``iterations`` is somewhat
technical, but essentially it controls how often we repeat a particular loop
over each document. It is important to set the number of "passes" and
"iterations" high enough.

I suggest the following way to choose iterations and passes. First, enable
logging (as described in many Gensim tutorials), and set ``eval_every = 1``
in ``LdaModel``. When training the model look for a line in the log that
looks something like this::

   2016-06-21 15:40:06,753 - gensim.models.ldamodel - DEBUG - 68/1566 documents converged within 400 iterations

If you set ``passes = 20`` you will see this line 20 times. Make sure that by
the final passes, most of the documents have converged. So you want to choose
both passes and iterations to be high enough for this to happen.

We set ``alpha = 'auto'`` and ``eta = 'auto'``. Again this is somewhat
technical, but essentially we are automatically learning two parameters in
the model that we usually would have to specify explicitly.



.. code-block:: default



    # Train LDA model.
    from gensim.models import LdaModel

    # Set training parameters.
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )







We can compute the topic coherence of each topic. Below we display the
average topic coherence and print the topics in order of topic coherence.

Note that we use the "Umass" topic coherence measure here (see
:py:func:`gensim.models.ldamodel.LdaModel.top_topics`), Gensim has recently
obtained an implementation of the "AKSW" topic coherence measure (see
accompanying blog post, http://rare-technologies.com/what-is-topic-coherence/).

If you are familiar with the subject of the articles in this dataset, you can
see that the topics below make a lot of sense. However, they are not without
flaws. We can see that there is substantial overlap between some topics,
others are hard to interpret, and most of them have at least some terms that
seem out of place. If you were able to do better, feel free to share your
methods on the blog at http://rare-technologies.com/lda-training-tips/ !



.. code-block:: default


    top_topics = model.top_topics(corpus) #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Average topic coherence: -1.1241.
    [([(0.025163664, 'neuron'),
       (0.014695453, 'cell'),
       (0.009174355, 'spike'),
       (0.008574755, 'synaptic'),
       (0.007183699, 'firing'),
       (0.006625933, 'activity'),
       (0.005360948, 'connection'),
       (0.005293554, 'dynamic'),
       (0.004822483, 'response'),
       (0.004687287, 'potential'),
       (0.004228337, 'memory'),
       (0.003953116, 'synapsis'),
       (0.0038689172, 'fig'),
       (0.0038664965, 'simulation'),
       (0.0037337197, 'phase'),
       (0.0034825401, 'excitatory'),
       (0.0034173392, 'inhibitory'),
       (0.0032120293, 'signal'),
       (0.0031823071, 'membrane'),
       (0.0030939183, 'threshold')],
      -0.9630445183762313),
     ([(0.012538756, 'visual'),
       (0.010721944, 'cell'),
       (0.010432726, 'stimulus'),
       (0.009539313, 'response'),
       (0.009375428, 'field'),
       (0.008074537, 'motion'),
       (0.007172039, 'direction'),
       (0.0067870775, 'eye'),
       (0.006605871, 'orientation'),
       (0.0060072606, 'map'),
       (0.005874502, 'signal'),
       (0.0057511893, 'spatial'),
       (0.0052507855, 'activity'),
       (0.0051356875, 'frequency'),
       (0.005135085, 'cortex'),
       (0.0048966897, 'neuron'),
       (0.0047698235, 'receptive'),
       (0.004372744, 'receptive_field'),
       (0.0043261987, 'position'),
       (0.00429431, 'movement')],
      -1.0239601445556414),
     ([(0.0083479555, 'noise'),
       (0.0068552294, 'matrix'),
       (0.0053806016, 'generalization'),
       (0.005088047, 'gradient'),
       (0.004749316, 'gaussian'),
       (0.004203275, 'solution'),
       (0.004060732, 'hidden'),
       (0.0038100502, 'variance'),
       (0.0036777114, 'optimal'),
       (0.003360351, 'minimum'),
       (0.0029607583, 'approximation'),
       (0.0029424587, 'regression'),
       (0.0029251142, 'prediction'),
       (0.0029126815, 'hidden_unit'),
       (0.0028326688, 'field'),
       (0.0027481643, 'eq'),
       (0.0027452093, 'curve'),
       (0.0027210945, 'component'),
       (0.0026757186, 'training_set'),
       (0.0025839263, 'convergence')],
      -1.0264554313733174),
     ([(0.016699685, 'layer'),
       (0.010172412, 'hidden'),
       (0.009873925, 'net'),
       (0.0069913934, 'signal'),
       (0.005923669, 'architecture'),
       (0.005888097, 'node'),
       (0.0053476817, 'recognition'),
       (0.0052034874, 'back'),
       (0.005092546, 'trained'),
       (0.0049700774, 'character'),
       (0.0049465224, 'propagation'),
       (0.0043775956, 'connection'),
       (0.003985166, 'rule'),
       (0.0039540627, 'hidden_layer'),
       (0.0039305384, 'back_propagation'),
       (0.0037728392, 'hidden_unit'),
       (0.0036262535, 'map'),
       (0.0033271443, 'memory'),
       (0.0032461125, 'recurrent'),
       (0.003240172, 'classification')],
      -1.0502625308865845),
     ([(0.029825492, 'image'),
       (0.014187608, 'object'),
       (0.00787895, 'recognition'),
       (0.0060228026, 'face'),
       (0.0057333205, 'distance'),
       (0.005409843, 'pixel'),
       (0.004563485, 'view'),
       (0.0041269716, 'human'),
       (0.0035166328, 'region'),
       (0.0033159077, 'scale'),
       (0.003022337, 'transformation'),
       (0.0029435642, 'vision'),
       (0.00290788, 'classification'),
       (0.002907101, 'visual'),
       (0.0029046696, 'scene'),
       (0.0027043023, 'shape'),
       (0.002684278, 'similarity'),
       (0.0026817669, 'location'),
       (0.002623082, 'hand'),
       (0.0025962282, 'class')],
      -1.097725649478162),
     ([(0.0077803913, 'class'),
       (0.00701301, 'bound'),
       (0.006212604, 'tree'),
       (0.0053535043, 'let'),
       (0.0050069927, 'sample'),
       (0.004901047, 'theorem'),
       (0.0048944, 'node'),
       (0.0048409384, 'approximation'),
       (0.0044921115, 'rule'),
       (0.0043274118, 'xi'),
       (0.0042111403, 'log'),
       (0.0034545094, 'threshold'),
       (0.0032513374, 'dimension'),
       (0.0031659885, 'estimate'),
       (0.0029222067, 'decision'),
       (0.0029162255, 'density'),
       (0.0027602788, 'polynomial'),
       (0.0027468363, 'proof'),
       (0.0026415242, 'complexity'),
       (0.0025608365, 'classification')],
      -1.1117463109554),
     ([(0.013324664, 'control'),
       (0.011131293, 'action'),
       (0.008681728, 'policy'),
       (0.007468294, 'reinforcement'),
       (0.006033033, 'optimal'),
       (0.0058535463, 'controller'),
       (0.0054354914, 'dynamic'),
       (0.0052186167, 'robot'),
       (0.004936079, 'reinforcement_learning'),
       (0.0046352767, 'environment'),
       (0.004009696, 'reward'),
       (0.0039904723, 'trajectory'),
       (0.003791848, 'goal'),
       (0.0033601716, 'path'),
       (0.0032013957, 'decision'),
       (0.002936101, 'sutton'),
       (0.0029276484, 'td'),
       (0.0028193546, 'cost'),
       (0.0027700174, 'trial'),
       (0.0027414286, 'learn')],
      -1.142543624221162),
     ([(0.011205725, 'speech'),
       (0.010960422, 'word'),
       (0.008931443, 'mixture'),
       (0.008391879, 'recognition'),
       (0.00627159, 'gaussian'),
       (0.0059411423, 'likelihood'),
       (0.005645111, 'classifier'),
       (0.0050329296, 'class'),
       (0.0048486707, 'sequence'),
       (0.004785308, 'kernel'),
       (0.00473608, 'hmm'),
       (0.004446403, 'context'),
       (0.0044057737, 'estimate'),
       (0.0042781103, 'density'),
       (0.0042298213, 'speaker'),
       (0.004190155, 'rbf'),
       (0.004078858, 'classification'),
       (0.0036422184, 'estimation'),
       (0.0036205945, 'prior'),
       (0.0036079672, 'hidden')],
      -1.179015820296614),
     ([(0.017962778, 'circuit'),
       (0.014027779, 'chip'),
       (0.013046392, 'analog'),
       (0.008767594, 'voltage'),
       (0.008703562, 'neuron'),
       (0.006658798, 'signal'),
       (0.0065101394, 'vlsi'),
       (0.00574142, 'implementation'),
       (0.0054374044, 'bit'),
       (0.0047924053, 'processor'),
       (0.0042650327, 'pulse'),
       (0.004172195, 'channel'),
       (0.003977853, 'design'),
       (0.003924252, 'gate'),
       (0.0039178976, 'digital'),
       (0.0038753191, 'transistor'),
       (0.0037934151, 'device'),
       (0.0037664415, 'hardware'),
       (0.0037505976, 'cell'),
       (0.0036221847, 'synapse')],
      -1.217220238817786),
     ([(0.0052642035, 'net'),
       (0.005045612, 'hidden'),
       (0.0046278588, 'sequence'),
       (0.004625344, 'machine'),
       (0.004386533, 'solution'),
       (0.004208156, 'language'),
       (0.004180493, 'node'),
       (0.0038425317, 'string'),
       (0.0037875888, 'hidden_unit'),
       (0.0037045274, 'cost'),
       (0.003578985, 'optimization'),
       (0.00333463, 'constraint'),
       (0.0033199114, 'table'),
       (0.0033088576, 'recurrent'),
       (0.003233348, 'code'),
       (0.0031989065, 'symbol'),
       (0.003080977, 'activation'),
       (0.003000487, 'matrix'),
       (0.002989608, 'search'),
       (0.0026564174, 'grammar')],
      -1.4290562789759915)]


Things to experiment with
-------------------------

* ``no_above`` and ``no_below`` parameters in ``filter_extremes`` method.
* Adding trigrams or even higher order n-grams.
* Consider whether using a hold-out set or cross-validation is the way to go for you.
* Try other datasets.

Where to go from here
---------------------

* Check out a RaRe blog post on the AKSW topic coherence measure (http://rare-technologies.com/what-is-topic-coherence/).
* pyLDAvis (https://pyldavis.readthedocs.io/en/latest/index.html).
* Read some more Gensim tutorials (https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials).
* If you haven't already, read [1] and [2] (see references).

References
----------

1. "Latent Dirichlet Allocation", Blei et al. 2003.
2. "Online Learning for Latent Dirichlet Allocation", Hoffman et al. 2010.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 2 minutes  11.266 seconds)

**Estimated memory usage:**  547 MB


.. _sphx_glr_download_auto_examples_tutorials_run_lda.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_lda.py <run_lda.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_lda.ipynb <run_lda.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
