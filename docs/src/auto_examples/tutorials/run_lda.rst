.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_lda.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_tutorials_run_lda.py:


LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.



.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








The purpose of this tutorial is to demonstrate how to train and tune an LDA model.

In this tutorial we will:

* Load input data.
* Pre-process that data.
* Transform documents into bag-of-words vectors.
* Train an LDA model.

This tutorial will **not**:

* Explain how Latent Dirichlet Allocation works
* Explain how the LDA model performs inference
* Teach you all the parameters and options for Gensim's LDA implementation

If you are not familiar with the LDA model or how to use it in Gensim, I (Olavur Mortensen)
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

    Average topic coherence: -1.1379.
    [([(0.0081748655, 'bound'),
       (0.007108706, 'let'),
       (0.006066193, 'theorem'),
       (0.005790631, 'optimal'),
       (0.0051151128, 'approximation'),
       (0.004763562, 'convergence'),
       (0.0043320647, 'class'),
       (0.00422147, 'generalization'),
       (0.0037292794, 'proof'),
       (0.0036608914, 'threshold'),
       (0.0034258896, 'sample'),
       (0.003380618, 'loss'),
       (0.0033234654, 'stochastic'),
       (0.003273838, 'finite'),
       (0.0031302413, 'dimension'),
       (0.002831012, 'complexity'),
       (0.002805536, 'assume'),
       (0.0028052146, 'condition'),
       (0.0027610834, 'gradient'),
       (0.002696474, 'xi')],
      -0.9567457219908955),
     ([(0.010490722, 'control'),
       (0.009010948, 'action'),
       (0.0059598447, 'reinforcement'),
       (0.0047827456, 'policy'),
       (0.0044869035, 'controller'),
       (0.0044079036, 'robot'),
       (0.004191326, 'dynamic'),
       (0.0037420355, 'reinforcement_learning'),
       (0.003445968, 'environment'),
       (0.003406964, 'goal'),
       (0.0033234078, 'memory'),
       (0.003115369, 'optimal'),
       (0.0029810574, 'net'),
       (0.00296787, 'architecture'),
       (0.002944391, 'path'),
       (0.0028523172, 'search'),
       (0.002817304, 'cost'),
       (0.0027405526, 'trajectory'),
       (0.0027168898, 'machine'),
       (0.0026868598, 'learn')],
      -0.9629919861002044),
     ([(0.011724874, 'gaussian'),
       (0.009415354, 'component'),
       (0.008502817, 'density'),
       (0.007683166, 'matrix'),
       (0.0075845774, 'mixture'),
       (0.0074258945, 'noise'),
       (0.0065725176, 'likelihood'),
       (0.005647718, 'prior'),
       (0.0055691865, 'bayesian'),
       (0.0055141784, 'signal'),
       (0.0054192273, 'source'),
       (0.0051946733, 'posterior'),
       (0.0047219293, 'variance'),
       (0.0046686446, 'log'),
       (0.004505882, 'independent'),
       (0.004505833, 'estimate'),
       (0.003942736, 'covariance'),
       (0.0038371435, 'filter'),
       (0.0038362807, 'field'),
       (0.003637417, 'sample')],
      -0.9809388920586589),
     ([(0.024790926, 'neuron'),
       (0.01242704, 'cell'),
       (0.010831918, 'spike'),
       (0.007468995, 'response'),
       (0.0074320505, 'firing'),
       (0.0073920223, 'signal'),
       (0.007372279, 'stimulus'),
       (0.007049763, 'circuit'),
       (0.0066122883, 'synaptic'),
       (0.0055389544, 'channel'),
       (0.0050250995, 'potential'),
       (0.0050005503, 'voltage'),
       (0.004911659, 'noise'),
       (0.004659519, 'chip'),
       (0.0043042223, 'threshold'),
       (0.004287492, 'analog'),
       (0.004195864, 'synapsis'),
       (0.0041318936, 'frequency'),
       (0.00403522, 'synapse'),
       (0.003926733, 'activity')],
      -1.081720021366841),
     ([(0.010492251, 'classifier'),
       (0.0091541875, 'class'),
       (0.009068395, 'classification'),
       (0.0067783683, 'recognition'),
       (0.0057456596, 'image'),
       (0.005456874, 'training_set'),
       (0.005113385, 'layer'),
       (0.0046058227, 'trained'),
       (0.004544005, 'character'),
       (0.004493871, 'hidden'),
       (0.0041203136, 'face'),
       (0.003932627, 'distance'),
       (0.003555957, 'prediction'),
       (0.003515055, 'generalization'),
       (0.0033360128, 'net'),
       (0.0031907677, 'validation'),
       (0.003140232, 'digit'),
       (0.003125957, 'sample'),
       (0.0030787578, 'table'),
       (0.002857956, 'test_set')],
      -1.0859109300715886),
     ([(0.015192852, 'image'),
       (0.012347493, 'visual'),
       (0.009574526, 'field'),
       (0.009226306, 'cell'),
       (0.008833845, 'motion'),
       (0.007704693, 'direction'),
       (0.007618932, 'object'),
       (0.006610031, 'response'),
       (0.005630947, 'position'),
       (0.0052231583, 'eye'),
       (0.0051186606, 'spatial'),
       (0.0047765267, 'movement'),
       (0.004564768, 'velocity'),
       (0.0045033526, 'orientation'),
       (0.0043742936, 'stimulus'),
       (0.004288555, 'location'),
       (0.003991875, 'receptive'),
       (0.0038655174, 'map'),
       (0.0038587663, 'pixel'),
       (0.003796966, 'motor')],
      -1.182742215838901),
     ([(0.013556894, 'neuron'),
       (0.0074776006, 'memory'),
       (0.0069611296, 'connection'),
       (0.0068744384, 'dynamic'),
       (0.0068192147, 'cell'),
       (0.0062380563, 'activity'),
       (0.0059009716, 'phase'),
       (0.004971466, 'map'),
       (0.0043283263, 'synaptic'),
       (0.0041240565, 'attractor'),
       (0.003905218, 'fig'),
       (0.0038343568, 'simulation'),
       (0.0035775776, 'field'),
       (0.0035449031, 'layer'),
       (0.0033309432, 'hopfield'),
       (0.0033046217, 'cortex'),
       (0.003234431, 'correlation'),
       (0.0031754398, 'matrix'),
       (0.003062395, 'frequency'),
       (0.0030611386, 'capacity')],
      -1.1977843845453149),
     ([(0.011207352, 'rule'),
       (0.009743068, 'hidden'),
       (0.0063162767, 'layer'),
       (0.0061172424, 'hidden_unit'),
       (0.0050221244, 'net'),
       (0.0040374184, 'image'),
       (0.003990155, 'activation'),
       (0.0035164147, 'node'),
       (0.003386315, 'gradient'),
       (0.0031175737, 'object'),
       (0.002975623, 'learn'),
       (0.0029010042, 'trained'),
       (0.0028902746, 'connectionist'),
       (0.00257973, 'string'),
       (0.002564398, 'recurrent'),
       (0.0024508797, 'language'),
       (0.0024454794, 'distributed'),
       (0.0024281202, 'symbol'),
       (0.0024280634, 'noise'),
       (0.002418099, 'generalization')],
      -1.2755725782614098),
     ([(0.005421432, 'kernel'),
       (0.0047018067, 'constraint'),
       (0.004656475, 'xi'),
       (0.004605332, 'matrix'),
       (0.0045576654, 'solution'),
       (0.0044916016, 'nonlinear'),
       (0.004330045, 'object'),
       (0.0042089145, 'optimization'),
       (0.0040005683, 'cluster'),
       (0.003932736, 'image'),
       (0.0035181807, 'regression'),
       (0.0034628492, 'estimate'),
       (0.003388594, 'clustering'),
       (0.0033658948, 'dimensional'),
       (0.0033489629, 'graph'),
       (0.003346669, 'distance'),
       (0.0031878203, 'transformation'),
       (0.0031517735, 'support'),
       (0.0031086474, 'inverse'),
       (0.0030645302, 'em')],
      -1.3165861629697555),
     ([(0.018751133, 'speech'),
       (0.015273304, 'word'),
       (0.014435461, 'recognition'),
       (0.011104881, 'sequence'),
       (0.008624092, 'node'),
       (0.008493986, 'hidden'),
       (0.0070799952, 'context'),
       (0.0067633255, 'hmm'),
       (0.006759897, 'net'),
       (0.0067320624, 'layer'),
       (0.0065835468, 'speaker'),
       (0.005791399, 'architecture'),
       (0.0053024716, 'signal'),
       (0.0049707727, 'acoustic'),
       (0.004757795, 'recurrent'),
       (0.0047305888, 'tree'),
       (0.0046766074, 'phoneme'),
       (0.004537447, 'frame'),
       (0.0040487116, 'speech_recognition'),
       (0.0039719325, 'markov')],
      -1.3380601736164677)]




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

   **Total running time of the script:** ( 3 minutes  15.684 seconds)

**Estimated memory usage:**  494 MB


.. _sphx_glr_download_auto_examples_tutorials_run_lda.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_lda.py <run_lda.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_lda.ipynb <run_lda.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
