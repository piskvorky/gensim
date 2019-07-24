.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_lda.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_lda.py:


.. _lda_py:

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
* Gensim tutorial: :ref:`tut2`
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

    Average topic coherence: -1.1343.
    [([(0.021013746, 'neuron'),
       (0.01644053, 'cell'),
       (0.007871925, 'spike'),
       (0.007662379, 'stimulus'),
       (0.0076456303, 'activity'),
       (0.0073500657, 'response'),
       (0.006813484, 'synaptic'),
       (0.005910189, 'firing'),
       (0.005157274, 'cortex'),
       (0.0050415667, 'connection'),
       (0.004253718, 'cortical'),
       (0.0039718486, 'signal'),
       (0.0039667464, 'fig'),
       (0.0039561396, 'orientation'),
       (0.003937164, 'potential'),
       (0.0038151569, 'visual'),
       (0.003567992, 'frequency'),
       (0.0035431546, 'noise'),
       (0.0033337239, 'layer'),
       (0.0033196036, 'simulation')],
      -0.7971230608691213),
     ([(0.020144086, 'hidden'),
       (0.013753201, 'layer'),
       (0.011993449, 'rule'),
       (0.011798527, 'hidden_unit'),
       (0.0064462046, 'net'),
       (0.005138334, 'face'),
       (0.004501826, 'propagation'),
       (0.00420462, 'trained'),
       (0.004179971, 'hidden_layer'),
       (0.0041453815, 'activation'),
       (0.003976286, 'back'),
       (0.0038522957, 'node'),
       (0.0038461683, 'generalization'),
       (0.0037686657, 'connection'),
       (0.003474419, 'architecture'),
       (0.0034541178, 'hinton'),
       (0.0033405602, 'human'),
       (0.0031976502, 'back_propagation'),
       (0.0030999542, 'training_set'),
       (0.0030800293, 'component')],
      -0.9490731903943914),
     ([(0.0060573444, 'noise'),
       (0.005816298, 'gradient'),
       (0.005538753, 'matrix'),
       (0.004506274, 'gaussian'),
       (0.004487658, 'approximation'),
       (0.004232299, 'solution'),
       (0.0040157884, 'generalization'),
       (0.0038083668, 'optimal'),
       (0.0034664562, 'minimum'),
       (0.0034521653, 'nonlinear'),
       (0.0034423, 'prediction'),
       (0.0033992517, 'variance'),
       (0.00325635, 'convergence'),
       (0.0031392751, 'optimization'),
       (0.0030667686, 'regression'),
       (0.003048971, 'dynamic'),
       (0.0029236276, 'estimate'),
       (0.00288764, 'descent'),
       (0.0028872795, 'dimensional'),
       (0.0027063866, 'eq')],
      -1.0286954380096913),
     ([(0.015551203, 'image'),
       (0.00808787, 'class'),
       (0.0065504466, 'object'),
       (0.005649323, 'kernel'),
       (0.00546157, 'classifier'),
       (0.005399731, 'classification'),
       (0.004532794, 'distance'),
       (0.004474136, 'sample'),
       (0.0041419175, 'recognition'),
       (0.00409126, 'xi'),
       (0.0036634838, 'code'),
       (0.0031293717, 'node'),
       (0.0029958424, 'support'),
       (0.0028794336, 'machine'),
       (0.0028494904, 'dimensional'),
       (0.002800369, 'matrix'),
       (0.002733493, 'cluster'),
       (0.0027275754, 'solution'),
       (0.00266063, 'estimate'),
       (0.0026073281, 'tree')],
      -1.095912701095505),
     ([(0.010482065, 'circuit'),
       (0.010398917, 'image'),
       (0.008734879, 'chip'),
       (0.008144406, 'motion'),
       (0.0079280315, 'neuron'),
       (0.007865912, 'visual'),
       (0.0075663277, 'analog'),
       (0.006521635, 'signal'),
       (0.006253541, 'field'),
       (0.006088743, 'filter'),
       (0.0055539473, 'response'),
       (0.0054280786, 'voltage'),
       (0.005061166, 'frequency'),
       (0.0049509904, 'direction'),
       (0.0048532938, 'cell'),
       (0.0045184074, 'object'),
       (0.0040257177, 'vlsi'),
       (0.004011863, 'spatial'),
       (0.0035638083, 'pixel'),
       (0.0033756576, 'implementation')],
      -1.1011649110222768),
     ([(0.014598401, 'mixture'),
       (0.009921789, 'likelihood'),
       (0.00927094, 'component'),
       (0.008110908, 'expert'),
       (0.007928639, 'density'),
       (0.0077190395, 'em'),
       (0.006837617, 'source'),
       (0.0067347838, 'gaussian'),
       (0.006350951, 'matrix'),
       (0.0062614563, 'signal'),
       (0.005682685, 'prior'),
       (0.00560245, 'log'),
       (0.0053511574, 'estimate'),
       (0.0051781815, 'independent'),
       (0.0048250565, 'estimation'),
       (0.004763583, 'posterior'),
       (0.003934131, 'maximum'),
       (0.0037864095, 'bayesian'),
       (0.0037242447, 'ica'),
       (0.0035476356, 'rule')],
      -1.1073963468932793),
     ([(0.019600479, 'control'),
       (0.008906189, 'position'),
       (0.008364389, 'motor'),
       (0.008119407, 'movement'),
       (0.007919714, 'controller'),
       (0.0072079613, 'robot'),
       (0.007149695, 'trajectory'),
       (0.0066165724, 'eye'),
       (0.0054811644, 'dynamic'),
       (0.0053690504, 'direction'),
       (0.0053256173, 'head'),
       (0.0048687095, 'object'),
       (0.004618846, 'field'),
       (0.0046145157, 'map'),
       (0.0044109044, 'arm'),
       (0.004407201, 'hand'),
       (0.0041512926, 'visual'),
       (0.0041088536, 'forward'),
       (0.0039988756, 'feedback'),
       (0.0039029063, 'target')],
      -1.2082328617374707),
     ([(0.014463285, 'recognition'),
       (0.012767247, 'speech'),
       (0.011237468, 'word'),
       (0.007489207, 'classifier'),
       (0.0064662024, 'layer'),
       (0.0059147417, 'classification'),
       (0.005725163, 'trained'),
       (0.005559622, 'class'),
       (0.0053695994, 'net'),
       (0.0053035575, 'hidden'),
       (0.0047456273, 'speaker'),
       (0.0044151614, 'signal'),
       (0.0042297463, 'hmm'),
       (0.004001576, 'frame'),
       (0.003972504, 'context'),
       (0.003944948, 'character'),
       (0.0038774577, 'architecture'),
       (0.0037378876, 'mlp'),
       (0.0032704268, 'phoneme'),
       (0.0032700084, 'table')],
      -1.2880043674900854),
     ([(0.008482227, 'bound'),
       (0.0070622917, 'action'),
       (0.006761605, 'policy'),
       (0.006171659, 'let'),
       (0.0061514694, 'theorem'),
       (0.0053646136, 'optimal'),
       (0.004417739, 'class'),
       (0.0042012273, 'approximation'),
       (0.004157366, 'threshold'),
       (0.004003736, 'node'),
       (0.003884812, 'decision'),
       (0.0038475958, 'tree'),
       (0.0037742066, 'proof'),
       (0.0034869409, 'reinforcement'),
       (0.003112567, 'machine'),
       (0.0030329637, 'reinforcement_learning'),
       (0.0029710322, 'reward'),
       (0.0029436839, 'markov'),
       (0.002906668, 'dimension'),
       (0.0028482438, 'sample')],
      -1.3266884779449142),
     ([(0.012729921, 'memory'),
       (0.006277447, 'recurrent'),
       (0.0061073927, 'sequence'),
       (0.0055221734, 'net'),
       (0.004861485, 'node'),
       (0.004076982, 'architecture'),
       (0.0040539615, 'activation'),
       (0.0040328642, 'dynamic'),
       (0.0035397534, 'connection'),
       (0.0033383458, 'attractor'),
       (0.0030969342, 'map'),
       (0.0029194853, 'string'),
       (0.0027578275, 'language'),
       (0.0027054616, 'capacity'),
       (0.002683315, 'symbol'),
       (0.002638282, 'context'),
       (0.002637339, 'associative'),
       (0.0025970018, 'recurrent_network'),
       (0.0025929457, 'bit'),
       (0.0025643448, 'simulation')],
      -1.4409098965687577)]


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

   **Total running time of the script:** ( 2 minutes  16.589 seconds)

**Estimated memory usage:**  634 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_lda.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/010_tutorials/run_lda.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_lda.py <run_lda.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_lda.ipynb <run_lda.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
