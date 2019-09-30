.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_040_lda.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_040_lda.py:


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
* Gensim tutorial: :ref:`sphx_glr_auto_examples_000_core_run_003_topics_and_transformations.py`
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

    Average topic coherence: -1.1631.
    [([(0.018941388, 'cell'),
       (0.011925007, 'neuron'),
       (0.009455616, 'response'),
       (0.009048982, 'stimulus'),
       (0.007671269, 'visual'),
       (0.0071386667, 'activity'),
       (0.0055580228, 'field'),
       (0.00529977, 'cortex'),
       (0.0052038473, 'motion'),
       (0.005054234, 'direction'),
       (0.0045649144, 'orientation'),
       (0.0044472697, 'synaptic'),
       (0.0042819376, 'eye'),
       (0.004275955, 'frequency'),
       (0.004262803, 'connection'),
       (0.004058557, 'cortical'),
       (0.00401355, 'layer'),
       (0.003865375, 'spatial'),
       (0.0036755602, 'firing'),
       (0.0036595368, 'fig')],
      -0.9966928981096688),
     ([(0.0061035696, 'gaussian'),
       (0.005531666, 'class'),
       (0.0051275007, 'bound'),
       (0.0047694743, 'density'),
       (0.0046171057, 'sample'),
       (0.0045263916, 'mixture'),
       (0.004437503, 'log'),
       (0.0044092033, 'approximation'),
       (0.004219231, 'matrix'),
       (0.0041365325, 'likelihood'),
       (0.003944473, 'prior'),
       (0.0037979945, 'let'),
       (0.003748894, 'estimate'),
       (0.0037473696, 'xi'),
       (0.003448978, 'bayesian'),
       (0.00320692, 'dimension'),
       (0.0031519046, 'em'),
       (0.00303186, 'component'),
       (0.002791507, 'distance'),
       (0.002778335, 'posterior')],
      -1.0113903580525427),
     ([(0.027790029, 'image'),
       (0.014158221, 'object'),
       (0.0060714516, 'recognition'),
       (0.0057953815, 'position'),
       (0.005577711, 'pixel'),
       (0.005371924, 'human'),
       (0.005366511, 'field'),
       (0.0052942163, 'visual'),
       (0.0047359117, 'face'),
       (0.0046315645, 'view'),
       (0.004047556, 'region'),
       (0.003937298, 'location'),
       (0.0038542815, 'map'),
       (0.0037850966, 'hand'),
       (0.0035074898, 'vision'),
       (0.0032475195, 'scale'),
       (0.0031939123, 'subject'),
       (0.0031544433, 'scene'),
       (0.0029759305, 'layer'),
       (0.0028098605, 'surface')],
      -1.0815035296808195),
     ([(0.010811567, 'control'),
       (0.010221234, 'action'),
       (0.008239147, 'policy'),
       (0.007557653, 'optimal'),
       (0.0068852054, 'reinforcement'),
       (0.005755256, 'dynamic'),
       (0.0050567104, 'controller'),
       (0.0046745935, 'reinforcement_learning'),
       (0.004623621, 'gradient'),
       (0.004581935, 'trajectory'),
       (0.0043983087, 'stochastic'),
       (0.004232467, 'convergence'),
       (0.0039427625, 'markov'),
       (0.0037591243, 'approximation'),
       (0.0037548228, 'reward'),
       (0.0033655996, 'cost'),
       (0.003346752, 'decision'),
       (0.0033388399, 'iteration'),
       (0.0032832809, 'environment'),
       (0.0032221666, 'goal')],
      -1.0834050336393657),
     ([(0.007187388, 'layer'),
       (0.0071183094, 'prediction'),
       (0.0065068565, 'noise'),
       (0.0063939993, 'hidden'),
       (0.005417221, 'net'),
       (0.004586174, 'generalization'),
       (0.0043244218, 'gradient'),
       (0.004280254, 'nonlinear'),
       (0.004160861, 'training_set'),
       (0.0035800238, 'back'),
       (0.0034490512, 'trained'),
       (0.003433542, 'validation'),
       (0.0033099167, 'hidden_unit'),
       (0.0032337194, 'backpropagation'),
       (0.0032011631, 'regression'),
       (0.0031305265, 'series'),
       (0.0029408198, 'bias'),
       (0.0028783414, 'estimate'),
       (0.0028704463, 'adaptive'),
       (0.0027912245, 'cross')],
      -1.0843775051186193),
     ([(0.013804747, 'recognition'),
       (0.011843204, 'classifier'),
       (0.010172561, 'speech'),
       (0.009980936, 'word'),
       (0.009781723, 'class'),
       (0.009163429, 'classification'),
       (0.0064395797, 'tree'),
       (0.005438045, 'character'),
       (0.0045893015, 'decision'),
       (0.0044410042, 'hmm'),
       (0.0043680836, 'context'),
       (0.0043257414, 'trained'),
       (0.0039209267, 'hidden'),
       (0.003907313, 'speaker'),
       (0.0037514635, 'layer'),
       (0.0034805012, 'training_set'),
       (0.0032691737, 'sequence'),
       (0.0030159175, 'database'),
       (0.0030080057, 'phoneme'),
       (0.0029121016, 'table')],
      -1.1424254474460098),
     ([(0.018475661, 'circuit'),
       (0.014311225, 'chip'),
       (0.013052046, 'analog'),
       (0.011133302, 'neuron'),
       (0.010811847, 'signal'),
       (0.0095082, 'voltage'),
       (0.0066621327, 'vlsi'),
       (0.005756836, 'implementation'),
       (0.005178151, 'channel'),
       (0.0048631816, 'pulse'),
       (0.0048209406, 'bit'),
       (0.0043345788, 'design'),
       (0.0042190365, 'synapse'),
       (0.0042170854, 'gate'),
       (0.004125957, 'digital'),
       (0.0040765954, 'frequency'),
       (0.0040159705, 'threshold'),
       (0.0039510205, 'transistor'),
       (0.0039062097, 'hardware'),
       (0.0037053376, 'device')],
      -1.1899334392220007),
     ([(0.018074784, 'neuron'),
       (0.008531857, 'dynamic'),
       (0.0066785896, 'noise'),
       (0.006553142, 'spike'),
       (0.0042096637, 'eq'),
       (0.0041981437, 'attractor'),
       (0.004116391, 'matrix'),
       (0.0040554316, 'phase'),
       (0.0038537139, 'rule'),
       (0.0038062043, 'solution'),
       (0.0034318683, 'memory'),
       (0.003383885, 'fig'),
       (0.0033570582, 'correlation'),
       (0.0033307914, 'fixed_point'),
       (0.003194164, 'synaptic'),
       (0.0030851834, 'limit'),
       (0.003077789, 'energy'),
       (0.0029919252, 'field'),
       (0.0029767177, 'signal'),
       (0.0029663297, 'firing')],
      -1.2092408330965387),
     ([(0.0087647475, 'rule'),
       (0.008604246, 'node'),
       (0.008419707, 'memory'),
       (0.0068872855, 'net'),
       (0.006783286, 'sequence'),
       (0.004984972, 'recurrent'),
       (0.004737816, 'activation'),
       (0.004594223, 'layer'),
       (0.0040220283, 'hidden'),
       (0.003971988, 'code'),
       (0.003830424, 'bit'),
       (0.0037076208, 'architecture'),
       (0.0034852005, 'connection'),
       (0.0033686326, 'language'),
       (0.0033634938, 'map'),
       (0.0032358621, 'threshold'),
       (0.0032208979, 'string'),
       (0.0032113537, 'machine'),
       (0.0031851404, 'symbol'),
       (0.0029434788, 'parallel')],
      -1.2131501034877972),
     ([(0.011617974, 'hidden'),
       (0.01078546, 'component'),
       (0.010491457, 'signal'),
       (0.008954079, 'source'),
       (0.00838201, 'hidden_unit'),
       (0.0066193896, 'expert'),
       (0.004846498, 'independent'),
       (0.0046763867, 'ica'),
       (0.004400512, 'matrix'),
       (0.0042493143, 'separation'),
       (0.004182415, 'layer'),
       (0.0036441232, 'trained'),
       (0.0035184035, 'rule'),
       (0.003488145, 'net'),
       (0.0034472933, 'target'),
       (0.0031606322, 'eeg'),
       (0.003153443, 'pca'),
       (0.0030816342, 'blind'),
       (0.0030724723, 'channel'),
       (0.00304423, 'noise')],
      -1.6185315413336652)]


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

   **Total running time of the script:** ( 2 minutes  0.724 seconds)

**Estimated memory usage:**  663 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_040_lda.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_040_lda.py <run_040_lda.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_040_lda.ipynb <run_040_lda.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
