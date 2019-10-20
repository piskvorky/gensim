:orphan:



.. _sphx_glr_auto_examples:

Documentation
=============

This section contains gensim tutorials and howto guides.
The **tutorials** are learning-oriented: they are lessons that introduce gensim functionality to new users.
The **guides** are goal-oriented: they demonstrate how to solve a specific problem using gensim.

We always welcome contributions to our documentation via GitHub pull requests, whether it's fixing a typo or authoring an entirely new tutorial or guide.
If you're thinking about contributing documentation, please see :ref:`sphx_glr_auto_examples_howtos_run_doc.py`.


.. raw:: html

    <div style='clear:both'></div>



.. _sphx_glr_auto_examples_core:

Core Tutorials
--------------

These tutorials introduce core gensim functionality.
Understanding this functionality is vital for using gensim effectively.
Each tutorial is stand-alone, but if you're new, we recommend going through all of them, in order.

Click on the thumbnails below to access the tutorials.
If you mouse-over a thumbnail, you will see more context the tutorial.
Each tutorial is available in several formats:

- HTML: this is the default.  Everything is viewable in your browser.
- `Jupyter notebook <https://jupyter.org/>`__: You'll see a separate download link at the end of each tutorial.
  If you have a working copy of the ``gensim`` `git repository <https://github.com/RaRe-Technologies/gensim>`__, you'll also find these
  notebooks under :file:`docs/src/gallery/auto_examples`.
- `Binder <https://mybinder.org>`__: see the :ref:`sphx_glr_auto_examples_howtos_run_binder.py` for more details.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial introduces Documents, Corpora, Vectors and Models: the basic concepts and terms n...">

.. only:: html

    .. figure:: /auto_examples/core/images/thumb/sphx_glr_run_core_concepts_thumb.png

        :ref:`sphx_glr_auto_examples_core_run_core_concepts.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/core/run_core_concepts

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates transforming text into a vector space representation.">

.. only:: html

    .. figure:: /auto_examples/core/images/thumb/sphx_glr_run_corpora_and_vector_spaces_thumb.png

        :ref:`sphx_glr_auto_examples_core_run_corpora_and_vector_spaces.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/core/run_corpora_and_vector_spaces

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces transformations and demonstrates their use on a toy corpus.">

.. only:: html

    .. figure:: /auto_examples/core/images/thumb/sphx_glr_run_topics_and_transformations_thumb.png

        :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/core/run_topics_and_transformations

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates querying a corpus for similar documents.">

.. only:: html

    .. figure:: /auto_examples/core/images/thumb/sphx_glr_run_similarity_queries_thumb.png

        :ref:`sphx_glr_auto_examples_core_run_similarity_queries.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/core/run_similarity_queries
.. raw:: html

    <div style='clear:both'></div>



.. _sphx_glr_auto_examples_tutorials:

Tutorials
---------

Once you're familiar with gensim's core concepts, try one of the tutorials below.
These tutorials are **learning-oriented lessons** that introduce a particular gensim feature, e.g. a model (Word2Vec, FastText) or technique (similarity queries or text summarization).



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces Gensim&#x27;s Doc2Vec model and demonstrates its use on the Lee Corpus.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_doc2vec_lee_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_doc2vec_lee.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_doc2vec_lee

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces Gensim&#x27;s Word2Vec model and demonstrates its use on the Lee Corpus.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_word2vec_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_word2vec

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces the annoy library for similarity queries using a Word2Vec model.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_annoy_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_annoy.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_annoy

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces Gensim&#x27;s fastText model and demonstrates its use on the Lee Corpus.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_fasttext_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_fasttext.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_fasttext

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces Gensim&#x27;s LDA model and demonstrates its use on the NIPS corpus.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_lda_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_lda.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_lda

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces the concept of distance between two bags of words or distributions, and demonstrates...">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_distance_metrics_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_distance_metrics.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_distance_metrics

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates using Gensim&#x27;s implemenation of the WMD.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_wmd_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_wmd.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_wmd

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates summarizing text by extracting the most important sentences from it.">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_summarization_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_summarization.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_summarization

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates using Pivoted Document Length Normalization to counter the effect of...">

.. only:: html

    .. figure:: /auto_examples/tutorials/images/thumb/sphx_glr_run_pivoted_doc_norm_thumb.png

        :ref:`sphx_glr_auto_examples_tutorials_run_pivoted_doc_norm.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/tutorials/run_pivoted_doc_norm
.. raw:: html

    <div style='clear:both'></div>



.. _sphx_glr_auto_examples_howtos:

How-to Guides
-------------

How-to guides will go here.
These **goal-oriented guides** demonstrate how to **solve a specific problem** using gensim.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how you can compare a model with itself and other models, and why you need it.">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_040_compare_lda_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_040_compare_lda.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_040_compare_lda

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates simple and quick access to common corpora, models, and other data.">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_downloader_api_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_downloader_api.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_downloader_api

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If you don&#x27;t have Jupyter installed locally, you can still play with the examples provided in t...">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_binder_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_binder.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_binder

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Some tips of how to author documentation for gensim.">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_doc_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_doc.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_doc

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shows how to reproduce results of the Le and Mikolov paper using Gensim.">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_doc2vec_imdb_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_doc2vec_imdb.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_doc2vec_imdb

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates classification of the Lee Corpus using a variety of topic models (LSI, HDP, LDA, e...">

.. only:: html

    .. figure:: /auto_examples/howtos/images/thumb/sphx_glr_run_news_classification_thumb.png

        :ref:`sphx_glr_auto_examples_howtos_run_news_classification.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/howtos/run_news_classification
.. raw:: html

    <div style='clear:both'></div>



.. _sphx_glr_auto_examples_other:

Other Resources
---------------

Quick-start
-----------

-  `Getting Started with
   gensim <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_core_concepts.html>`__

Text to Vectors
---------------

-  We first need to transform text to vectors
-  `String to vectors
   tutorial <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_corpora_and_vector_spaces.html>`__

   -  Create a dictionary first that maps words to ids
   -  Transform the text into vectors through
      ``dictionary.doc2bow(texts)``

-  `Corpus streaming
   tutorial <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_corpora_and_vector_spaces.html>`__
   (For very large corpuses)

Models and Transformation
-------------------------

-  Models (e.g. LsiModel, Word2Vec) are built / trained from a corpus
-  `Transformation interface
   tutorial <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_topics_and_transformations.html>`__

TF-IDF (Model)
--------------

-  `Docs <https://radimrehurek.com/gensim/models/tfidfmodel.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/tfidfmodel.py>`__
-  `tf-idf scores are
   normalized <http://stackoverflow.com/questions/9470479/how-is-tf-idf-implemented-in-gensim-tool-in-python>`__
   (sum of squares of scores = 1)

Phrases (Model)
---------------

-  Detects words that belong in a phrase, useful for models like
   Word2Vec ("new", "york" -> "new york")
-  `Docs <https://radimrehurek.com/gensim/models/phrases.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/phrases.py>`__
   (uses bigram detectors underneath)
-  `Phrases example on How I Met Your
   Mother <http://www.markhneedham.com/blog/2015/02/12/pythongensim-creating-bigrams-over-how-i-met-your-mother-transcripts/>`__

Topic Modeling
--------------

LSI (Model)
~~~~~~~~~~~

-  `Docs <https://radimrehurek.com/gensim/models/lsimodel.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/lsimodel.py>`__
   (very standard LSI implementation)
-  `How to interpret negative LSI
   values <https://www.researchgate.net/post/LSA_SVD_How_to_statistically_interpret_negative_values_in_U_and_Vt>`__
-  `Random
   Projection <https://radimrehurek.com/gensim/models/rpmodel.html>`__
   (used as an option to speed up LSI)

LDA (Model)
~~~~~~~~~~~

-  `Docs <https://radimrehurek.com/gensim/models/ldamodel.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/ldamodel.py>`__
-  `Example with Android issue
   reports <http://christop.club/2014/05/06/using-gensim-for-lda/>`__,
   `Another
   example <https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html>`__,
   `Another
   example <http://brandonrose.org/clustering#Latent-Dirichlet-Allocation>`__

Topic Model Tuning
~~~~~~~~~~~~~~~~~~

-  `Colouring words by topic in a document, print words in a
   topics <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb>`__
-  `Topic Coherence, a metric that correlates that human judgement on
   topic
   quality. <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence_tutorial.ipynb>`__
-  `Compare topics and documents using Jaccard, Kullback-Leibler and
   Hellinger
   similarities <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/tutorials/run_distance_metrics.html>`__
-  `America's Next Topic Model
   slides <https://speakerdeck.com/tmylk/americas-next-topic-model-at-pydata-berlin-august-2016?slide=7>`__
   -- How to choose your next topic model, presented at Pydata Berlin 10
   August 2016 by Lev Konstantinovsky
-  `Classification of News Articles using Topic
   Modeling <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/howtos/run_news_classification.html>`__
-  `LDA: pre-processing and training
   tips <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/tutorials/run_lda.html>`__

Query Similarities
~~~~~~~~~~~~~~~~~~

-  Tool to get the most similar documents for LDA, LSI
-  `Similarity queries
   tutorial <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_similarity_queries.html>`__

Dynamic Topic Modeling
~~~~~~~~~~~~~~~~~~~~~~

-  Model evolution of topics through time
-  `Easy intro to DTM. Evolution of Voldemort topic through the 7 Harry
   Potter
   books. <http://rare-technologies.com/understanding-and-coding-dynamic-topic-models/>`__
-  `Dynamic Topic Modeling and Dynamic Influence Model
   Tutorial <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/dtm_example.ipynb>`__
-  `Python Dynamic Topic Modelling Theory and
   Tutorial <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/ldaseqmodel.ipynb>`__

Word Embeddings
---------------

Word2Vec (Model)
~~~~~~~~~~~~~~~~

-  `Docs <https://radimrehurek.com/gensim/models/word2vec.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py>`__
   (very simple interface)
-  `Simple word2vec
   tutorial <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/tutorials/run_word2vec.html>`__
   (examples of ``most_similar, similarity, doesnt_match``)
-  `Comparison of FastText and
   Word2Vec <https://github.com/RaRe-Technologies/gensim/blob/ba1ce894a5192fc493a865c535202695bb3c0424/docs/notebooks/Word2Vec_FastText_Comparison.ipynb>`__

Doc2Vec (Model)
~~~~~~~~~~~~~~~

-  `Doc2vec Quick Start on Lee
   Corpus <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/tutorials/run_doc2vec_lee.html>`__
-  `Docs <https://radimrehurek.com/gensim/models/doc2vec.html>`__,
   `Source <https://github.com/piskvorky/gensim/blob/develop/gensim/models/doc2vec.py>`__
   (Docs are not very good)
-  Doc2Vec requires a non-standard corpus (need sentiment label for each
   document)
-  `Great illustration of corpus
   preparation <https://linanqiu.github.io/2015/10/07/word2vec-sentiment/>`__,
   `Code <https://github.com/linanqiu/word2vec-sentiments>`__
   (`Alternative <https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1#.nv2lxvbj1>`__,
   `Alternative
   2 <https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis>`__)
-  `Doc2Vec on customer
   review <http://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/>`__
   (example)
-  `Doc2Vec on Airline Tweets Sentiment
   Analysis <https://www.zybuluo.com/HaomingJiang/note/462804>`__
-  `Doc2vec to predict IMDB review star rating. Reproducing the Google
   paper <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/howtos/run_doc2vec_imdb.html>`__

Similarity Queries
~~~~~~~~~~~~~~~~~~

-  `Similarity queries using Annoy with word2vec and
   doc2vec <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/tutorials/run_annoy.html>`__

Word Movers Distance
~~~~~~~~~~~~~~~~~~~~

-  Tool to get the most similar documents for word2vec
-  `Word Movers Distance for Yelp Reviews
   tutorial <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/WMD_tutorial.ipynb>`__

Deep Inverse Regression
~~~~~~~~~~~~~~~~~~~~~~~

-  Document Classification using Bayesian Inversion and several word2vec
   models(one for each class)
-  `Deep Inverse Regression with Yelp
   Reviews <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/deepir.ipynb>`__

Other techniques
----------------

Summarization
~~~~~~~~~~~~~

-  Extract most important keywords and sentences from the text
-  `Tutorial on TextRank
   summarisation <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_similarity_queries.html>`__

Overviews
~~~~~~~~~

-  Tutorial showing API for document classification with various
   techniques: TF-IDF, word2vec averaging, Deep IR, Word Movers Distance
   and doc2vec
-  `Movie plots by
   genre <https://github.com/RaRe-Technologies/movie-plots-by-genre>`__

Videos
======

-  `Radim Řehůřek - Faster than Google? Optimization lessons in
   Python. <https://www.youtube.com/watch?v=vU4TlwZzTfU>`__
-  `MLMU.cz - Radim Řehůřek - Word2vec & friends
   (7.1.2015) <https://www.youtube.com/watch?v=wTp3P2UnTfQ>`__
-  `Making an Impact with
   NLP <https://www.youtube.com/watch?v=oSSnDeOXTZQ>`__ -- Pycon 2016
   Tutorial by Hobsons Lane
-  `NLP with NLTK and
   Gensim <https://www.youtube.com/watch?v=itKNpCPHq3I>`__ -- Pycon 2016
   Tutorial by Tony Ojeda, Benjamin Bengfort, Laura Lorenz from District
   Data Labs
-  `Word Embeddings for Fun and
   Profit <https://www.youtube.com/watch?v=lfqW46u0UKc>`__ -- Talk at
   PyData London 2016 talk by Lev Konstantinovskiy. See accompanying
   `repo <https://github.com/RaRe-Technologies/movie-plots-by-genre>`__

Credits
=======

Based on wonderful
`resource <https://github.com/jxieeducation/DIY-Data-Science/blob/master/frameworks/gensim.md>`__
by Jason Xie.


.. raw:: html

    <div style='clear:both'></div>



.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-gallery


  .. container:: sphx-glr-download

    :download:`Download all examples in Python source code: auto_examples_python.zip <//home/misha/git/gensim-numfocus/docs/src/auto_examples/auto_examples_python.zip>`



  .. container:: sphx-glr-download

    :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip <//home/misha/git/gensim-numfocus/docs/src/auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
