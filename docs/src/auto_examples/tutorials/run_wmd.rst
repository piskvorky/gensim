.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_wmd.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tutorials_run_wmd.py:


Word Mover's Distance
=====================

Demonstrates using Gensim's implemenation of the WMD.

Word Mover's Distance (WMD) is a promising new tool in machine learning that
allows us to submit a query and return the most relevant documents. This
tutorial introduces WMD and shows how you can compute the WMD distance
between two documents using ``wmdistance``.

WMD Basics
----------

WMD enables us to assess the "distance" between two documents in a meaningful
way even when they have no words in common. It uses `word2vec
<http://rare-technologies.com/word2vec-tutorial/>`_ [4] vector embeddings of
words. It been shown to outperform many of the state-of-the-art methods in
k-nearest neighbors classification [3].

WMD is illustrated below for two very similar sentences (illustration taken
from `Vlad Niculae's blog
<http://vene.ro/blog/word-movers-distance-in-python.html>`_). The sentences
have no words in common, but by matching the relevant words, WMD is able to
accurately measure the (dis)similarity between the two sentences. The method
also uses the bag-of-words representation of the documents (simply put, the
word's frequencies in the documents), noted as $d$ in the figure below. The
intuition behind the method is that we find the minimum "traveling distance"
between documents, in other words the most efficient way to "move" the
distribution of document 1 to the distribution of document 2.



.. code-block:: default


    # Image from https://vene.ro/images/wmd-obama.png
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('wmd-obama.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()




.. image:: /auto_examples/tutorials/images/sphx_glr_run_wmd_001.png
    :class: sphx-glr-single-img




This method was introduced in the article "From Word Embeddings To Document
Distances" by Matt Kusner et al. (\ `link to PDF
<http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf>`_\ ). It is inspired
by the "Earth Mover's Distance", and employs a solver of the "transportation
problem".

In this tutorial, we will learn how to use Gensim's WMD functionality, which
consists of the ``wmdistance`` method for distance computation, and the
``WmdSimilarity`` class for corpus based similarity queries.

.. Important::
   If you use Gensim's WMD functionality, please consider citing [1], [2] and [3].

Computing the Word Mover's Distance
-----------------------------------

To use WMD, you need some existing word embeddings.
You could train your own Word2Vec model, but that is beyond the scope of this tutorial
(check out :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py` if you're interested).
For this tutorial, we'll be using an existing Word2Vec model.

Let's take some sentences to compute the distance between.



.. code-block:: default


    # Initialize logging.
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'







These sentences have very similar content, and as such the WMD should be low.
Before we compute the WMD, we want to remove stopwords ("the", "to", etc.),
as these do not contribute a lot to the information in the sentences.



.. code-block:: default


    # Import and download stopwords from NLTK.
    from nltk.corpus import stopwords
    from nltk import download
    download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')

    def preprocess(sentence):
        return [w for w in sentence.lower().split() if w not in stop_words]

    sentence_obama = preprocess(sentence_obama)
    sentence_president = preprocess(sentence_president)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/feature_extraction/image.py:167: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      dtype=np.int):
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:30: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:167: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:284: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_Gram=True, verbose=0,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:862: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:1101: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:1127: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, positive=False):
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:1362: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:1602: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    /home/witiko/.virtualenvs/gensim4/lib/python3.7/site-packages/sklearn/linear_model/least_angle.py:1738: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, positive=False):
    [nltk_data] Downloading package stopwords to /home/witiko/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



Now, as mentioned earlier, we will be using some downloaded pre-trained
embeddings. We load these into a Gensim Word2Vec model class.

.. Important::
  The embeddings we have chosen here require a lot of memory.



.. code-block:: default

    import gensim.downloader as api
    model = api.load('word2vec-google-news-300')







So let's compute WMD using the ``wmdistance`` method.



.. code-block:: default

    distance = model.wmdistance(sentence_obama, sentence_president)
    print('distance = %.4f' % distance)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    distance = 1.0175



Let's try the same thing with two completely unrelated sentences. Notice that the distance is larger.



.. code-block:: default

    sentence_orange = preprocess('Oranges are my favorite fruit')
    distance = model.wmdistance(sentence_obama, sentence_orange)
    print('distance = %.4f' % distance)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    distance = 1.3663



References
----------

1. Ofir Pele and Michael Werman, *A linear time histogram metric for improved SIFT matching*, 2008.
2. Ofir Pele and Michael Werman, *Fast and robust earth mover's distances*, 2009.
3. Matt Kusner et al. *From Embeddings To Document Distances*, 2015.
4. Tomáš Mikolov et al. *Efficient Estimation of Word Representations in Vector Space*, 2013.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  55.983 seconds)

**Estimated memory usage:**  7537 MB


.. _sphx_glr_download_auto_examples_tutorials_run_wmd.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_wmd.py <run_wmd.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_wmd.ipynb <run_wmd.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
