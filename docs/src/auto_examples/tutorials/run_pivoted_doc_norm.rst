.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_pivoted_doc_norm.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_tutorials_run_pivoted_doc_norm.py:


Pivoted Document Length Normalization
=====================================

This tutorial demonstrates using Pivoted Document Length Normalization to
counter the effect of short document bias when working with TfIdf, thereby
increasing classification accuracy.

In many cases, normalizing the tfidf weights for each term favors weights of terms
of the documents with shorter length. The *pivoted document length normalization* scheme
counters the effect of this bias for short documents, by making tfidf independent of the document length.

This is achieved by *tilting* the normalization curve along a pivot point and slope, which
must be defined by the user.

Roughly following the equation:

``pivoted_norm = (1 - slope) * pivot + slope * old_norm``

This scheme is proposed in the paper `Pivoted Document Length Normalization <http://singhal.info/pivoted-dln.pdf>`_
by Singhal, Buckley and Mitra.

Overall this approach can increase the accuracy of the model where document lengths are hugely varying across the corpus.

Introduction
------------

This guide demonstrates how to perform pivoted document length normalization.

We will train a logistic regression model to distinguish between text from two different newsgroups.

Our results will show that using pivoted document length normalization yields a better model (higher classification accuracy).



.. code-block:: default


    #
    # Download our dataset
    #
    import gensim.downloader as api
    nws = api.load("20-newsgroups")

    #
    # Pick texts from relevant newsgroups, split into training and test set.
    #
    cat1, cat2 = ('sci.electronics', 'sci.space')

    #
    # X_* contain the actual texts as strings.
    # Y_* contain labels, 0 for cat1 (sci.electronics) and 1 for cat2 (sci.space)
    #
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in nws:
        if i["set"] == "train" and i["topic"] == cat1:
            X_train.append(i["data"])
            y_train.append(0)
        elif i["set"] == "train" and i["topic"] == cat2:
            X_train.append(i["data"])
            y_train.append(1)
        elif i["set"] == "test" and i["topic"] == cat1:
            X_test.append(i["data"])
            y_test.append(0)
        elif i["set"] == "test" and i["topic"] == cat2:
            X_test.append(i["data"])
            y_test.append(1)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/work/workspace/vew/gensim3.6/lib/python3.6/site-packages/smart_open/smart_open_lib.py:254: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
      'See the migration notes for details: %s' % _MIGRATION_NOTES_URL




Preprocess the data



.. code-block:: default

    from gensim.parsing.preprocessing import preprocess_string
    from gensim.corpora import Dictionary

    id2word = Dictionary([preprocess_string(doc) for doc in X_train])
    train_corpus = [id2word.doc2bow(preprocess_string(doc)) for doc in X_train]
    test_corpus = [id2word.doc2bow(preprocess_string(doc)) for doc in X_test]

    print(len(X_train), len(X_test))

    # We perform our analysis on top k documents which is almost top 10% most scored documents
    k = len(X_test) // 10





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1184 787




Prepare our evaluation function



.. code-block:: default

    from gensim.sklearn_api.tfidf import TfIdfTransformer
    from sklearn.linear_model import LogisticRegression
    from gensim.matutils import corpus2csc

    def get_tfidf_scores(kwargs):
        """
        Return a model's accuracy along with individual document probability values, using
        Gensim's TfIdfTransformer and sklearn's LogisticRegression.

        """
        tfidf_transformer = TfIdfTransformer(**kwargs).fit(train_corpus)

        X_train_tfidf = corpus2csc(tfidf_transformer.transform(train_corpus), num_terms=len(id2word)).T
        X_test_tfidf = corpus2csc(tfidf_transformer.transform(test_corpus), num_terms=len(id2word)).T

        clf = LogisticRegression().fit(X_train_tfidf, y_train)

        model_accuracy = clf.score(X_test_tfidf, y_test)
        doc_scores = clf.decision_function(X_test_tfidf)

        return model_accuracy, doc_scores








Get TFIDF scores for corpus without pivoted document length normalisation
-------------------------------------------------------------------------



.. code-block:: default

    params = {}
    model_accuracy, doc_scores = get_tfidf_scores(params)
    print(model_accuracy)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9682337992376112




Examine the bias towards shorter documents


.. code-block:: default

    import numpy as np

    # Sort the document scores by their scores and return a sorted list
    # of document score and corresponding document lengths.
    def sort_length_by_score(doc_scores, X_test):
        doc_scores = sorted(enumerate(doc_scores), key=lambda x: x[1])
        doc_leng = np.empty(len(doc_scores))

        ds = np.empty(len(doc_scores))

        for i, _ in enumerate(doc_scores):
            doc_leng[i] = len(X_test[_[0]])
            ds[i] = _[1]

        return ds, doc_leng


    print(
        f"Normal cosine normalisation favors short documents as our top {k} docs have a smaller "
        f"mean doc length of {sort_length_by_score(doc_scores, X_test)[1][:k].mean():.3f} "
        f"compared to the corpus mean doc length of {sort_length_by_score(doc_scores, X_test)[1].mean():.3f}"
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Normal cosine normalisation favors short documents as our top 78 docs have a smaller mean doc length of 1668.179 compared to the corpus mean doc length of 1577.799




Get TFIDF scores for corpus with pivoted document length normalisation
----------------------------------------------------------------------

Test various values of alpha (slope) and pick the best one.


.. code-block:: default

    best_model_accuracy = 0
    optimum_slope = 0
    for slope in np.arange(0, 1.1, 0.1):
        params = {"pivot": 10, "slope": slope}

        model_accuracy, doc_scores = get_tfidf_scores(params)

        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            optimum_slope = slope

        print(f"Score for slope {slope} is {model_accuracy}")

    print(f"We get best score of {best_model_accuracy} at slope {optimum_slope}")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Score for slope 0.0 is 0.9720457433290979
    Score for slope 0.1 is 0.9758576874205845
    Score for slope 0.2 is 0.97712833545108
    Score for slope 0.30000000000000004 is 0.9783989834815756
    Score for slope 0.4 is 0.97712833545108
    Score for slope 0.5 is 0.9758576874205845
    Score for slope 0.6000000000000001 is 0.9733163913595934
    Score for slope 0.7000000000000001 is 0.9733163913595934
    Score for slope 0.8 is 0.9733163913595934
    Score for slope 0.9 is 0.9733163913595934
    Score for slope 1.0 is 0.9682337992376112
    We get best score of 0.9783989834815756 at slope 0.30000000000000004




Evaluate the model with optimum slope



.. code-block:: default

    params = {"pivot": 10, "slope": optimum_slope}
    model_accuracy, doc_scores = get_tfidf_scores(params)
    print(model_accuracy)

    print(
        f"With pivoted normalisation top {k} docs have a mean length of "
        f"{sort_length_by_score(doc_scores, X_test)[1][:k].mean():.3f} which is much "
        f"closer to the corpus mean doc length of {sort_length_by_score(doc_scores, X_test)[1].mean():.3f}"
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9783989834815756
    With pivoted normalisation top 78 docs have a mean length of 2077.346 which is much closer to the corpus mean doc length of 1577.799




Visualizing the pivoted normalization
-------------------------------------

From the plot we can see that when the slope was 1 (i.e. when pivoted normalisation
was not applied at all), short documents with length of around 500 had very good scores.
This is a bias for short documents. As we varied the value of slope from 1 to 0
we introdcued a new bias for long documents to counter the bias caused by
cosine normalisation. At a certain point we got an optimum value of
slope (0.5 here) where the overall accuracy of the model was maximized.



.. code-block:: default

    import matplotlib.pyplot as py

    best_model_accuracy = 0
    optimum_slope = 0

    w = 2
    h = 2
    f, axarr = py.subplots(h, w, figsize=(15, 7))

    it = 0
    for slope in [1, 0.2]:
        params = {"pivot": 10, "slope": slope}

        model_accuracy, doc_scores = get_tfidf_scores(params)

        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            optimum_slope = slope

        doc_scores, doc_leng = sort_length_by_score(doc_scores, X_test)

        y = abs(doc_scores[:k, np.newaxis])
        x = doc_leng[:k, np.newaxis]

        py.subplot(1, 2, it+1).bar(x, y, width=20, linewidth=0)
        py.title(f"Slope = {slope} Model accuracy = {model_accuracy}")
        py.ylim([0, 4.5])
        py.xlim([0, 3200])
        py.xlabel("document length")
        py.ylabel("confidence score")

        it += 1

    py.tight_layout()
    py.show()




.. image:: /auto_examples/tutorials/images/sphx_glr_run_pivoted_doc_norm_001.png
    :alt: Slope = 1 Model accuracy = 0.9682337992376112, Slope = 0.2 Model accuracy = 0.97712833545108
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/work/workspace/vew/gensim3.6/lib/python3.6/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())




The above histogram plot helps us visualize the effect of ``slope``. For top
k documents we have document length on the x axis and their respective scores
of belonging to a specific class on y axis.

As we decrease the slope the density of bins is shifted from low document
length (around ~250-500) to over ~500 document length. This suggests that the
positive biasness which was seen at ``slope=1`` (or when regular tfidf was
used) for short documents is now reduced. We get the optimum slope or the max
model accuracy when slope is 0.2.

Conclusion
==========

Using pivoted document normalization improved the classification accuracy a little bit:

* Before (slope=1, identical to default cosine normalization): 0.9682
* After (slope=0.2): 0.9771



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  27.237 seconds)

**Estimated memory usage:**  44 MB


.. _sphx_glr_download_auto_examples_tutorials_run_pivoted_doc_norm.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_pivoted_doc_norm.py <run_pivoted_doc_norm.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_pivoted_doc_norm.ipynb <run_pivoted_doc_norm.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
