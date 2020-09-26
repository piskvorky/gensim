.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_annoy.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_tutorials_run_annoy.py:


Fast Similarity Queries with Annoy and Word2Vec
===============================================

Introduces the annoy library for similarity queries using a Word2Vec model.


.. code-block:: default


    LOGS = False  # Set to True if you want to see progress in logs.
    if LOGS:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








The `Annoy "Approximate Nearest Neighbors Oh Yeah"
<https://github.com/spotify/annoy>`_ library enables similarity queries with
a Word2Vec model.  The current implementation for finding k nearest neighbors
in a vector space in gensim has linear complexity via brute force in the
number of indexed documents, although with extremely low constant factors.
The retrieved results are exact, which is an overkill in many applications:
approximate results retrieved in sub-linear time may be enough. Annoy can
find approximate nearest neighbors much faster.

Outline
-------

1. Download Text8 Corpus
2. Train the Word2Vec model
3. Construct AnnoyIndex with model & make a similarity query
4. Compare to the traditional indexer
5. Persist indices to disk
6. Save memory by via memory-mapping indices saved to disk
7. Evaluate relationship of ``num_trees`` to initialization time and accuracy
8. Work with Google's word2vec C formats


1. Download Text8 corpus
------------------------


.. code-block:: default

    import gensim.downloader as api
    text8_path = api.load('text8', return_path=True)
    print("Using corpus from", text8_path)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using corpus from /Users/kofola3/gensim-data/text8/text8.gz




2. Train the Word2Vec model
---------------------------

For more details, see :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py`.


.. code-block:: default

    from gensim.models import Word2Vec, KeyedVectors
    from gensim.models.word2vec import Text8Corpus

    # Using params from Word2Vec_FastText_Comparison
    params = {
        'alpha': 0.05,
        'vector_size': 100,
        'window': 5,
        'epochs': 5,
        'min_count': 5,
        'sample': 1e-4,
        'sg': 1,
        'hs': 0,
        'negative': 5
    }
    model = Word2Vec(Text8Corpus(text8_path), **params)
    print("Using model", model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using model Word2Vec(vocab=71290, size=100, alpha=0.05)




3. Construct AnnoyIndex with model & make a similarity query
------------------------------------------------------------

An instance of ``AnnoyIndexer`` needs to be created in order to use Annoy in gensim. The ``AnnoyIndexer`` class is located in ``gensim.similarities.annoy``.

``AnnoyIndexer()`` takes two parameters:

* **model**: A ``Word2Vec`` or ``Doc2Vec`` model.
* **num_trees**: A positive integer. ``num_trees`` effects the build
  time and the index size. **A larger value will give more accurate results,
  but larger indexes**. More information on what trees in Annoy do can be found
  `here <https://github.com/spotify/annoy#how-does-it-work>`__. The relationship
  between ``num_trees``\ , build time, and accuracy will be investigated later
  in the tutorial.

Now that we are ready to make a query, lets find the top 5 most similar words
to "science" in the Text8 corpus. To make a similarity query we call
``Word2Vec.most_similar`` like we would traditionally, but with an added
parameter, ``indexer``.

Apart from Annoy, Gensim also supports the NMSLIB indexer. NMSLIB is a similar library to
Annoy – both support fast, approximate searches for similar vectors.



.. code-block:: default

    from gensim.similarities.annoy import AnnoyIndexer

    # 100 trees are being used in this example
    annoy_index = AnnoyIndexer(model, 100)
    # Derive the vector for the word "science" in our model
    vector = model.wv["science"]
    # The instance of AnnoyIndexer we just created is passed
    approximate_neighbors = model.wv.most_similar([vector], topn=11, indexer=annoy_index)
    # Neatly print the approximate_neighbors and their corresponding cosine similarity values
    print("Approximate Neighbors")
    for neighbor in approximate_neighbors:
        print(neighbor)

    normal_neighbors = model.wv.most_similar([vector], topn=11)
    print("\nExact Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Approximate Neighbors
    ('science', 0.9998273665260058)
    ('astrobiology', 0.5996885895729065)
    ('psychohistory', 0.5911669135093689)
    ('actuarial', 0.5909044742584229)
    ('sciences', 0.587712824344635)
    ('astronautics', 0.5794282257556915)
    ('scientific', 0.5745473206043243)
    ('nanomedicine', 0.5686345398426056)
    ('climatologists', 0.5657966732978821)
    ('buzan', 0.565343976020813)
    ('sensationalism', 0.5640498399734497)

    Exact Neighbors
    ('science', 1.0000001192092896)
    ('fiction', 0.7536318302154541)
    ('astrobiology', 0.6795015335083008)
    ('psychohistory', 0.665710985660553)
    ('actuarial', 0.6652816534042358)
    ('vinge', 0.6611783504486084)
    ('sciences', 0.6600385904312134)
    ('popularizer', 0.6583578586578369)
    ('xenobiology', 0.6504358053207397)
    ('vernor', 0.6491150259971619)
    ('astronautics', 0.6462388634681702)




The closer the cosine similarity of a vector is to 1, the more similar that
word is to our query, which was the vector for "science". There are some
differences in the ranking of similar words and the set of words included
within the 10 most similar words.

4. Compare to the traditional indexer
-------------------------------------


.. code-block:: default


    # Set up the model and vector that we are using in the comparison
    annoy_index = AnnoyIndexer(model, 100)

    # Dry run to make sure both indexes are fully in RAM
    normed_vectors = model.wv.get_normed_vectors()
    vector = normed_vectors[0]
    model.wv.most_similar([vector], topn=5, indexer=annoy_index)
    model.wv.most_similar([vector], topn=5)

    import time
    import numpy as np

    def avg_query_time(annoy_index=None, queries=1000):
        """Average query time of a most_similar method over 1000 random queries."""
        total_time = 0
        for _ in range(queries):
            rand_vec = normed_vectors[np.random.randint(0, len(model.wv))]
            start_time = time.process_time()
            model.wv.most_similar([rand_vec], topn=5, indexer=annoy_index)
            total_time += time.process_time() - start_time
        return total_time / queries

    queries = 1000

    gensim_time = avg_query_time(queries=queries)
    annoy_time = avg_query_time(annoy_index, queries=queries)
    print("Gensim (s/query):\t{0:.5f}".format(gensim_time))
    print("Annoy (s/query):\t{0:.5f}".format(annoy_time))
    speed_improvement = gensim_time / annoy_time
    print ("\nAnnoy is {0:.2f} times faster on average on this particular run".format(speed_improvement))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Gensim (s/query):       0.00635
    Annoy (s/query):        0.00054

    Annoy is 11.75 times faster on average on this particular run




**This speedup factor is by no means constant** and will vary greatly from
run to run and is particular to this data set, BLAS setup, Annoy
parameters(as tree size increases speedup factor decreases), machine
specifications, among other factors.

.. Important::
   Initialization time for the annoy indexer was not included in the times.
   The optimal knn algorithm for you to use will depend on how many queries
   you need to make and the size of the corpus. If you are making very few
   similarity queries, the time taken to initialize the annoy indexer will be
   longer than the time it would take the brute force method to retrieve
   results. If you are making many queries however, the time it takes to
   initialize the annoy indexer will be made up for by the incredibly fast
   retrieval times for queries once the indexer has been initialized

.. Important::
   Gensim's 'most_similar' method is using numpy operations in the form of
   dot product whereas Annoy's method isnt. If 'numpy' on your machine is
   using one of the BLAS libraries like ATLAS or LAPACK, it'll run on
   multiple cores (only if your machine has multicore support ). Check `SciPy
   Cookbook
   <http://scipy-cookbook.readthedocs.io/items/ParallelProgramming.html>`_
   for more details.


5. Persisting indices to disk
-----------------------------

You can save and load your indexes from/to disk to prevent having to
construct them each time. This will create two files on disk, *fname* and
*fname.d*. Both files are needed to correctly restore all attributes. Before
loading an index, you will have to create an empty AnnoyIndexer object.



.. code-block:: default

    fname = '/tmp/mymodel.index'

    # Persist index to disk
    annoy_index.save(fname)

    # Load index back
    import os.path
    if os.path.exists(fname):
        annoy_index2 = AnnoyIndexer()
        annoy_index2.load(fname)
        annoy_index2.model = model

    # Results should be identical to above
    vector = model.wv["science"]
    approximate_neighbors2 = model.wv.most_similar([vector], topn=11, indexer=annoy_index2)
    for neighbor in approximate_neighbors2:
        print(neighbor)

    assert approximate_neighbors == approximate_neighbors2





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('science', 0.9998273665260058)
    ('astrobiology', 0.5996885895729065)
    ('psychohistory', 0.5911669135093689)
    ('actuarial', 0.5909044742584229)
    ('sciences', 0.587712824344635)
    ('astronautics', 0.5794282257556915)
    ('scientific', 0.5745473206043243)
    ('nanomedicine', 0.5686345398426056)
    ('climatologists', 0.5657966732978821)
    ('buzan', 0.565343976020813)
    ('sensationalism', 0.5640498399734497)




Be sure to use the same model at load that was used originally, otherwise you
will get unexpected behaviors.


6. Save memory via memory-mapping indexes saved to disk
-------------------------------------------------------

Annoy library has a useful feature that indices can be memory-mapped from
disk. It saves memory when the same index is used by several processes.

Below are two snippets of code. First one has a separate index for each
process. The second snipped shares the index between two processes via
memory-mapping. The second example uses less total RAM as it is shared.



.. code-block:: default


    # Remove verbosity from code below (if logging active)
    if LOGS:
        logging.disable(logging.CRITICAL)

    from multiprocessing import Process
    import os
    import psutil








Bad example: two processes load the Word2vec model from disk and create their
own Annoy index from that model.



.. code-block:: default


    model.save('/tmp/mymodel.pkl')

    def f(process_id):
        print('Process Id: {}'.format(os.getpid()))
        process = psutil.Process(os.getpid())
        new_model = Word2Vec.load('/tmp/mymodel.pkl')
        vector = new_model.wv["science"]
        annoy_index = AnnoyIndexer(new_model,100)
        approximate_neighbors = new_model.wv.most_similar([vector], topn=5, indexer=annoy_index)
        print('\nMemory used by process {}: {}\n---'.format(os.getpid(), process.memory_info()))

    # Creating and running two parallel process to share the same index file.
    p1 = Process(target=f, args=('1',))
    p1.start()
    p1.join()
    p2 = Process(target=f, args=('2',))
    p2.start()
    p2.join()








Good example: two processes load both the Word2vec model and index from disk
and memory-map the index.



.. code-block:: default


    model.save('/tmp/mymodel.pkl')

    def f(process_id):
        print('Process Id: {}'.format(os.getpid()))
        process = psutil.Process(os.getpid())
        new_model = Word2Vec.load('/tmp/mymodel.pkl')
        vector = new_model.wv["science"]
        annoy_index = AnnoyIndexer()
        annoy_index.load('/tmp/mymodel.index')
        annoy_index.model = new_model
        approximate_neighbors = new_model.wv.most_similar([vector], topn=5, indexer=annoy_index)
        print('\nMemory used by process {}: {}\n---'.format(os.getpid(), process.memory_info()))

    # Creating and running two parallel process to share the same index file.
    p1 = Process(target=f, args=('1',))
    p1.start()
    p1.join()
    p2 = Process(target=f, args=('2',))
    p2.start()
    p2.join()








7. Evaluate relationship of ``num_trees`` to initialization time and accuracy
-----------------------------------------------------------------------------



.. code-block:: default

    import matplotlib.pyplot as plt








Build dataset of Initialization times and accuracy measures:



.. code-block:: default


    exact_results = [element[0] for element in model.wv.most_similar([normed_vectors[0]], topn=100)]

    x_values = []
    y_values_init = []
    y_values_accuracy = []

    for x in range(1, 300, 10):
        x_values.append(x)
        start_time = time.time()
        annoy_index = AnnoyIndexer(model, x)
        y_values_init.append(time.time() - start_time)
        approximate_results = model.wv.most_similar([normed_vectors[0]], topn=100, indexer=annoy_index)
        top_words = [result[0] for result in approximate_results]
        y_values_accuracy.append(len(set(top_words).intersection(exact_results)))








Plot results:


.. code-block:: default


    plt.figure(1, figsize=(12, 6))
    plt.subplot(121)
    plt.plot(x_values, y_values_init)
    plt.title("num_trees vs initalization time")
    plt.ylabel("Initialization time (s)")
    plt.xlabel("num_trees")
    plt.subplot(122)
    plt.plot(x_values, y_values_accuracy)
    plt.title("num_trees vs accuracy")
    plt.ylabel("% accuracy")
    plt.xlabel("num_trees")
    plt.tight_layout()
    plt.show()




.. image:: /auto_examples/tutorials/images/sphx_glr_run_annoy_001.png
    :alt: num_trees vs initalization time, num_trees vs accuracy
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/work/workspace/vew/gensim3.6/lib/python3.6/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())




From the above, we can see that the initialization time of the annoy indexer
increases in a linear fashion with num_trees. Initialization time will vary
from corpus to corpus, in the graph above the lee corpus was used

Furthermore, in this dataset, the accuracy seems logarithmically related to
the number of trees. We see an improvement in accuracy with more trees, but
the relationship is nonlinear.


7. Work with Google word2vec files
----------------------------------

Our model can be exported to a word2vec C format. There is a binary and a
plain text word2vec format. Both can be read with a variety of other
software, or imported back into gensim as a ``KeyedVectors`` object.



.. code-block:: default


    # To export our model as text
    model.wv.save_word2vec_format('/tmp/vectors.txt', binary=False)

    from smart_open import open
    # View the first 3 lines of the exported file

    # The first line has the total number of entries and the vector dimension count.
    # The next lines have a key (a string) followed by its vector.
    with open('/tmp/vectors.txt') as myfile:
        for i in range(3):
            print(myfile.readline().strip())

    # To import a word2vec text model
    wv = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)

    # To export our model as binary
    model.wv.save_word2vec_format('/tmp/vectors.bin', binary=True)

    # To import a word2vec binary model
    wv = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)

    # To create and save Annoy Index from a loaded `KeyedVectors` object (with 100 trees)
    annoy_index = AnnoyIndexer(wv, 100)
    annoy_index.save('/tmp/mymodel.index')

    # Load and test the saved word vectors and saved annoy index
    wv = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)
    annoy_index = AnnoyIndexer()
    annoy_index.load('/tmp/mymodel.index')
    annoy_index.model = wv

    vector = wv["cat"]
    approximate_neighbors = wv.most_similar([vector], topn=11, indexer=annoy_index)
    # Neatly print the approximate_neighbors and their corresponding cosine similarity values
    print("Approximate Neighbors")
    for neighbor in approximate_neighbors:
        print(neighbor)

    normal_neighbors = wv.most_similar([vector], topn=11)
    print("\nExact Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    71290 100
    the 0.0024114326 -0.01658697 -0.03555953 -0.10153896 0.18787915 -0.1635462 -0.28673044 -0.041369613 -0.19595902 0.07011118 0.056620464 0.02271993 -0.11009094 0.04158417 0.020375583 -0.084979765 -0.07266931 -0.076348744 -0.025672905 0.0014125895 0.049288113 -0.113621734 -0.054461427 -0.3548957 -0.16930902 0.14919026 -0.1359916 0.11747352 -0.10112707 -0.10097838 0.03537573 -0.11102786 0.09662876 0.23934129 0.03430543 0.031142373 -0.10773377 0.18936938 0.16569573 -0.12907895 0.19478081 -0.031971604 -0.22358014 0.010947946 0.22119588 -0.034503784 -0.10515277 -0.0149329 0.03839723 -0.08875417 0.020486698 0.121592954 0.22152981 0.35021618 0.015179496 0.18801002 -0.08069974 0.38830653 0.20972545 0.027219895 0.09098711 -0.15508457 0.20050901 -0.083856024 -0.092396446 -0.004771586 0.16696249 -0.20392004 -0.021265335 0.05848724 0.042290248 -0.23386984 0.10066098 0.24922445 0.33556217 -0.16766794 0.057491954 0.18030043 0.009967926 -0.03610459 -0.18041357 0.08075151 -0.10740809 -0.07137133 -0.06789198 -0.035547882 -0.2261159 -0.11013532 0.144337 -0.11964697 0.052678265 0.08476612 -0.09595462 -0.06770006 0.07560474 -0.4369961 -0.08699813 -0.24849728 -0.13127735 0.10296198
    of 0.15834092 0.06914118 -0.10107699 -0.118953355 0.23389126 -0.10860545 -0.22906052 -0.14216071 -0.12289489 -0.0012629399 0.074761786 -0.06659 -0.1430411 -0.014833263 -0.16868858 -0.007957721 -0.14431834 -0.057504084 -0.10158825 -0.09107199 -0.05050294 -0.17399603 -0.104178846 -0.37724152 -0.17095697 0.045356687 -0.1641648 0.049223717 -0.00488149 -0.020546727 0.10161075 -0.09690393 -0.06261461 0.22022204 -0.0804387 0.055082984 -0.10739114 0.23073868 0.1234579 0.09754846 0.18950391 -0.12133284 -0.369799 0.11367094 0.076141596 -0.21988852 -0.016046293 0.05841635 0.13228759 -0.09812193 0.015954286 0.2045245 0.27682808 0.26974434 -0.03128133 0.15385458 -0.26725027 0.2796234 0.24427672 0.026197268 -0.10170346 -0.16725655 0.16234653 -0.015511761 -0.14291462 0.042616863 0.16123989 -0.24213533 0.108378075 0.16561604 -0.07071194 -0.14839227 0.032803435 0.3410274 0.36961436 -0.01081987 -0.094509445 0.17811705 0.061423916 0.12526047 -0.21204133 0.04394304 -0.002062126 -0.05939061 0.08941769 -0.10346588 -0.1656505 0.0004983001 0.09247532 -0.09488792 0.17639682 -0.013249175 -0.22949116 -0.1667382 0.050704297 -0.46825656 -0.059015192 -0.23415534 0.10704609 -0.032225966
    Approximate Neighbors
    ('cat', 1.0)
    ('purr', 0.5825351476669312)
    ('prionailurus', 0.5775779187679291)
    ('cats', 0.5702914297580719)
    ('meow', 0.5674391090869904)
    ('rabbits', 0.5620074272155762)
    ('sighthound', 0.561018168926239)
    ('saimiri', 0.5574325621128082)
    ('kitten', 0.5556513369083405)
    ('eared', 0.5553555190563202)
    ('badger', 0.5542813241481781)

    Exact Neighbors
    ('cat', 0.9999999403953552)
    ('leopardus', 0.6718246340751648)
    ('purr', 0.6514462232589722)
    ('felis', 0.6486470103263855)
    ('prionailurus', 0.6431191563606262)
    ('cats', 0.6307010054588318)
    ('asinus', 0.6302586197853088)
    ('meow', 0.6257820129394531)
    ('oncifelis', 0.6220626831054688)
    ('rabbits', 0.6163250207901001)
    ('lynxes', 0.6148800253868103)




Recap
-----

In this notebook we used the Annoy module to build an indexed approximation
of our word embeddings. To do so, we did the following steps:

1. Download Text8 Corpus
2. Train Word2Vec Model
3. Construct AnnoyIndex with model & make a similarity query
4. Persist indices to disk
5. Save memory by via memory-mapping indices saved to disk
6. Evaluate relationship of ``num_trees`` to initialization time and accuracy
7. Work with Google's word2vec C formats



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 14 minutes  40.672 seconds)

**Estimated memory usage:**  753 MB


.. _sphx_glr_download_auto_examples_tutorials_run_annoy.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_annoy.py <run_annoy.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_annoy.ipynb <run_annoy.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
