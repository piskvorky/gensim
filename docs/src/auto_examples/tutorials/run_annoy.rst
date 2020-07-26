.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_annoy.py>` to download the full example code
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
    ('science', 0.9998779296875)
    ('actuarial', 0.5997203588485718)
    ('multidisciplinary', 0.5991933047771454)
    ('sciences', 0.5958340764045715)
    ('scientific', 0.5955467224121094)
    ('astrobiology', 0.5923857390880585)
    ('popularizer', 0.5910311937332153)
    ('nanomedicine', 0.5902301669120789)
    ('astronautics', 0.5890269577503204)
    ('poststructuralism', 0.5872606933116913)
    ('crichton', 0.5863060653209686)

    Exact Neighbors
    ('science', 1.0)
    ('fiction', 0.7604568004608154)
    ('actuarial', 0.679552435874939)
    ('multidisciplinary', 0.678708016872406)
    ('sciences', 0.6732996702194214)
    ('scientific', 0.672835111618042)
    ('astrobiology', 0.6677011847496033)
    ('popularizer', 0.6654890179634094)
    ('nanomedicine', 0.6641773581504822)
    ('astronautics', 0.6622022986412048)
    ('poststructuralism', 0.6592925786972046)



The closer the cosine similarity of a vector is to 1, the more similar that
word is to our query, which was the vector for "science". There are some
differences in the ranking of similar words and the set of words included
within the 10 most similar words.

4. Compare to the traditional indexer
-------------------------------------


.. code-block:: default


    # Set up the model and vector that we are using in the comparison
    model.init_sims()
    annoy_index = AnnoyIndexer(model, 100)

    # Dry run to make sure both indexes are fully in RAM
    vector = model.wv.vectors_norm[0]
    model.wv.most_similar([vector], topn=5, indexer=annoy_index)
    model.wv.most_similar([vector], topn=5)

    import time
    import numpy as np

    def avg_query_time(annoy_index=None, queries=1000):
        """Average query time of a most_similar method over 1000 random queries."""
        total_time = 0
        for _ in range(queries):
            rand_vec = model.wv.vectors_norm[np.random.randint(0, len(model.wv))]
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

    Gensim (s/query):       0.00879
    Annoy (s/query):        0.00036

    Annoy is 24.36 times faster on average on this particular run



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

    ('science', 0.9998779296875)
    ('actuarial', 0.5997203588485718)
    ('multidisciplinary', 0.5991933047771454)
    ('sciences', 0.5958340764045715)
    ('scientific', 0.5955467224121094)
    ('astrobiology', 0.5923857390880585)
    ('popularizer', 0.5910311937332153)
    ('nanomedicine', 0.5902301669120789)
    ('astronautics', 0.5890269577503204)
    ('poststructuralism', 0.5872606933116913)
    ('crichton', 0.5863060653209686)



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


    exact_results = [element[0] for element in model.wv.most_similar([model.wv.vectors_norm[0]], topn=100)]

    x_values = []
    y_values_init = []
    y_values_accuracy = []

    for x in range(1, 300, 10):
        x_values.append(x)
        start_time = time.time()
        annoy_index = AnnoyIndexer(model, x)
        y_values_init.append(time.time() - start_time)
        approximate_results = model.wv.most_similar([model.wv.vectors_norm[0]], topn=100, indexer=annoy_index)
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
    the 0.040282175 0.2165622 -0.11749034 -0.07902362 0.051659793 -0.19823891 0.06806079 0.13111936 0.06739034 0.28645086 -0.05745192 -0.26289585 0.22901897 0.16851662 0.06712559 0.074796304 -0.035081368 -0.16047138 0.21728243 0.054436874 0.15153275 0.04205451 0.25825807 0.06590562 0.003011158 0.004102882 -0.100709945 -0.43598634 0.06541784 0.1916927 -0.031570192 0.29551303 -0.09695051 0.16501299 0.19717866 -0.058377385 -0.28637803 0.05174591 -0.16937277 0.059432484 -0.10389606 -0.20887667 -0.05919355 0.04137692 -0.15064004 -0.09648997 -0.22527762 0.033556122 -0.101461716 0.02389651 -0.08515353 0.042458646 -0.17026003 -0.08010368 -0.06399739 -0.078073114 -0.21165106 0.19857563 0.16560182 -0.17774524 -0.27337983 -0.05858462 0.3014273 0.25362 -0.19963813 0.03708622 0.2326357 0.1216157 -0.008749145 -0.036809318 -0.049642608 -0.15324257 0.05383394 0.47451496 -0.4227195 -0.31459892 -0.15150371 -0.14932543 0.063661754 -0.10880057 -0.021235414 0.02103542 -0.16286005 0.09865164 -0.02635211 0.11226083 -0.11042501 0.3655904 -0.12234078 -0.1566903 0.02300252 -0.030776313 0.22856043 0.0644399 0.11757082 -0.08796379 0.23301896 0.018121734 -0.060471278 0.013380485
    of -0.117521346 0.25582337 0.099465266 -0.08187441 0.0003451583 -0.3138303 0.017125094 -0.02423271 0.008411589 0.23745868 -0.030340558 -0.0042213113 0.16474192 0.17288761 0.18377638 0.09036244 -0.20894344 -0.044563673 0.14401074 -0.0023740537 0.069194905 0.16009597 0.3646139 0.04876363 0.04529528 0.04463972 -0.11413204 -0.46372798 0.09147736 0.19938193 -0.0743411 0.3425023 -0.1404897 0.13559413 0.091858536 -0.046490498 -0.07325774 0.036205 -0.13261133 0.060684923 -0.2155932 -0.18659447 0.06216802 0.15724409 -0.0615914 -0.1047893 -0.21563342 0.03399876 -0.14627206 0.13622384 -0.038415514 0.040357746 -0.0950334 -0.1240751 -0.025667595 -0.0762319 -0.19063687 0.42665502 0.26953915 -0.095091894 0.033522405 -0.06997937 0.32336366 0.17219204 -0.17910816 -0.17552538 0.15011984 0.005003567 0.14006145 0.0067128018 -0.0964597 -0.15371965 0.07887949 0.38889432 -0.41947517 -0.14448255 -0.39335525 -0.14048906 0.05884099 -0.0022921925 -0.10194997 0.20200975 -0.20999992 0.12407939 -0.084994085 0.13595223 -0.16718066 0.3076714 -0.1003335 -0.13627231 0.15710355 -0.097152695 0.18041256 -0.013511332 -0.06089443 -0.02832937 0.1863955 0.094825216 0.073157795 0.030848853
    Approximate Neighbors
    ('cat', 1.0)
    ('polydactyl', 0.5950324535369873)
    ('kitten', 0.5930328965187073)
    ('sighthound', 0.5881928503513336)
    ('leopardus', 0.585923820734024)
    ('prionailurus', 0.5840010941028595)
    ('pug', 0.5792734324932098)
    ('cats', 0.5770905315876007)
    ('eared', 0.5770178437232971)
    ('badger', 0.5747911930084229)
    ('albino', 0.5721485614776611)

    Exact Neighbors
    ('cat', 1.0)
    ('polydactyl', 0.6720025539398193)
    ('kitten', 0.6687554717063904)
    ('meow', 0.6622823476791382)
    ('sighthound', 0.6608296632766724)
    ('leopardus', 0.6570817828178406)
    ('prionailurus', 0.6538898348808289)
    ('pug', 0.6459784507751465)
    ('proboscis', 0.6438583731651306)
    ('cats', 0.6422951221466064)
    ('eared', 0.6421721577644348)



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

   **Total running time of the script:** ( 15 minutes  0.569 seconds)

**Estimated memory usage:**  794 MB


.. _sphx_glr_download_auto_examples_tutorials_run_annoy.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_annoy.py <run_annoy.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_annoy.ipynb <run_annoy.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
