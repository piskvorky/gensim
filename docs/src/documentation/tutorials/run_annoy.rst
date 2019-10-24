.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_annoy.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tutorials_run_annoy.py:


Similarity Queries with Annoy and Word2Vec
==========================================

Introduces the annoy library for similarity queries using a Word2Vec model.

.. code-block:: default


    LOGS = False
    if LOGS:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)







The `Annoy Approximate Nearest Neighbors Oh Yeah
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
    text8_path







2. Train the Word2Vec model
---------------------------

For more details, see :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py`.


.. code-block:: default

    from gensim.models import Word2Vec, KeyedVectors
    from gensim.models.word2vec import Text8Corpus

    # Using params from Word2Vec_FastText_Comparison
    params = {
        'alpha': 0.05,
        'size': 100,
        'window': 5,
        'iter': 5,
        'min_count': 5,
        'sample': 1e-4,
        'sg': 1,
        'hs': 0,
        'negative': 5
    }
    model = Word2Vec(Text8Corpus(text8_path), **params)
    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Word2Vec(vocab=71290, size=100, alpha=0.05)


3. Construct AnnoyIndex with model & make a similarity query
------------------------------------------------------------

An instance of ``AnnoyIndexer`` needs to be created in order to use Annoy in gensim. The ``AnnoyIndexer`` class is located in ``gensim.similarities.index``

``AnnoyIndexer()`` takes two parameters:

* **model**: A ``Word2Vec`` or ``Doc2Vec`` model
* **num_trees**: A positive integer. ``num_trees`` effects the build
  time and the index size. **A larger value will give more accurate results,
  but larger indexes**. More information on what trees in Annoy do can be found
  `here <https://github.com/spotify/annoy#how-does-it-work>`__. The relationship
  between ``num_trees``\ , build time, and accuracy will be investigated later
  in the tutorial. 

Now that we are ready to make a query, lets find the top 5 most similar words
to "science" in the Text8 corpus. To make a similarity query we call
``Word2Vec.most_similar`` like we would traditionally, but with an added
parameter, ``indexer``. The only supported indexer in gensim as of now is
Annoy. 



.. code-block:: default

    from gensim.similarities.index import AnnoyIndexer

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
    print("\nNormal (not Annoy-indexed) Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Approximate Neighbors
    ('science', 1.0)
    ('astrobiology', 0.5924032926559448)
    ('transhumanist', 0.5916061401367188)
    ('bimonthly', 0.5861886739730835)
    ('sciences', 0.5851120948791504)
    ('robotics', 0.5844891369342804)
    ('nanomedicine', 0.5836333632469177)
    ('protoscience', 0.5796476304531097)
    ('biostatistics', 0.5791448056697845)
    ('astronautics', 0.5787959098815918)
    ('scientific', 0.5772265493869781)

    Normal (not Annoy-indexed) Neighbors
    ('science', 1.0)
    ('fiction', 0.7320358157157898)
    ('popularizer', 0.6709892153739929)
    ('astrobiology', 0.6677298545837402)
    ('transhumanist', 0.6664289236068726)
    ('technology', 0.660341739654541)
    ('bimonthly', 0.6575203537940979)
    ('sciences', 0.655735969543457)
    ('multidisciplinary', 0.6556889414787292)
    ('robotics', 0.6547014713287354)
    ('nanomedicine', 0.6532777547836304)


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

    # Dry run to make sure both indices are fully in RAM
    vector = model.wv.vectors_norm[0]
    model.wv.most_similar([vector], topn=5, indexer=annoy_index)
    model.wv.most_similar([vector], topn=5)

    import time
    import numpy as np

    def avg_query_time(annoy_index=None, queries=1000):
        """
        Average query time of a most_similar method over 1000 random queries,
        uses annoy if given an indexer
        """
        total_time = 0
        for _ in range(queries):
            rand_vec = model.wv.vectors_norm[np.random.randint(0, len(model.wv.vocab))]
            start_time = time.process_time()
            model.wv.most_similar([rand_vec], topn=5, indexer=annoy_index)
            total_time += time.process_time() - start_time
        return total_time / queries

    queries = 10000

    gensim_time = avg_query_time(queries=queries)
    annoy_time = avg_query_time(annoy_index, queries=queries)
    print("Gensim (s/query):\t{0:.5f}".format(gensim_time))
    print("Annoy (s/query):\t{0:.5f}".format(annoy_time))
    speed_improvement = gensim_time / annoy_time
    print ("\nAnnoy is {0:.2f} times faster on average on this particular run".format(speed_improvement))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Gensim (s/query):       0.02169
    Annoy (s/query):        0.00034

    Annoy is 63.71 times faster on average on this particular run


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

    ('science', 1.0)
    ('astrobiology', 0.5924032926559448)
    ('transhumanist', 0.5916061401367188)
    ('bimonthly', 0.5861886739730835)
    ('sciences', 0.5851120948791504)
    ('robotics', 0.5844891369342804)
    ('nanomedicine', 0.5836333632469177)
    ('protoscience', 0.5796476304531097)
    ('biostatistics', 0.5791448056697845)
    ('astronautics', 0.5787959098815918)
    ('scientific', 0.5772265493869781)


Be sure to use the same model at load that was used originally, otherwise you
will get unexpected behaviors.


6. Save memory via memory-mapping indices saved to disk
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







Bad example: two processes load the Word2vec model from disk and create there
own Annoy indices from that model.



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
and memory-map the index



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
    print("\nNormal (not Annoy-indexed) Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    71290 100
    the -0.086056426 0.15772334 -0.14391488 -0.10746263 -0.0036995178 -0.117373854 0.03937252 -0.14037031 -0.1252817 0.07694562 -0.021327982 0.007244886 0.16763417 -0.1226697 0.21137153 -0.063393526 -0.032362897 -0.0059070205 0.020281527 0.12367236 -0.025050493 -0.09774958 -0.24607891 -0.0064472477 -0.03055981 -0.4010833 -0.27916044 0.029562823 -0.071846716 -0.014671225 0.1420381 -0.053756475 -0.0855766 -0.090253495 0.60468906 0.09920296 0.35082236 -0.14631268 0.26485506 -0.08550774 0.09919222 -0.12538795 0.03159077 0.083675735 -0.13480936 0.043789566 -0.08674448 -0.079143874 0.05721798 0.023238886 -0.34467545 0.1550529 -0.18082479 -0.18602926 -0.18052024 0.074512914 0.15894942 -0.09034081 0.011110278 -0.15301983 -0.07879341 0.0013416538 -0.04413061 0.042708833 0.07895842 0.276121 0.11723857 0.18091062 0.07765438 0.023454918 0.07083069 0.001930411 0.2261552 -0.053920075 -0.14016616 -0.09455421 0.056401417 -0.06034534 -0.012578158 0.08775011 -0.089770935 -0.111630015 0.11005583 -0.091560066 0.0717941 -0.19018368 -0.049423326 0.29770434 0.17694262 -0.14268364 -0.1372601 0.14867909 -0.12172974 -0.07506602 0.09508915 -0.10644571 0.16355318 -0.1895201 0.04572383 -0.05629312
    of -0.24958447 0.33094105 -0.067723416 -0.15613635 0.15851182 -0.20777571 0.067617305 -0.14223038 -0.19351995 0.17955166 -0.01125617 -0.11227111 0.22649609 -0.07805858 0.08556426 0.10083455 -0.19243951 0.14512464 0.01395792 0.17216091 -0.008735538 -0.037496135 -0.3364987 0.03891899 0.036126327 -0.23090963 -0.22778185 0.09917219 0.12856483 0.0838603 0.17832059 0.021860743 -0.07048738 -0.18962148 0.5110143 0.07669086 0.2822584 -0.12050834 0.25681993 -0.021447591 0.21239889 -0.14476615 0.11061543 0.05422637 -0.02524366 0.08702608 -0.16577256 -0.20307428 0.011992565 -0.060010254 -0.3261019 0.2446808 -0.16701153 -0.079560414 -0.18528645 0.068947345 0.012339692 -0.06444969 -0.2089124 0.05786413 0.123009294 0.061585456 -0.042849902 0.16915381 0.03432279 0.13971788 0.25727242 0.09388416 0.1682245 -0.094005674 0.07307955 0.1292721 0.3170865 0.07673286 -0.07462851 -0.10278059 0.23569265 0.035961017 -0.06366512 0.034729835 -0.1799267 -0.12194269 0.19733816 -0.07210646 0.19601586 -0.09816554 -0.13614751 0.35114622 0.08043916 -0.10852109 -0.16087142 0.1783411 0.0321268 -0.14652534 0.026698181 -0.11104949 0.15343753 -0.28783563 0.08911155 -0.17888589
    Approximate Neighbors
    ('cat', 1.0)
    ('cats', 0.5971987545490265)
    ('felis', 0.5874168574810028)
    ('albino', 0.5703404247760773)
    ('marten', 0.5679939687252045)
    ('leopardus', 0.5678345859050751)
    ('barsoomian', 0.5672095417976379)
    ('prionailurus', 0.567060798406601)
    ('ferret', 0.5667355954647064)
    ('eared', 0.566079169511795)
    ('sighthound', 0.5649237632751465)

    Normal (not Annoy-indexed) Neighbors
    ('cat', 0.9999998807907104)
    ('cats', 0.6755023002624512)
    ('felis', 0.6595503091812134)
    ('albino', 0.6307852268218994)
    ('marten', 0.6267415881156921)
    ('leopardus', 0.6264660954475403)
    ('barsoomian', 0.6253848075866699)
    ('prionailurus', 0.6251273155212402)
    ('ferret', 0.6245640516281128)
    ('eared', 0.6234253644943237)
    ('sighthound', 0.6214173436164856)


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

   **Total running time of the script:** ( 11 minutes  41.168 seconds)

**Estimated memory usage:**  807 MB


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

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
