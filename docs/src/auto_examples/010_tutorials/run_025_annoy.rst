.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_025_annoy.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_025_annoy.py:


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

For more details, see :ref:`sphx_glr_auto_examples_010_tutorials_run_020_word2vec.py`.


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
    ('fiction', 0.6646454632282257)
    ('astrobiology', 0.5906687378883362)
    ('cryobiology', 0.5898622274398804)
    ('psychohistory', 0.5836915969848633)
    ('subdiscipline', 0.5829995274543762)
    ('sciences', 0.5802876055240631)
    ('astronautics', 0.578422874212265)
    ('multidisciplinary', 0.5774217844009399)
    ('scientific', 0.5771709084510803)
    ('bimonthly', 0.5730613172054291)

    Normal (not Annoy-indexed) Neighbors
    ('science', 1.0000001192092896)
    ('fiction', 0.7750747203826904)
    ('astrobiology', 0.6648958325386047)
    ('cryobiology', 0.6635739803314209)
    ('psychohistory', 0.6533745527267456)
    ('subdiscipline', 0.6522213220596313)
    ('sciences', 0.6476829051971436)
    ('astronautics', 0.6445454359054565)
    ('multidisciplinary', 0.6428552865982056)
    ('scientific', 0.6424310803413391)
    ('bimonthly', 0.6354467272758484)


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

    Gensim (s/query):       0.02291
    Annoy (s/query):        0.00032

    Annoy is 71.18 times faster on average on this particular run


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
    ('fiction', 0.6646454632282257)
    ('astrobiology', 0.5906687378883362)
    ('cryobiology', 0.5898622274398804)
    ('psychohistory', 0.5836915969848633)
    ('subdiscipline', 0.5829995274543762)
    ('sciences', 0.5802876055240631)
    ('astronautics', 0.578422874212265)
    ('multidisciplinary', 0.5774217844009399)
    ('scientific', 0.5771709084510803)
    ('bimonthly', 0.5730613172054291)


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




.. image:: /auto_examples/010_tutorials/images/sphx_glr_run_025_annoy_001.png
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
    the -0.008877677 -0.16178143 -0.041890636 -0.14486739 -0.114816144 -0.19139393 0.29965618 0.26063424 -0.050092306 -0.13699013 0.027312662 -0.11600112 0.057511587 0.038626015 0.090375476 0.2117741 0.007968547 0.20590279 -0.34600076 0.1575698 0.3747151 -0.26465425 0.07756059 0.14341223 0.11173541 0.15749337 0.14232372 -0.12824887 0.052798793 0.03622227 0.059895013 -0.062976375 -0.2195778 -0.13657983 -0.068801075 -0.15458456 -0.044779815 0.040269848 0.16631392 -0.025678338 0.0023440733 0.15350549 -0.10201911 0.2405792 -0.013499753 -0.04562671 0.008241601 0.23794737 -0.029357407 0.038136706 0.048529215 0.27666563 -0.05592086 -0.05729889 -0.014598321 0.010149134 0.16827142 -0.001242163 0.06090431 -0.31480762 0.01129965 0.21857782 -0.068782724 -0.10593644 0.13697655 -0.010484434 0.10308812 -0.15077662 -0.14537275 0.07355051 0.045693677 0.18758582 -0.19937238 0.004293586 -0.13835987 0.16932209 -0.06677947 -0.14445989 -0.2732761 -0.104275025 -0.10563875 0.09393968 0.17659709 -0.04665651 -0.10145272 0.07615343 -0.123800345 0.09237656 0.021453477 -0.13675718 0.010144629 -0.29110703 0.2057081 -0.09255506 -0.054220088 -0.43776104 0.26176882 0.119828835 -0.07476847 0.034220032
    of 0.057812527 -0.040832467 -0.10108069 -0.17980209 -0.050963726 -0.1156882 0.2797317 0.30010927 -0.074353956 -0.1880128 0.008626096 -0.12354891 0.081979804 0.03791263 0.16799867 -0.02268378 -0.06348159 0.25409827 -0.19388804 0.024280272 0.25820243 -0.2606216 0.1737779 0.14841811 0.10896455 0.1936879 -0.017354602 -0.19451316 -0.03919017 0.07150693 -0.006186833 0.064523034 -0.23477787 0.13462563 0.040441148 -0.16914757 0.0021874027 0.049427353 0.1476423 -0.09252517 0.052843023 -0.016920505 -0.06931757 0.3243218 -0.038953003 -0.021064185 -0.010963274 0.26387486 0.015677923 -0.06487413 -0.0071705543 0.46162283 0.015544745 -0.16395465 -0.033174727 0.006102506 0.24251048 -0.028176796 0.15116666 -0.2536896 0.03570488 0.27419424 -0.18038616 -0.22793444 0.23096019 0.11795743 0.1957931 -0.2027984 -0.1718665 0.09556287 -0.049765557 0.083867066 -0.18442379 -0.07646543 0.056592505 0.27677026 -0.06869161 -0.16141924 -0.11739019 -0.0787136 -0.058190815 -0.14153633 0.19784677 -0.119485125 -0.23956907 0.20247057 -0.22489299 0.14543393 0.27593163 -0.13117215 -0.028312035 -0.239989 0.026754523 0.016047047 -0.004090589 -0.36234108 0.31486917 0.17445625 -0.019496562 0.20636901
    Approximate Neighbors
    ('cat', 1.0)
    ('leopardus', 0.607132226228714)
    ('marten', 0.5900265574455261)
    ('prionailurus', 0.587386965751648)
    ('eared', 0.5838236808776855)
    ('toed', 0.5817761123180389)
    ('sighthound', 0.5761100649833679)
    ('cats', 0.5733330249786377)
    ('tapir', 0.5721204876899719)
    ('pictus', 0.5715179443359375)
    ('dog', 0.5706620216369629)

    Normal (not Annoy-indexed) Neighbors
    ('cat', 1.0)
    ('leopardus', 0.6913098096847534)
    ('felis', 0.6683568358421326)
    ('marten', 0.6638437509536743)
    ('prionailurus', 0.6595009565353394)
    ('eared', 0.6535945534706116)
    ('toed', 0.650177538394928)
    ('sighthound', 0.6406346559524536)
    ('meow', 0.6397078633308411)
    ('cats', 0.6359107494354248)
    ('tapir', 0.6338382363319397)


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

   **Total running time of the script:** ( 10 minutes  56.419 seconds)

**Estimated memory usage:**  883 MB


.. _sphx_glr_download_auto_examples_010_tutorials_run_025_annoy.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_025_annoy.py <run_025_annoy.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_025_annoy.ipynb <run_025_annoy.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
