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
    ('science', 1.0)
    ('actuarial', 0.6134995520114899)
    ('astrobiology', 0.605967104434967)
    ('scientific', 0.5998380184173584)
    ('psychohistory', 0.5983279347419739)
    ('aaas', 0.5953519940376282)
    ('sciences', 0.5930938422679901)
    ('multidisciplinary', 0.5905506014823914)
    ('brookings', 0.5872612297534943)
    ('protoscience', 0.5871273577213287)
    ('robotics', 0.5763619542121887)

    Exact Neighbors
    ('science', 1.0)
    ('fiction', 0.7634229063987732)
    ('actuarial', 0.7012346982955933)
    ('astrobiology', 0.6894761323928833)
    ('scientific', 0.6797406673431396)
    ('psychohistory', 0.6773191690444946)
    ('aaas', 0.6725200414657593)
    ('sciences', 0.6688549518585205)
    ('technology', 0.6647214889526367)
    ('multidisciplinary', 0.664702296257019)
    ('brookings', 0.6592934131622314)




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
            rand_vec = model.wv.vectors_norm[np.random.randint(0, len(model.wv.vocab))]
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

    Gensim (s/query):       0.00818
    Annoy (s/query):        0.00065

    Annoy is 12.60 times faster on average on this particular run




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
    ('actuarial', 0.6134995520114899)
    ('astrobiology', 0.605967104434967)
    ('scientific', 0.5998380184173584)
    ('psychohistory', 0.5983279347419739)
    ('aaas', 0.5953519940376282)
    ('sciences', 0.5930938422679901)
    ('multidisciplinary', 0.5905506014823914)
    ('brookings', 0.5872612297534943)
    ('protoscience', 0.5871273577213287)
    ('robotics', 0.5763619542121887)




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

    /Volumes/work/workspace/gensim/trunk/docs/src/gallery/tutorials/run_annoy.py:319: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()




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
    the -0.093956634 -0.23012587 0.26526973 -0.070751935 -0.04763924 0.3800344 0.00058125437 -0.37860185 -0.30190718 -0.33000842 0.00018963935 -0.051288195 0.14841056 0.07899927 0.0782879 0.2748283 0.06865467 0.033617623 0.23343633 -0.14310712 0.05970611 0.047652725 0.25243673 -0.037743393 -0.08227887 -0.13634236 -0.045384776 -0.11800709 -0.21941897 -0.14257649 -0.08368551 -0.23509803 0.24342766 -0.071133874 0.16312273 0.29646543 0.19113424 -0.016952338 -0.122792475 -0.15435502 0.03669005 -0.0132581405 0.19434701 0.23526081 0.10232508 -0.08873911 0.12518178 -0.034376215 -0.029859077 0.03897621 -0.14511836 -0.13433056 0.124084644 -0.17298278 -0.2384294 0.10490191 -0.09075346 0.068099104 -0.25829297 -0.043966103 -0.26129034 0.058955386 0.06891225 0.10275935 0.23172262 0.18205403 -0.031837042 -0.010094245 0.005386607 0.17057273 0.057838008 0.09608744 -0.0047167335 -0.088014826 -0.031114876 -0.019799907 -0.11345058 0.16654405 -0.2768999 -0.03593294 -0.08378541 -0.226621 0.13354443 -0.093731284 -0.107354954 -0.0224138 0.13090602 -0.1462274 -0.10129779 0.062061697 -0.020099644 0.096954696 -0.21072951 0.15149264 0.24161623 -0.13526335 -0.061208013 0.16302744 -0.098646834 -0.04380757
    of -0.008969164 -0.2968002 0.314561 0.02852287 -0.00078028237 0.21124291 -0.06693117 -0.3984108 -0.33785093 -0.30351913 0.12920533 0.10624988 0.082755074 0.21052273 0.09315073 0.14871123 -0.034639347 0.16682656 0.19270846 -0.15999612 0.10588848 0.12476963 0.14933349 0.018821964 0.056002345 -0.25375253 -0.09874099 -0.17829226 -0.15542915 -0.25009328 -0.25002068 -0.22487502 0.047477134 -0.17303415 -0.038107544 0.42119476 0.14560683 -0.1958168 -0.017695677 -0.13824363 0.033620406 -0.14019567 0.119756095 0.22004323 0.17888938 -0.088430464 -0.012002373 -0.17217776 -0.16610356 -0.0029083597 -0.22056521 -0.060856182 -0.07275941 -0.005672054 -0.12612358 0.09198229 -0.081438884 0.055194274 -0.30819967 -0.06791485 -0.29599926 0.081666365 0.109533764 0.13906485 0.32025543 0.26031888 0.09384002 0.068416335 0.07796767 0.17372635 0.015832478 0.053521108 0.04928199 0.07576942 -0.030521782 -0.095616445 -0.07487119 0.16824985 -0.18423142 0.05483694 -0.11041758 -0.23955221 0.12596992 -0.035123825 -0.05362532 -0.094509184 -0.006634746 -0.24070449 -0.0899755 -0.040458627 -0.04916877 -0.0049628043 -0.271932 0.23019491 0.28511706 -0.1984597 -0.08756601 0.11145819 -0.053986672 -0.21573606
    Approximate Neighbors
    ('cat', 1.0)
    ('cats', 0.5908078849315643)
    ('marten', 0.5744927823543549)
    ('eared', 0.5717700123786926)
    ('possum', 0.5715496838092804)
    ('leopardus', 0.5705506801605225)
    ('prionailurus', 0.5678126215934753)
    ('kitten', 0.5652846097946167)
    ('hairless', 0.5651653409004211)
    ('sighthound', 0.5645075142383575)
    ('ferret', 0.5641167163848877)

    Exact Neighbors
    ('cat', 1.0)
    ('cats', 0.665123701095581)
    ('crustacean', 0.6480848789215088)
    ('meow', 0.638325572013855)
    ('marten', 0.6378872394561768)
    ('eared', 0.6332381963729858)
    ('possum', 0.6328606605529785)
    ('leopardus', 0.6311464309692383)
    ('prionailurus', 0.6264281272888184)
    ('kitten', 0.62204509973526)
    ('hairless', 0.6218376159667969)




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

   **Total running time of the script:** ( 13 minutes  22.753 seconds)

**Estimated memory usage:**  700 MB


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
