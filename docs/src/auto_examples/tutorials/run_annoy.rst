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
    ('science', 1.0)
    ('multidisciplinary', 0.608674556016922)
    ('astrobiology', 0.5977040827274323)
    ('interdisciplinary', 0.5937487781047821)
    ('bioethics', 0.5934497117996216)
    ('astronautics', 0.5890172123908997)
    ('astrophysics', 0.58620685338974)
    ('psychohistory', 0.5828591883182526)
    ('sciences', 0.5820683240890503)
    ('actuarial', 0.5794413983821869)
    ('scientific', 0.578777939081192)

    Exact Neighbors
    ('science', 1.0)
    ('fiction', 0.7297012209892273)
    ('multidisciplinary', 0.6937288641929626)
    ('astrobiology', 0.6763160228729248)
    ('interdisciplinary', 0.6699198484420776)
    ('bioethics', 0.6694337725639343)
    ('vernor', 0.6654549837112427)
    ('vinge', 0.6640741229057312)
    ('astronautics', 0.6621862649917603)
    ('astrophysics', 0.6575504541397095)
    ('technology', 0.6531316637992859)




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

    Gensim (s/query):       0.00654
    Annoy (s/query):        0.00055

    Annoy is 11.88 times faster on average on this particular run




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
    ('multidisciplinary', 0.608674556016922)
    ('astrobiology', 0.5977040827274323)
    ('interdisciplinary', 0.5937487781047821)
    ('bioethics', 0.5934497117996216)
    ('astronautics', 0.5890172123908997)
    ('astrophysics', 0.58620685338974)
    ('psychohistory', 0.5828591883182526)
    ('sciences', 0.5820683240890503)
    ('actuarial', 0.5794413983821869)
    ('scientific', 0.578777939081192)




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
    the 0.16007873 -0.061821494 0.16403554 0.20444521 -0.33250368 -0.18388326 -0.11518438 0.26039606 -0.09880219 -0.114337094 -0.24393205 -0.16671345 0.010349793 0.22049113 0.014908477 -0.10886409 -0.050133377 0.014529925 0.0066863606 -0.14707142 0.0400251 0.07787519 -0.18831152 -0.13362508 0.282132 -0.050551824 0.13475767 -0.06569664 0.0031677599 0.07820668 -0.35095257 0.30480665 -0.033180837 -0.048362087 0.04275413 -0.05222876 -0.071952015 -0.035658896 0.07901254 -0.10421657 -0.10299242 0.06549932 0.24401794 -0.15140945 0.029012768 0.04028114 -0.22667517 -0.14450285 -0.23805015 0.08867654 -0.18326522 -0.04525019 0.106588475 -0.00038971985 0.2078292 -0.31376663 0.19781663 0.17066158 0.16820915 -0.047588248 0.20909792 -0.0993302 0.11492583 0.07690898 0.026019907 -0.24461384 -0.15658146 -0.097338416 0.13501963 0.038800433 -0.10874414 0.016372694 0.08403854 0.16431145 0.25076985 -0.10206646 -0.12634891 0.047575567 -0.04372017 0.056058753 -0.06418228 -0.1375621 0.14441249 -0.37270537 -0.12438262 -0.084386 -0.0616519 -0.04610768 -0.1488726 -0.2879129 0.02876804 -0.0783338 0.049880445 -0.2890527 0.052437592 -0.11808442 0.09637225 0.17164731 -0.03777877 0.10314265
    of 0.12728788 -0.07128835 0.22709015 0.21735586 -0.26458326 -0.14139651 -0.21792462 0.08618337 0.08907982 -0.083991244 -0.11595708 -0.20405494 0.11473529 0.106475234 0.16436335 -0.16281348 -0.050799012 0.044015124 0.023081105 -0.08942257 0.12587656 0.17702717 -0.23259656 -0.0012328548 0.25392023 -0.0049020797 0.18065476 -0.15828626 -0.009485071 0.112988144 -0.3753395 0.060977582 0.018466623 0.09084287 -0.12861633 -0.059884362 -0.07544826 0.040726017 0.10942843 -0.21307503 0.00090036006 0.11597715 0.22929604 -0.11609176 0.035484787 0.00071995956 -0.32539764 -0.12604061 -0.005495456 0.04436327 -0.1105619 -0.12655294 0.045705166 -0.14065112 0.21226525 -0.4863211 0.09879361 0.07101748 0.20841932 -0.028169975 0.075062476 -0.26905793 0.057516105 0.031906158 0.1752423 -0.19624741 -0.20997943 -0.10417411 -0.004082244 0.029495642 -0.07799115 -0.061133463 0.028057387 0.06255617 0.25191864 -0.048677184 -0.40772855 -0.025113298 0.019805929 -0.010906071 0.029409314 -0.17279296 0.14616875 -0.44125536 -0.1683791 -0.39358017 -0.04599949 0.10306317 -0.10953343 -0.36125863 -0.103272185 0.09990804 0.026997609 -0.17567022 0.12559506 -0.014309327 0.015485785 0.170501 -0.13221653 0.04849726
    Approximate Neighbors
    ('cat', 0.9998273665260058)
    ('leopardus', 0.594965249300003)
    ('cats', 0.5882971882820129)
    ('prionailurus', 0.5790365040302277)
    ('proboscis', 0.5778042674064636)
    ('eared', 0.5742282271385193)
    ('dogs', 0.5695933997631073)
    ('skunks', 0.5693343579769135)
    ('albino', 0.56873419880867)
    ('coyote', 0.5658003985881805)
    ('ferret', 0.5657358169555664)

    Exact Neighbors
    ('cat', 1.0)
    ('leopardus', 0.6718936562538147)
    ('felis', 0.6702097654342651)
    ('cats', 0.6610016822814941)
    ('lynxes', 0.6600459218025208)
    ('meow', 0.6570931077003479)
    ('prionailurus', 0.6455793380737305)
    ('proboscis', 0.6435014605522156)
    ('eared', 0.6374367475509644)
    ('crustacean', 0.6350691914558411)
    ('dogs', 0.6295004487037659)




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

   **Total running time of the script:** ( 15 minutes  21.247 seconds)

**Estimated memory usage:**  732 MB


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
