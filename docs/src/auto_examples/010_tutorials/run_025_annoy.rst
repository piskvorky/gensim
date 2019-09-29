.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_010_tutorials_run_025_annoy.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_010_tutorials_run_025_annoy.py:


.. _annoy_py:

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

For more details, see :ref:`word2vec_py`.


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
    ('actuarial', 0.6072465479373932)
    ('scientific', 0.602117657661438)
    ('psychohistory', 0.5988825261592865)
    ('interdisciplinary', 0.5885585844516754)
    ('astronautics', 0.5860425531864166)
    ('aaas', 0.5845794379711151)
    ('econometrics', 0.5834062695503235)
    ('sciences', 0.5820188522338867)
    ('astrobiology', 0.5809557735919952)
    ('xenobiology', 0.5724596679210663)

    Normal (not Annoy-indexed) Neighbors
    ('science', 1.0)
    ('fiction', 0.7702922821044922)
    ('actuarial', 0.6914894580841064)
    ('scientific', 0.6833791732788086)
    ('vinge', 0.679933488368988)
    ('psychohistory', 0.6782095432281494)
    ('vernor', 0.6690018773078918)
    ('interdisciplinary', 0.6614320278167725)
    ('astronautics', 0.6572784185409546)
    ('aaas', 0.654851496219635)
    ('econometrics', 0.6528993844985962)


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

    Gensim (s/query):       0.02303
    Annoy (s/query):        0.00037

    Annoy is 62.09 times faster on average on this particular run


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
    ('actuarial', 0.6072465479373932)
    ('scientific', 0.602117657661438)
    ('psychohistory', 0.5988825261592865)
    ('interdisciplinary', 0.5885585844516754)
    ('astronautics', 0.5860425531864166)
    ('aaas', 0.5845794379711151)
    ('econometrics', 0.5834062695503235)
    ('sciences', 0.5820188522338867)
    ('astrobiology', 0.5809557735919952)
    ('xenobiology', 0.5724596679210663)


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
    the -0.18395565 -0.015361177 0.08397919 -0.07501288 0.010971526 0.008795478 0.0010885551 0.09293926 0.24763535 -0.0001843912 0.13640349 0.11277892 -0.052380946 -0.30408052 -0.25086612 0.5083817 0.2129764 0.12961014 -0.23943749 0.07904556 0.3107087 0.19794293 0.09660584 -0.0654535 0.009358463 0.0401344 0.11823861 0.15519024 0.3520571 0.21338835 0.09736391 -0.081011645 -0.13500625 0.11319087 -0.15906958 0.2792225 0.047634594 -0.29144284 0.031940382 -0.014366943 -0.017406357 0.049795173 -0.0131750675 0.05754607 -0.038468108 0.070423394 -0.07113191 0.0064027975 -0.20760006 0.025915816 0.16060196 0.29822364 -0.17123637 0.003463074 0.2264824 0.08915844 -0.17212324 0.11781398 -0.086406246 -0.124525435 0.0095292535 0.014133081 -0.055610564 -0.085644774 0.15268208 -0.04600042 0.14077722 0.052425724 -0.019699693 0.35600862 -0.23356779 -0.024686804 -0.10365586 -0.038440425 0.13434231 -0.03914209 0.13203721 0.024594193 -0.13437696 0.036800154 0.2281823 -0.1496753 -0.09030526 0.059661184 -0.020240275 -0.33445755 -0.10408854 0.011251482 -0.042800482 -0.082380556 -0.349739 0.018962448 0.20590012 -0.18086083 0.08516421 0.19721507 -0.08006474 0.05125941 -0.05906177 0.09534557
    of -0.09386086 -0.09997297 0.107172914 -0.14112245 -0.06190288 -0.2015995 0.051556118 0.015507449 0.18916722 -0.014963632 0.07873747 0.14684114 -0.15620837 -0.24201381 -0.28984547 0.5010303 0.24530838 0.23498188 -0.24717513 0.042025443 0.31890133 0.15990637 0.16482458 0.04276489 0.04624985 0.110656664 -0.009911845 0.14211953 0.30632466 0.18246149 0.097348705 -0.015386323 -0.04130826 0.21747632 0.010935479 0.23221208 -0.1466754 -0.25050545 0.031367484 -0.024762979 0.088719875 -0.010854793 0.02038931 0.092519976 0.002876951 0.2017433 -0.016342157 -0.07631585 -0.15698256 0.022284476 0.37190118 0.292232 -0.0018952816 0.0136836255 0.17445624 0.11495022 -0.27118754 0.10997932 0.08051957 -0.019119145 0.17819701 -0.009440146 -0.036142215 -0.1386614 0.17518324 -0.018648358 0.20594072 0.16595082 -0.071064085 0.34954277 -0.38425535 0.096485846 -0.0016466301 -0.08830942 0.26106372 -0.071637996 0.15544547 0.11792928 -0.23733717 0.2223326 0.13823675 -0.016166156 0.06644764 0.09522214 -0.03249609 0.032166895 -0.049139787 -0.06425062 -0.12292215 -0.1422828 -0.3979004 0.033003196 0.04985052 -0.17423545 -0.101575576 -0.021629866 0.020093296 0.09153096 0.10846912 0.16454037
    Approximate Neighbors
    ('cat', 1.0)
    ('prionailurus', 0.5951117277145386)
    ('cats', 0.5909131467342377)
    ('kitten', 0.589909553527832)
    ('sighthound', 0.5882692039012909)
    ('dog', 0.5836004018783569)
    ('leopardus', 0.5755427181720734)
    ('meow', 0.5740945339202881)
    ('hairless', 0.5720171332359314)
    ('llamas', 0.5711125731468201)
    ('proboscis', 0.570663720369339)

    Normal (not Annoy-indexed) Neighbors
    ('cat', 1.0000001192092896)
    ('prionailurus', 0.6721309423446655)
    ('cats', 0.6652958393096924)
    ('kitten', 0.6636516451835632)
    ('sighthound', 0.6609554290771484)
    ('felis', 0.6539616584777832)
    ('dog', 0.6532227396965027)
    ('guppy', 0.6413432359695435)
    ('leopardus', 0.6396721601486206)
    ('meow', 0.6372090578079224)
    ('hairless', 0.6336612701416016)


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

   **Total running time of the script:** ( 12 minutes  36.113 seconds)

**Estimated memory usage:**  781 MB


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
