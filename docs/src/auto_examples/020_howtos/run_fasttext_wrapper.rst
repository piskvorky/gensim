.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_020_howtos_run_fasttext_wrapper.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_020_howtos_run_fasttext_wrapper.py:


.. _fasttext_tutorial:

How to Use Gensim as a Wrapper for Facebook's FastText
======================================================

This tutorial is about using `fastText <https://github.com/facebookresearch/fastText>`_ model in Gensim.
There are two ways you can use fastText in Gensim - Gensim's native implementation of fastText and Gensim wrapper for fastText's original C++ code.
This howto explains the latter.
For the former, see :ref:`fasttext_tutorial`.


.. code-block:: default



    from gensim.models.wrappers.fasttext import FastText as FT_wrapper








You will need fastText setup locally to be able to train models.
See `installation instructions for fastText <https://github.com/facebookresearch/fastText/#requirements>`_ if you don't have fastText installed already.



.. code-block:: default


    # Set FastText home to the path to the FastText executable
    ft_home = '/home/misha/src/fastText-0.1.0/fasttext'

    # train the model
    from gensim.test.utils import datapath

    corpus_file = datapath('lee_background.cor')
    model_wrapper = FT_wrapper.train(ft_home, corpus_file)
    print(model_wrapper)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    FastText(vocab=1763, size=100, alpha=0.025)


Perform common operations with the model
For a more detailed introduction to FastText, :ref:`fasttext_tutorial`.
The native and wrapped model behave identically, and differ in implementation only.



.. code-block:: default


    print('night' in model_wrapper.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True



.. code-block:: default

    print('nights' in model_wrapper.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False



.. code-block:: default

    print(model_wrapper['night'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.58978295  0.8426159  -0.06617589  0.03257823  0.59658366 -0.18775244
     -0.1209754   0.04814022  0.07168142  0.33514315  0.45747155 -0.03126722
     -0.5044903  -0.31042826 -0.16879548  0.04043612  0.21056254 -0.51706195
     -0.5437506   0.63012564 -0.32364595  0.29704517  0.02074578 -0.6267816
     -0.18130697 -0.39228055 -0.17108509 -0.43117905  0.39292988 -0.30212766
     -0.3155277  -0.04916786 -0.11951092  0.51950324  0.69708097 -0.19310777
      0.40499705 -0.00174429  0.17571536  0.14801039  0.08918523 -0.21303771
      0.10828466 -0.34582406 -0.2723538  -0.21456425  0.206348   -0.5920652
      0.22465128 -0.1664295   0.06154541  0.55948025 -0.0433962   0.12031706
     -0.30288517 -0.39531484 -0.34388143  0.47573575 -0.12175087 -0.03379979
     -0.23427525 -0.16888586  0.46181354  0.565824    0.1039295   0.3196718
     -0.5992658  -0.31585726 -0.15929429 -0.08063186 -0.31945604 -0.12173763
     -0.10911509 -0.3053498  -0.29422343 -0.5009908   0.08478063  0.24098763
      0.28110197 -0.33254778  0.693258    1.1737784   0.2614068  -0.7187338
      0.40009603  0.05380035  0.07761053 -0.20687915  0.2561323  -0.4304768
      0.28695467  0.2993385   0.401913    0.06296781  0.0834905   0.01684867
     -0.48562098  0.10795977  0.28098068  0.4456529 ]



.. code-block:: default

    print(model_wrapper['nights'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.5355776   0.7657215  -0.06084945  0.03128933  0.5391221  -0.17138074
     -0.1107616   0.04536759  0.06715514  0.3041941   0.4165099  -0.0282925
     -0.458154   -0.28126416 -0.15437806  0.03620834  0.19185504 -0.47007284
     -0.492974    0.570655   -0.29554746  0.27044308  0.01816352 -0.56726706
     -0.16626313 -0.35725173 -0.15637018 -0.39067256  0.35639378 -0.2751909
     -0.28812438 -0.04578764 -0.10776299  0.47223416  0.63353056 -0.17690566
      0.36799955 -0.0012087   0.15782937  0.13611239  0.08104339 -0.19354115
      0.09703788 -0.3158152  -0.2471307  -0.19503833  0.18881644 -0.5388569
      0.20489565 -0.15064138  0.05592088  0.50836563 -0.03969989  0.10952631
     -0.27544647 -0.35971498 -0.31184983  0.43027174 -0.10899004 -0.03172619
     -0.21067771 -0.15413919  0.41884112  0.51320916  0.09513052  0.2893828
     -0.5438015  -0.28410307 -0.14278802 -0.07243418 -0.28824195 -0.11226971
     -0.0961733  -0.2784101  -0.2683405  -0.45599788  0.07622197  0.21856138
      0.25413215 -0.30291924  0.6299003   1.0648143   0.23351006 -0.65175706
      0.36367148  0.04801689  0.06816668 -0.18762454  0.23217642 -0.39049855
      0.2601725   0.27210653  0.3647062   0.05675317  0.07371369  0.01476919
     -0.44055575  0.09843462  0.25596043  0.40296102]


Tests if word present in vocab



.. code-block:: default

    print("word" in model_wrapper.wv.vocab)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False


Tests if vector present for word



.. code-block:: default

    print("word" in model_wrapper)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True


saving a model trained via fastText wrapper



.. code-block:: default

    model_wrapper.save('saved_model_wrapper')
    loaded_model = FT_wrapper.load('saved_model_wrapper')
    print(loaded_model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    FastText(vocab=1763, size=100, alpha=0.025)


That's all! You've made it to the end of this tutorial.

Here we show a pretty fastText logo so that our gallery picks it up as a thumbnail.



.. code-block:: default

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('fasttext-logo-color-web.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/020_howtos/images/sphx_glr_run_fasttext_wrapper_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  15.304 seconds)

**Estimated memory usage:**  774 MB


.. _sphx_glr_download_auto_examples_020_howtos_run_fasttext_wrapper.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/020_howtos/run_fasttext_wrapper.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_fasttext_wrapper.py <run_fasttext_wrapper.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_fasttext_wrapper.ipynb <run_fasttext_wrapper.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
