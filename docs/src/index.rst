.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Topic Modelling for Humans
=====================================================


.. raw:: html
   :file: _static/tagcloud.html


.. raw:: html

   <table class="links" style="font-size: 150%;">
        <tr>
                <td><a href='intro.html'>What is this?</a></td>
                <td><a href='install.html'>Installation</a></td>
        </tr>
        <tr>
                <td><a href='tutorial.html'>Get started!</a></td>
                <td><a href='about.html'>About</a></td>
        </tr>
   </table>


Quick Reference Example
------------------------

>>> from gensim import corpora, models, similarities
>>>
>>> # Load corpus iterator from a Matrix Market file on disk.
>>> corpus = corpora.MmCorpus('/path/to/corpus.mm')
>>>
>>> # Initialize a transformation (Latent Semantic Indexing with 200 latent dimensions).
>>> lsi = models.LsiModel(corpus, num_topics=200)
>>>
>>> # Convert another corpus to the latent space and index it.
>>> index = similarities.MatrixSimilarity(lsi[another_corpus])
>>>
>>> # determine similarity of a query document against each document in the index
>>> sims = index[query]


.. admonition:: What's new?

   * 9 Mar 2012: release 0.8.4: new model `Hierarchical Dirichlet Process <http://radimrehurek.com/gensim/models/hdpmodel.html>`_ (full `CHANGELOG <https://github.com/piskvorky/gensim/blob/develop/CHANGELOG.txt>`_)
   * 2 Dec 2011: bug-fix release 0.8.3 out; `CHANGELOG <https://github.com/piskvorky/gensim/blob/develop/CHANGELOG.txt>`_
   * 1 Dec 2011: released `simserver <http://pypi.python.org/pypi/simserver>`_, a Python document similarity server based on gensim




.. toctree::
   :hidden:
   :maxdepth: 1

   intro
   install
   tutorial
   distributed
   wiki
   apiref
