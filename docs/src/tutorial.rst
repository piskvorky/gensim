.. _tutorial:

Tutorial
========


This tutorial is organized as a series of examples that highlight various features 
of `gensim`. It is assumed that the reader is familiar with the Python language 
and has read the :doc:`intro`.


All the examples can be directly copied to your Python interpreter shell (assuming 
you have :doc:`gensim installed <install>`, of course). 
`IPython <http://ipython.scipy.org>`_'s ``cpaste`` command is especially handy for copypasting code fragments which include superfluous 
characters, such as the leading ``>>>``.

Gensim uses Python's standard :mod:`logging` module to log various stuff at various 
priority levels; to activate logging (optional), run

>>> import logging
>>> logging.root.level = logging.INFO # will suppress DEBUG level events

The examples are divided into parts on:

.. toctree::
   :maxdepth: 2
   
   tut1
   tut2
   tut3

