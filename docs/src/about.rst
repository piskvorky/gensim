:orphan:

.. _about:

============
About
============

History
-------

Gensim started off as a collection of various Python scripts for the Czech Digital Mathematics Library `dml.cz <http://dml.cz/>`_ in 2008,
where it served to generate a short list of the most similar articles to a given article (**gensim = "generate similar"**).
I also wanted to try these fancy "Latent Semantic Methods", but the libraries that
realized the necessary computation were `not much fun to work with <http://soi.stanford.edu/~rmunk/PROPACK/>`_.

Naturally, I set out to reinvent the wheel. Our `2010 LREC publication <http://radimrehurek.com/gensim/lrec2010_final.pdf>`_
describes the initial design decisions behind Gensim (clarity, efficiency and scalability)
and is fairly representative of how Gensim works even today.

Later versions of Gensim improved this efficiency and scalability tremendously. In fact,
I made algorithmic scalability of distributional semantics the topic of my `PhD thesis <http://radimrehurek.com/phd_rehurek.pdf>`_.

By now, Gensim is---to my knowledge---the most robust, efficient and hassle-free piece
of software to realize unsupervised semantic modelling from plain text. It stands
in contrast to brittle homework-assignment-implementations that do not scale on one hand,
and robust java-esque projects that take forever just to run "hello world".

In 2011, I started using `Github <https://github.com/piskvorky/gensim>`_ for source code hosting
and the Gensim website moved to its present domain. In 2013, Gensim got its current logo and website design. In 2018, we reworked the documentation style and started using the `ReadTheDocs theme <https://sphinx-rtd-theme.readthedocs.io/en/latest/>`_.


The creation of Gensim was motivated by a perceived lack of available, scalable software
frameworks that realize topic modelling, and/or their overwhelming internal complexity (hail Java!).
You can read more about the motivation in our `LREC 2010 workshop paper <lrec2010_final.pdf>`_.
If you want to cite Gensim in your own work, please refer to `that article (`BibTeX <bibtex_gensim.bib>`_).

You're welcome to share your results and experiments on the `mailing list <https://groups.google.com/group/gensim>`_.


Contributors
--------------

Credit goes to all the people who contributed to gensim, be it in `discussions <http://groups.google.com/group/gensim>`_,
ideas, `code contributions <https://github.com/piskvorky/gensim/pulls>`_ or `bug reports <https://github.com/piskvorky/gensim/issues>`_.

It's really useful and motivating to get feedback, in any shape or form, so big thanks to you all!

Some honorable mentions are included in the `CHANGELOG.txt <https://github.com/piskvorky/gensim/blob/develop/CHANGELOG.md>`_.
