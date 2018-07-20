:orphan:

.. _about:

=====
About
=====

History
-------

Gensim started off as a collection of various Python scripts for the Czech Digital Mathematics Library `dml.cz <http://dml.cz/>`_ in 2008,
where it served to generate a short list of the most similar articles to a given article (**gensim = "generate similar"**).
I also wanted to try these fancy "Latent Semantic Methods", but the libraries that
realized the necessary computation were `not much fun to work with <http://soi.stanford.edu/~rmunk/PROPACK/>`_.

Naturally, I set out to reinvent the wheel. Our `2010 LREC publication <http://radimrehurek.com/gensim/lrec2010_final.pdf>`_
describes the initial design decisions behind Gensim: clarity, efficiency and scalability. It is fairly representative of how Gensim works even today.

Later versions of gensim improved this efficiency and scalability tremendously. In fact,
I made algorithmic scalability of distributional semantics the topic of my `PhD thesis <http://radimrehurek.com/phd_rehurek.pdf>`_.

By now, Gensim is---to my knowledge---the most robust, efficient and hassle-free piece
of software to realize unsupervised semantic modelling from plain text. It stands
in contrast to brittle homework-assignment-implementations that do not scale on one hand,
and robust java-esque projects that take forever just to run "hello world".

In 2011, I started using `Github <https://github.com/piskvorky/gensim>`_ for source code hosting
and the Gensim website moved to its present domain. In 2013, Gensim got its current logo and website design.


Licensing
----------

Gensim is licensed under the OSI-approved `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.
This means that it's free for both personal and commercial use, but if you make any
modification to Gensim that you distribute to other people, you have to disclose
the source code of these modifications.

Apart from that, you are free to redistribute Gensim in any way you like, though you're
not allowed to modify its license (doh!).

My intent here is to **get more help and community involvement** with the development of Gensim.
The legalese is therefore less important to me than your input and contributions.

`Contact me <mailto:me@radimrehurek.com>`_ if LGPL doesn't fit your bill and you'd like the open source restrictions lifted.

.. seealso::

    We also built a high performance commercial server for NLP, document analysis, indexing, search and clustering: https://scaletext.ai. ScaleText is available both on-prem and as SaaS.

    Reach out at info@scaletext.com if you need an industry-grade NLP tool with professional support.


Contributors
------------

Credit goes to the many people who contributed to Gensim, be it in `discussions <http://groups.google.com/group/gensim>`_,
ideas, `code contributions <https://github.com/piskvorky/gensim/pulls>`_ or `bug reports <https://github.com/piskvorky/gensim/issues>`_.

It's really useful and motivating to get feedback, in any shape or form, so big thanks to you all!

Some honorable mentions are included in the `CHANGELOG.txt <https://github.com/piskvorky/gensim/blob/develop/CHANGELOG.md>`_.

Academic citing
---------------

Gensim has been used in `over a thousand research paper and student theses <https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:NaGl4SEjCO4C>`_.

When citing Gensim, please use `this BibTeX entry <bibtex_gensim.bib>`_::

  @inproceedings{rehurek_lrec,
        title = {{Software Framework for Topic Modelling with Large Corpora}},
        author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
        booktitle = {{Proceedings of the LREC 2010 Workshop on New
             Challenges for NLP Frameworks}},
        pages = {45--50},
        year = 2010,
        month = May,
        day = 22,
        publisher = {ELRA},
        address = {Valletta, Malta},
        note={\url{http://is.muni.cz/publication/884893/en}},
        language={English}
  }
