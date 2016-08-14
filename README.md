gensim – Topic Modelling in Python
==================================

![Travis]![Wheel]

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

Features
--------

-   All algorithms are **memory-independent** w.r.t. the corpus size
    (can process input larger than RAM, streamed, out-of-core),
-   **Intuitive interfaces**
    -   easy to plug in your own input corpus/datastream (trivial
        streaming API)
    -   easy to extend with other Vector Space algorithms (trivial
        transformation API)
-   Efficient multicore implementations of popular algorithms, such as
    online **Latent Semantic Analysis (LSA/LSI/SVD)**, **Latent
    Dirichlet Allocation (LDA)**, **Random Projections (RP)**,
    **Hierarchical Dirichlet Process (HDP)** or **word2vec deep
    learning**.
-   **Distributed computing**: can run *Latent Semantic Analysis* and
    *Latent Dirichlet Allocation* on a cluster of computers.
-   Extensive [documentation and Jupyter Notebook tutorials].

If this feature list left you scratching your head, you can first read
more about the [Vector Space Model] and [unsupervised document analysis]
on Wikipedia.

Installation
------------

This software depends on [NumPy and Scipy], two Python packages for
scientific computing. You must have them installed prior to installing
gensim.

It is also recommended you install a fast BLAS library before installing
NumPy. This is optional, but using an optimized BLAS such as [ATLAS] or
[OpenBLAS] is known to improve performance by as much as an order of
magnitude. On OS X, NumPy picks up the BLAS that comes with it
automatically, so you don’t need to do anything special.

The simple way to install gensim is:

    pip install -U gensim

Or, if you have instead downloaded and unzipped the [source tar.gz]
package, you’d run:

    python setup.py test
    python setup.py install

For alternative modes of installation (without root privileges,
development installation, optional install features), see the
[documentation].

This version has been tested under Python 2.6, 2.7, 3.3, 3.4 and 3.5
(support for Python 2.5 was dropped in gensim 0.10.0; install gensim
0.9.1 if you *must* use Python 2.5). Gensim’s github repo is hooked
against [Travis CI for automated testing] on every commit push and pull
request.

How come gensim is so fast and memory efficient? Isn’t it pure Python, and isn’t Python slow and greedy?
--------------------------------------------------------------------------------------------------------

Many scientific algorithms can be expressed in terms of large matrix
operations (see the BLAS note above). Gensim taps into these low-level
BLAS libraries, by means of its dependency on NumPy. So while
gensim-the-top-level-code is pure Python, it actually executes highly
optimized Fortran/C under the hood, including multithreading (if your
BLAS is so configured).

Memory-wise, gensim makes heavy use of Python’s built-in generators and
iterators for streamed data processing. Memory efficiency was one of
gensim’s [design goals], and is a central feature of gensim, rather than
something bolted on as an afterthought.

Documentation
-------------

-   [QuickStart]
-   [Tutorials]
-   [Tutorial Videos]
-   [Official Documentation and Walkthrough]

  [QuickStart]: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim%20Quick%20Start.ipynb
  [Tutorials]: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials
  [Tutorial Videos]: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#videos
  [Official Documentation and Walkthrough]: http://radimrehurek.com/gensim/
  
---------

Adopters
--------



| Name                                   | Logo                                                                                                                           | URL                                                                                              | Description                                                                                                                                                                                                           |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                         
| RaRe Technologies                            | <img src="http://rare-technologies.com/wp-content/uploads/2016/02/rare_image_only.png" width="100"> | [rare-technologies.com](http://rare-technologies.com)                                                           | Machine learning & NLP consulting and training. Gensim creators. |
| Talentpair                            | ![Talentpair](https://avatars3.githubusercontent.com/u/8418395?v=3&s=100)                                                 | [talentpair.com](http://talentpair.com)                                                           | Data science driving high-touch recruiting                                                    | 
| Tailwind          | <img src="http://blog.tailwindapp.com/wp-content/uploads/2013/10/Tailwind-Square-Logo-Blue-White-300x300.png" width="100"> | [Tailwindapp.com](https://www.tailwindapp.com/)| Post interesting and relevant content to Pinterest              |
| Issuu          | <img src="https://static.isu.pub/fe/issuu-brandpages/s3/155/press/assets/brand_package_zip/issuu%20logos/png/issuu-logo-stacked-colour.png" width="100"> | [Issuu.com](https://issuu.com/)| Gensim’s LDA module lies at the very core of the analysis we perform on each uploaded publication to figure out what it’s all about.
| Sports Authority        | <img src="http://www.sportsauthority.com/images/tsa_logo_red_bg.gif" width="100"> | [sportsauthority.com](http://www.sportsauthority.com)| Text mining of customer surveys and social media sources |
| Search Metrics        | <img src="http://www.searchmetrics.com/wp-content/uploads/Logo_searchmetrics_Webversion.png" width="100"> | [searchmetrics.com](http://www.searchmetrics.com/)| Gensim word2vec used for entity disambiguation in Search Engine Optimisation
| Cisco Security        | <img src="https://supportforums.cisco.com/sites/default/files/legacy/1/6/1/2161-CiscoSystems.gif" width="100"> | [cisco.com](http://www.cisco.com/c/en/us/products/security/index.html)|  Large-scale fraud detection
|  12K Research         | <img src="https://techberlin.com/media/CACHE/images/threesixty/FtA5ANuJ/1aa5b8517ec65ef1c7d69c9bda5f9a3c.jpg" width="100"> | [12k.co](https://12k.co/)|   Document similarity analysis on media articles
|  National Institutes of Health         | <img src="https://www.nih.gov/sites/default/files/styles/featured_media_breakpoint-large/public/about-nih/2012-logo.png" width="100"> | [github/NIHOPA](https://github.com/NIHOPA/pipeline_word2vec)|   Processing grants and publications with word2vec
|  Codeq LLC         | <img src="https://codeq.com/wp-content/themes/codeq/assets/img/logo.svg" width="100"> | [codeq.com](https://codeq.com)|   Document classification with word2vec
| Mass Cognition     | <img src="http://static1.squarespace.com/static/5637b16ee4b050255657c537/t/56a683bf9cadb6bf86a0ea13/1461016648294/?format=1500w" width="100"> | [masscognition.com](http://www.masscognition.com/)                                  | Topic analysis service for consumer text data and general text data |
| Stillwater Supercomputing     | <img src="http://www.stillwater-sc.com/img/stillwater-logo.png" width="100"> | [stillwater-sc.com](http://www.stillwater-sc.com/)                                  | Document comprehension and association with word2vec |
| Channel 4     | <img src="http://www.channel4.com/static/info/images/lib/c4logo_2015_info_corporate.jpg" width="100"> | [channel4.com](http://www.channel4.com/)                                  | Recommendation engine |
| Amazon     |  <img src="http://g-ec2.images-amazon.com/images/G/01/social/api-share/amazon_logo_500500._V323939215_.png" width="100"> | [amazon.com](http://www.amazon.com/)                                  |  Document similarity|

-------

Citing gensim
------------

When [citing gensim in academic papers and theses], please use this
BibTeX entry:

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

  [citing gensim in academic papers and theses]: https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC

  [Travis CI for automated testing]: https://travis-ci.org/RaRe-Technologies/gensim
  [design goals]: http://radimrehurek.com/gensim/about.html
  [RaRe Technologies]: http://rare-technologies.com/wp-content/uploads/2016/02/rare_image_only.png%20=10x20
  [rare\_tech]: //rare-technologies.com
  [Talentpair]: https://avatars3.githubusercontent.com/u/8418395?v=3&s=100
  [citing gensim in academic papers and theses]: https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC

  [Travis]: https://img.shields.io/travis/RaRe-Technologies/gensim/develop.svg
  [Wheel]: https://img.shields.io/pypi/wheel/gensim.svg
  [documentation and Jupyter Notebook tutorials]: https://github.com/RaRe-Technologies/gensim/#documentation
  [Vector Space Model]: http://en.wikipedia.org/wiki/Vector_space_model
  [unsupervised document analysis]: http://en.wikipedia.org/wiki/Latent_semantic_indexing
  [NumPy and Scipy]: http://www.scipy.org/Download
  [ATLAS]: http://math-atlas.sourceforge.net/
  [OpenBLAS]: http://xianyi.github.io/OpenBLAS/
  [source tar.gz]: http://pypi.python.org/pypi/gensim
  [documentation]: http://radimrehurek.com/gensim/install.html
