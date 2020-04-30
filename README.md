gensim – Topic Modelling in Python
==================================

<!--
The following image URLs are obfuscated = proxied and cached through
Google because of Github's proxying issues. See:
https://github.com/RaRe-Technologies/gensim/issues/2805
-->

[![Build Status](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=3600&url=https%3A%2F%2Ftravis-ci.org%2FRaRe-Technologies%2Fgensim.svg%3Fbranch%3Ddevelop)](https://travis-ci.org/RaRe-Technologies/gensim)
[![GitHub release](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=3600&url=https%3A%2F%2Fimg.shields.io%2Fgithub%2Frelease%2Frare-technologies%2Fgensim.svg%3FmaxAge%3D3600)](https://github.com/RaRe-Technologies/gensim/releases)
[![Downloads](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=86400&url=https%3A%2F%2Fimg.shields.io%2Fpypi%2Fdm%2Fgensim%3Fcolor%3Dblue)](https://pepy.tech/project/gensim/month)
[![DOI](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=604800&url=https%3A%2F%2Fzenodo.org%2Fbadge%2FDOI%2F10.13140%2F2.1.2393.1847.svg)](https://doi.org/10.13140/2.1.2393.1847)
[![Mailing List](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=604800&url=https%3A%2F%2Fimg.shields.io%2Fbadge%2F-Mailing%2520List-blue.svg)](https://groups.google.com/forum/#!forum/gensim)
[![Follow](https://images1-focus-opensocial.googleusercontent.com/gadgets/proxy?container=focus&refresh=86400&url=https%3A%2F%2Fimg.shields.io%2Ftwitter%2Ffollow%2Fgensim_py.svg%3Fstyle%3Dsocial%26style%3Dflat%26logo%3Dtwitter%26label%3DFollow%26color%3Dblue)](https://twitter.com/gensim_py)

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

<!--
## :pizza: Hacktoberfest 2019 :beer:

We are accepting PRs for Hacktoberfest!
See [here](HACKTOBERFEST.md) for details.
-->

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

Install the latest version of gensim:

```bash
    pip install --upgrade gensim
```

Or, if you have instead downloaded and unzipped the [source tar.gz]
package:

```bash
    python setup.py install
```

For alternative modes of installation, see the [documentation].

Gensim is being [continuously tested](https://travis-ci.org/RaRe-Technologies/gensim) under Python 3.6, 3.7 and 3.8.
Support for Python 2.7 was dropped in gensim 4.0.0 – install gensim 3.8.3 if you must use Python 2.7.

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
-   [Official API Documentation]

  [QuickStart]: https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html
  [Tutorials]: https://radimrehurek.com/gensim/auto_examples/
  [Official Documentation and Walkthrough]: http://radimrehurek.com/gensim/
  [Official API Documentation]: http://radimrehurek.com/gensim/apiref.html

Support
-------

Ask open-ended or research questions on the [Gensim Mailing List](https://groups.google.com/forum/#!forum/gensim).

Raise bugs on [Github](https://github.com/RaRe-Technologies/gensim/blob/develop/CONTRIBUTING.md) but **make sure you follow the [issue template](https://github.com/RaRe-Technologies/gensim/blob/develop/ISSUE_TEMPLATE.md)**. Issues that are not bugs or fail to follow the issue template will be closed without inspection.

---------

Adopters
--------

| Company | Logo | Industry | Use of Gensim |
|---------|------|----------|---------------|
| [RARE Technologies](http://rare-technologies.com) | ![rare](docs/src/readme_images/rare.png) | ML & NLP consulting | Creators of Gensim – this is us! |
| [Amazon](http://www.amazon.com/) |  ![amazon](docs/src/readme_images/amazon.png) | Retail |  Document similarity. |
| [National Institutes of Health](https://github.com/NIHOPA/pipeline_word2vec) | ![nih](docs/src/readme_images/nih.png) | Health | Processing grants and publications with word2vec. |
| [Cisco Security](http://www.cisco.com/c/en/us/products/security/index.html) | ![cisco](docs/src/readme_images/cisco.png) | Security |  Large-scale fraud detection. |
| [Mindseye](http://www.mindseyesolutions.com/) | ![mindseye](docs/src/readme_images/mindseye.png) | Legal | Similarities in legal documents. |
| [Channel 4](http://www.channel4.com/) | ![channel4](docs/src/readme_images/channel4.png) | Media | Recommendation engine. |
| [Talentpair](http://talentpair.com) | ![talent-pair](docs/src/readme_images/talent-pair.png) | HR | Candidate matching in high-touch recruiting. |
| [Juju](http://www.juju.com/)  | ![juju](docs/src/readme_images/juju.png) | HR | Provide non-obvious related job suggestions. |
| [Tailwind](https://www.tailwindapp.com/) | ![tailwind](docs/src/readme_images/tailwind.png) | Media | Post interesting and relevant content to Pinterest. |
| [Issuu](https://issuu.com/) | ![issuu](docs/src/readme_images/issuu.png) | Media | Gensim's LDA module lies at the very core of the analysis we perform on each uploaded publication to figure out what it's all about. |
| [Search Metrics](http://www.searchmetrics.com/) | ![search-metrics](docs/src/readme_images/search-metrics.png) | Content Marketing | Gensim word2vec used for entity disambiguation in Search Engine Optimisation. |
| [12K Research](https://12k.co/) | ![12k](docs/src/readme_images/12k.png)| Media |   Document similarity analysis on media articles. |
| [Stillwater Supercomputing](http://www.stillwater-sc.com/) | ![stillwater](docs/src/readme_images/stillwater.png) | Hardware | Document comprehension and association with word2vec. |
| [SiteGround](https://www.siteground.com/) |  ![siteground](docs/src/readme_images/siteground.png) | Web hosting | An ensemble search engine which uses different embeddings models and similarities, including word2vec, WMD, and LDA. |
| [Capital One](https://www.capitalone.com/) | ![capitalone](docs/src/readme_images/capitalone.png) | Finance | Topic modeling for customer complaints exploration. |

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

  [citing gensim in academic papers and theses]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:NaGl4SEjCO4C

  [Travis CI for automated testing]: https://travis-ci.org/RaRe-Technologies/gensim
  [design goals]: http://radimrehurek.com/gensim/about.html
  [RaRe Technologies]: http://rare-technologies.com/wp-content/uploads/2016/02/rare_image_only.png%20=10x20
  [rare\_tech]: //rare-technologies.com
  [Talentpair]: https://avatars3.githubusercontent.com/u/8418395?v=3&s=100
  [citing gensim in academic papers and theses]: https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC



  [documentation and Jupyter Notebook tutorials]: https://github.com/RaRe-Technologies/gensim/#documentation
  [Vector Space Model]: http://en.wikipedia.org/wiki/Vector_space_model
  [unsupervised document analysis]: http://en.wikipedia.org/wiki/Latent_semantic_indexing
  [NumPy and Scipy]: http://www.scipy.org/Download
  [ATLAS]: http://math-atlas.sourceforge.net/
  [OpenBLAS]: http://xianyi.github.io/OpenBLAS/
  [source tar.gz]: http://pypi.python.org/pypi/gensim
  [documentation]: http://radimrehurek.com/gensim/install.html
