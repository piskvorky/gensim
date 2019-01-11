Changes
===========
## 3.6.0, 2018-09-20

### :star2: New features
* File-based training for `*2Vec` models (__[@persiyanov](https://github.com/persiyanov)__, [#2127](https://github.com/RaRe-Technologies/gensim/pull/2127) & [#2078](https://github.com/RaRe-Technologies/gensim/pull/2078) & [#2048](https://github.com/RaRe-Technologies/gensim/pull/2048))

  New training mode for `*2Vec` models (word2vec, doc2vec, fasttext) that allows model training to scale linearly with the number of cores (full GIL elimination). The result of our Google Summer of Code 2018 project by Dmitry Persiyanov.

  **Benchmark**
  - Dataset: `full English Wikipedia`
  - Cloud: `GCE`
  - CPU: `Intel(R) Xeon(R) CPU @ 2.30GHz 32 cores`
  - BLAS: `MKL`


  | Model | Queue-based version [sec] | File-based version [sec] | speed up | Accuracy (queue-based) | Accuracy (file-based) |
  |-------|------------|--------------------|----------|----------------|-----------------------|
  | Word2Vec | 9230 | **2437** | **3.79x** | 0.754 (± 0.003) | 0.750 (± 0.001) |
  | Doc2Vec | 18264 | **2889** | **6.32x** | 0.721 (± 0.002) | 0.683 (± 0.003) |
  | FastText | 16361 | **10625** | **1.54x** | 0.642 (± 0.002) | 0.660 (± 0.001) |

  Usage:

  ```python
  import gensim.downloader as api
  from multiprocessing import cpu_count
  from gensim.utils import save_as_line_sentence
  from gensim.test.utils import get_tmpfile
  from gensim.models import Word2Vec, Doc2Vec, FastText


  # Convert any corpus to the needed format: 1 document per line, words delimited by " "
  corpus = api.load("text8")
  corpus_fname = get_tmpfile("text8-file-sentence.txt")
  save_as_line_sentence(corpus, corpus_fname)

  # Choose num of cores that you want to use (let's use all, models scale linearly now!)
  num_cores = cpu_count()

  # Train models using all cores
  w2v_model = Word2Vec(corpus_file=corpus_fname, workers=num_cores)
  d2v_model = Doc2Vec(corpus_file=corpus_fname, workers=num_cores)
  ft_model = FastText(corpus_file=corpus_fname, workers=num_cores)

  ```
  [Read notebook tutorial with full description.](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Any2Vec_Filebased.ipynb)


### :+1: Improvements

* Add scikit-learn wrapper for `FastText` (__[@mcemilg](https://github.com/mcemilg)__, [#2178](https://github.com/RaRe-Technologies/gensim/pull/2178))
* Add multiprocessing support for `BM25` (__[@Shiki-H](https://github.com/Shiki-H)__, [#2146](https://github.com/RaRe-Technologies/gensim/pull/2146))
* Add `name_only` option for downloader api (__[@aneesh-joshi](https://github.com/aneesh-joshi)__, [#2143](https://github.com/RaRe-Technologies/gensim/pull/2143))
* Make `word2vec2tensor` script compatible with `python3` (__[@vsocrates](https://github.com/vsocrates)__, [#2147](https://github.com/RaRe-Technologies/gensim/pull/2147))
* Add custom filter for `Wikicorpus` (__[@mattilyra](https://github.com/mattilyra)__, [#2089](https://github.com/RaRe-Technologies/gensim/pull/2089))
* Make `similarity_matrix` support non-contiguous dictionaries (__[@Witiko](https://github.com/Witiko)__, [#2047](https://github.com/RaRe-Technologies/gensim/pull/2047))


### :red_circle: Bug fixes

* Fix memory consumption in `AuthorTopicModel` (__[@philipphager](https://github.com/philipphager)__, [#2122](https://github.com/RaRe-Technologies/gensim/pull/2122))
* Correctly process empty documents in `AuthorTopicModel` (__[@probinso](https://github.com/probinso)__, [#2133](https://github.com/RaRe-Technologies/gensim/pull/2133))
* Fix ZeroDivisionError `keywords` issue with short input (__[@LShostenko](https://github.com/LShostenko)__, [#2154](https://github.com/RaRe-Technologies/gensim/pull/2154))
* Fix `min_count` handling in phrases detection using `npmi_scorer` (__[@lopusz](https://github.com/lopusz)__, [#2072](https://github.com/RaRe-Technologies/gensim/pull/2072))
* Remove duplicate count from `Phraser` log message (__[@robguinness](https://github.com/robguinness)__, [#2151](https://github.com/RaRe-Technologies/gensim/pull/2151))
* Replace `np.integer` -> `np.int` in `AuthorTopicModel` (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#2145](https://github.com/RaRe-Technologies/gensim/pull/2145))


### :books: Tutorial and doc improvements

* Update docstring with new analogy evaluation method (__[@akutuzov](https://github.com/akutuzov)__, [#2130](https://github.com/RaRe-Technologies/gensim/pull/2130))
* Improve `prune_at` parameter description for `gensim.corpora.Dictionary` (__[@yxonic](https://github.com/yxonic)__, [#2128](https://github.com/RaRe-Technologies/gensim/pull/2128))
* Fix `default` -> `auto` prior parameter in documentation for lda-related models (__[@Laubeee](https://github.com/Laubeee)__, [#2156](https://github.com/RaRe-Technologies/gensim/pull/2156))
* Use heading instead of bold style in `gensim.models.translation_matrix` (__[@nzw0301](https://github.com/nzw0301)__, [#2164](https://github.com/RaRe-Technologies/gensim/pull/2164))
* Fix quote of vocabulary from `gensim.models.Word2Vec` (__[@nzw0301](https://github.com/nzw0301)__, [#2161](https://github.com/RaRe-Technologies/gensim/pull/2161))
* Replace deprecated parameters with new in docstring of `gensim.models.Doc2Vec` (__[@xuhdev](https://github.com/xuhdev)__, [#2165](https://github.com/RaRe-Technologies/gensim/pull/2165))
* Fix formula in Mallet documentation (__[@Laubeee](https://github.com/Laubeee)__, [#2186](https://github.com/RaRe-Technologies/gensim/pull/2186))
* Fix minor semantic issue in docs for `Phrases` (__[@RunHorst](https://github.com/RunHorst)__, [#2148](https://github.com/RaRe-Technologies/gensim/pull/2148))
* Fix typo in documentation (__[@KenjiOhtsuka](https://github.com/KenjiOhtsuka)__, [#2157](https://github.com/RaRe-Technologies/gensim/pull/2157))
* Additional documentation fixes (__[@piskvorky](https://github.com/piskvorky)__, [#2121](https://github.com/RaRe-Technologies/gensim/pull/2121))

### :warning: Deprecations (will be removed in the next major release)

* Remove
    - `gensim.models.wrappers.fasttext` (obsoleted by the new native `gensim.models.fasttext` implementation)
    - `gensim.examples`
    - `gensim.nosy`
    - `gensim.scripts.word2vec_standalone`
    - `gensim.scripts.make_wiki_lemma`
    - `gensim.scripts.make_wiki_online`
    - `gensim.scripts.make_wiki_online_lemma`
    - `gensim.scripts.make_wiki_online_nodebug`
    - `gensim.scripts.make_wiki` (all of these obsoleted by the new native  `gensim.scripts.segment_wiki` implementation)
    - "deprecated" functions and attributes

* Move
    - `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
    - `gensim.summarization` ➡ `gensim.models.summarization`
    - `gensim.topic_coherence` ➡ `gensim.models._coherence`
    - `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
    - `gensim.parsing.*` ➡ `gensim.utils.text_utils`


## 3.5.0, 2018-07-06

This release comprises a glorious 38 pull requests from 28 contributors. Most of the effort went into improving the documentation—hence the release code name "Docs 💬"!

Apart from the **massive overhaul of all Gensim documentation** (including docstring style and examples—[you asked for it](https://rare-technologies.com/gensim-survey-2018/)), we also managed to sneak in some new functionality and a number of bug fixes. As usual, see the notes below for a complete list, with links to pull requests for more details.

**Huge thanks to all contributors!** Nobody loves working on documentation. 3.5.0 is a result of several months of laborious, unglamorous, and sometimes invisible work. Enjoy!


### :books: Documentation improvements

* Overhaul documentation for `*2vec` models (__[@steremma](https://github.com/steremma)__ & __[@piskvorky](https://github.com/piskvorky)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1944](https://github.com/RaRe-Technologies/gensim/pull/1944), [#2087](https://github.com/RaRe-Technologies/gensim/pull/2087))
* Fix documentation for LDA-related models (__[@steremma](https://github.com/steremma)__ & __[@piskvorky](https://github.com/piskvorky)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#2026](https://github.com/RaRe-Technologies/gensim/pull/2026))
* Fix documentation for utils, corpora, inferfaces (__[@piskvorky](https://github.com/piskvorky)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#2096](https://github.com/RaRe-Technologies/gensim/pull/2096))
* Update non-API docs (about, intro, license etc) (__[@piskvorky](https://github.com/piskvorky)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#2101](https://github.com/RaRe-Technologies/gensim/pull/2101))
* Refactor documentation for `gensim.models.phrases` (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1950](https://github.com/RaRe-Technologies/gensim/pull/1950))
* Fix HashDictionary documentation (__[@piskvorky](https://github.com/piskvorky)__, [#2073](https://github.com/RaRe-Technologies/gensim/pull/2073))
* Fix docstrings for `gensim.models.AuthorTopicModel` (__[@souravsingh](https://github.com/souravsingh)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1907](https://github.com/RaRe-Technologies/gensim/pull/1907))
* Fix docstrings for HdpModel, lda_worker & lda_dispatcher (__[@gyanesh-m](https://github.com/gyanesh-m)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1912](https://github.com/RaRe-Technologies/gensim/pull/1912))
* Fix format & links for `gensim.similarities.docsim` (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#2030](https://github.com/RaRe-Technologies/gensim/pull/2030))
* Remove duplication of class documentation for `IndexedCorpus` (__[@darindf](https://github.com/darindf)__, [#2033](https://github.com/RaRe-Technologies/gensim/pull/2033))
* Refactor documentation for `gensim.models.coherencemodel` (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1933](https://github.com/RaRe-Technologies/gensim/pull/1933))
* Fix docstrings for `gensim.sklearn_api` (__[@steremma](https://github.com/steremma)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1895](https://github.com/RaRe-Technologies/gensim/pull/1895))
* Disable google-style docstring support (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#2106](https://github.com/RaRe-Technologies/gensim/pull/2106))
* Fix docstring of `gensim.models.KeyedVectors.similarity_matrix` (__[@Witiko](https://github.com/Witiko)__, [#1971](https://github.com/RaRe-Technologies/gensim/pull/1971))
* Consistently use `smart_open()` instead of `open()` in notebooks (__[@sharanry](https://github.com/sharanry)__, [#1812](https://github.com/RaRe-Technologies/gensim/pull/1812))


### :star2: New features:

* Add `add_entity` method to `KeyedVectors` to allow adding word vectors manually (__[@persiyanov](https://github.com/persiyanov)__, [#1957](https://github.com/RaRe-Technologies/gensim/pull/1957))
* Add inference for new unseen author to `AuthorTopicModel` (__[@Stamenov](https://github.com/Stamenov)__, [#1766](https://github.com/RaRe-Technologies/gensim/pull/1766))
* Add `evaluate_word_analogies` (will replace `accuracy`) method to `KeyedVectors` (__[@akutuzov](https://github.com/akutuzov)__, [#1935](https://github.com/RaRe-Technologies/gensim/pull/1935))
* Add Pivot Normalization to `TfidfModel` (__[@markroxor](https://github.com/markroxor)__, [#1780](https://github.com/RaRe-Technologies/gensim/pull/1780))



### :+1: Improvements

* Allow initialization with `max_final_vocab` in lieu of `min_count` in `Word2Vec`(__[@aneesh-joshi](https://github.com/aneesh-joshi)__, [#1915](https://github.com/RaRe-Technologies/gensim/pull/1915))
* Add `dtype` argument for `chunkize_serial` in `LdaModel` (__[@darindf](https://github.com/darindf)__, [#2027](https://github.com/RaRe-Technologies/gensim/pull/2027))
* Increase performance in `Phrases.analyze_sentence` (__[@JonathanHourany](https://github.com/JonathanHourany)__, [#2070](https://github.com/RaRe-Technologies/gensim/pull/2070))
* Add `ns_exponent` parameter to control the negative sampling distribution for `*2vec` models (__[@fernandocamargoti](https://github.com/fernandocamargoti)__, [#2093](https://github.com/RaRe-Technologies/gensim/pull/2093))


### :red_circle: Bug fixes:


* Fix `Doc2Vec.infer_vector` + notebook cleanup (__[@gojomo](https://github.com/gojomo)__, [#2103](https://github.com/RaRe-Technologies/gensim/pull/2103))
* Fix linear decay for learning rate in `Doc2Vec.infer_vector` (__[@umangv](https://github.com/umangv)__, [#2063](https://github.com/RaRe-Technologies/gensim/pull/2063))
* Fix negative sampling floating-point error for `gensim.models.Poincare (__[@jayantj](https://github.com/jayantj)__, [#1959](https://github.com/RaRe-Technologies/gensim/pull/1959))
* Fix loading `word2vec` and `doc2vec` models saved using old Gensim versions (__[@manneshiva](https://github.com/manneshiva)__, [#2012](https://github.com/RaRe-Technologies/gensim/pull/2012))
* Fix `SoftCosineSimilarity.get_similarities` on corpora ssues/1955) (__[@Witiko](https://github.com/Witiko)__, [#1972](https://github.com/RaRe-Technologies/gensim/pull/1972))
* Fix return dtype for `matutils.unitvec` according to input dtype (__[@o-P-o](https://github.com/o-P-o)__, [#1992](https://github.com/RaRe-Technologies/gensim/pull/1992))
* Fix passing empty dictionary to `gensim.corpora.WikiCorpus` (__[@steremma](https://github.com/steremma)__, [#2042](https://github.com/RaRe-Technologies/gensim/pull/2042))
* Fix bug in `Similarity.query_shards` in multiprocessing case (__[@bohea](https://github.com/bohea)__, [#2044](https://github.com/RaRe-Technologies/gensim/pull/2044))
* Fix SMART from TfidfModel for case when `df == "n"` (__[@PeteBleackley](https://github.com/PeteBleackley)__, [#2021](https://github.com/RaRe-Technologies/gensim/pull/2021))
* Fix OverflowError when loading a large term-document matrix in compiled MatrixMarket format (__[@arlenk](https://github.com/arlenk)__, [#2001](https://github.com/RaRe-Technologies/gensim/pull/2001))
* Update rules for removing table markup from Wikipedia dumps (__[@chaitaliSaini](https://github.com/chaitaliSaini)__, [#1954](https://github.com/RaRe-Technologies/gensim/pull/1954))
* Fix `_is_single` from `Phrases` for case when corpus is a NumPy array (__[@rmalouf](https://github.com/rmalouf)__, [#1987](https://github.com/RaRe-Technologies/gensim/pull/1987))
* Fix tests for `EuclideanKeyedVectors.similarity_matrix` (__[@Witiko](https://github.com/Witiko)__, [#1984](https://github.com/RaRe-Technologies/gensim/pull/1984))
* Fix deprecated parameters in `D2VTransformer` and `W2VTransformer`(__[@MritunjayMohitesh](https://github.com/MritunjayMohitesh)__, [#1945](https://github.com/RaRe-Technologies/gensim/pull/1945))
* Fix `Doc2Vec.infer_vector` after loading old `Doc2Vec` (`gensim<=3.2`)(__[@manneshiva](https://github.com/manneshiva)__, [#1974](https://github.com/RaRe-Technologies/gensim/pull/1974))
* Fix inheritance chain for `load_word2vec_format` (__[@DennisChen0307](https://github.com/DennisChen0307)__, [#1968](https://github.com/RaRe-Technologies/gensim/pull/1968))
* Update Keras version (avoid bug from `keras==2.1.5`) (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1963](https://github.com/RaRe-Technologies/gensim/pull/1963))



### :warning: Deprecations (will be removed in the next major release)
* Remove
    - `gensim.models.wrappers.fasttext` (obsoleted by the new native `gensim.models.fasttext` implementation)
    - `gensim.examples`
    - `gensim.nosy`
    - `gensim.scripts.word2vec_standalone`
    - `gensim.scripts.make_wiki_lemma`
    - `gensim.scripts.make_wiki_online`
    - `gensim.scripts.make_wiki_online_lemma`
    - `gensim.scripts.make_wiki_online_nodebug`
    - `gensim.scripts.make_wiki` (all of these obsoleted by the new native  `gensim.scripts.segment_wiki` implementation)
    - "deprecated" functions and attributes

* Move
    - `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
    - `gensim.summarization` ➡ `gensim.models.summarization`
    - `gensim.topic_coherence` ➡ `gensim.models._coherence`
    - `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
    - `gensim.parsing.*` ➡ `gensim.utils.text_utils`


## 3.4.0, 2018-03-01

### :star2: New features:
* Massive optimizations of `gensim.models.LdaModel`: much faster training, using Cython. (__[@arlenk](https://github.com/arlenk)__, [#1767](https://github.com/RaRe-Technologies/gensim/pull/1767))
    - Training benchmark :boom:

      | dataset | old LDA [sec] | optimized LDA [sec] | speed up |
      |---------|---------------|---------------------|---------|
      | nytimes | 3473 | **1975** | **1.76x** |
      | enron   | 774 | **437** |  **1.77x** |

    - This change **affects all models that depend on `LdaModel`**, such as `LdaMulticore`, `LdaSeqModel`, `AuthorTopicModel`.
* Huge speed-ups to corpus I/O with `MmCorpus` (Cython) (__[@arlenk](https://github.com/arlenk)__, [#1825](https://github.com/RaRe-Technologies/gensim/pull/1825))
    - File reading benchmark

      |     dataset   | file compressed? | old MmReader [sec] | optimized MmReader [sec] | speed up      |
      |---------------|:-----------:|:------------:|:------------------:|:-------------:|
      | enron         |      no     |      22.3    |     **2.6**        |    **8.7x**   |
      |               |     yes     |      37.3    |    **14.4**        |    **2.6x**   |
      | nytimes       |      no     |      419.3   |    **49.2**        |    **8.5x**   |
      |               |     yes     |      686.2   |    **275.1**       |    **2.5x**   |
      | text8         |      no     |      25.4    |     **2.5**        |   **10.1x**   |
      |               |     yes     |      41.9    |    **17.0**        |    **2.5x**   |

    - Overall, a **2.5x** speedup for compressed `.mm.gz` input and **8.5x** :fire::fire::fire: for uncompressed plaintext `.mm`.

* Performance and memory optimization to `gensim.models.FastText` :rocket: (__[@jbaiter](https://github.com/jbaiter)__, [#1916](https://github.com/RaRe-Technologies/gensim/pull/1916))
    - Benchmark (first 500,000 articles from English Wikipedia)

      | Metric                 | old FastText         | optimized FastText  | improvement |
      | -----------------------| -----------------| -------------------|-------------|
      | Training time (1 epoch)     |      4823.4s (80.38 minutes)    |  **1873.6s (31.22 minutes)**  | **2.57x** |
      | Training time (full) | 1h 26min 13s | **36min 43s** | **2.35x** |
      | Training words/sec   |  72,781  | **187,366** | **2.57x** |
      | Training peak memory   | 5.2 GB  |  **3.7 GB** | **1.4x** |

    - Overall, a **2.5x** speedup & memory usage reduced by **30%**.

* Implemented [Soft Cosine Measure](https://en.wikipedia.org/wiki/Cosine_similarity#Soft_cosine_measure) (__[@Witiko](https://github.com/Witiko)__, [#1827](https://github.com/RaRe-Technologies/gensim/pull/1827))
    - New method for assessing document similarity, a nice faster alternative to [WMD, Word Mover's Distance](http://proceedings.mlr.press/v37/kusnerb15.pdf)
    - Benchmark

      | Technique | MAP score | Duration     |
      |-----------|-----------|--------------|
      | softcossim| **45.99** | **1.24 sec** |
      | wmd-relax | 44.48     | 12.22 sec    |
      |  cossim   | 44.22     | 4.39 sec     |
      | wmd-gensim| 44.08     | 98.29 sec    |

    - [Soft Cosine notebook with detailed description, examples & benchmarks](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb)
    - Related papers:
        - [Soft Similarity and Soft Cosine Measure: Similarity of Features in Vector Space Model](http://www.scielo.org.mx/pdf/cys/v18n3/v18n3a7.pdf)
        - [SimBow at SemEval-2017 Task 3: Soft-Cosine Semantic Similarity between Questions for Community Question Answering](http://www.aclweb.org/anthology/S17-2051)
        - [Vector Space Representations in IR](https://github.com/witiko-masters-thesis/thesis/blob/master/main.pdf)


### :+1: Improvements:
* New method to show the Gensim installation parameters: `python -m gensim.scripts.package_info --info`. Use this when reporting problems, for easier debugging. Fix #1902 (__[@sharanry](https://github.com/sharanry)__, [#1903](https://github.com/RaRe-Technologies/gensim/pull/1903))
* Added a flag to optionally skip network-related tests, to help maintainers avoid network issues with CI services (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1930](https://github.com/RaRe-Technologies/gensim/pull/1930))
* Added `license` field to `setup.py`, allowing the use of tools like `pip-licenses` (__[@nils-werner](https://github.com/nils-werner)__, [#1909](https://github.com/RaRe-Technologies/gensim/pull/1909))

### :red_circle: Bug fixes:
* Fix Python 3 compatibility for `gensim.corpora.UciCorpus.save_corpus` (__[@darindf](https://github.com/darindf)__, [#1875](https://github.com/RaRe-Technologies/gensim/pull/1875))
* Add `wv` property to KeyedVectors for backward compatibility. Fix #1882 (__[@manneshiva](https://github.com/manneshiva)__, [#1884](https://github.com/RaRe-Technologies/gensim/pull/1884))
* Fix deprecation warning from `inspect.getargspec`. Fix #1878 (__[@aneesh-joshi](https://github.com/aneesh-joshi)__, [#1887](https://github.com/RaRe-Technologies/gensim/pull/1887))
* Add `LabeledSentence` to `gensim.models.doc2vec` for backward compatibility. Fix #1886 (__[@manneshiva](https://github.com/manneshiva)__, [#1891](https://github.com/RaRe-Technologies/gensim/pull/1891))
* Fix empty output bug in `Phrases` (when using `model[tokens]` twice). Fix #1401 (__[@sj29-innovate](https://github.com/sj29-innovate)__, [#1853](https://github.com/RaRe-Technologies/gensim/pull/1853))
* Fix type problems for `D2VTransformer.fit_transform`. Fix #1834 (__[@Utkarsh-Mishra-CIC](https://github.com/Utkarsh-Mishra-CIC)__, [#1845](https://github.com/RaRe-Technologies/gensim/pull/1845))
* Fix `datatype` parameter for `KeyedVectors.load_word2vec_format`. Fix #1682 (__[@pushpankar](https://github.com/pushpankar)__, [#1819](https://github.com/RaRe-Technologies/gensim/pull/1819))
* Fix deprecated parameters in `doc2vec-lee` notebook (__[@TheFlash10](https://github.com/TheFlash10)__, [#1918](https://github.com/RaRe-Technologies/gensim/pull/1918))
* Fix file-like closing bug in `gensim.corpora.MmCorpus`. Fix #1869 (__[@sj29-innovate](https://github.com/sj29-innovate)__, [#1911](https://github.com/RaRe-Technologies/gensim/pull/1911))
* Fix precision problem in `test_similarities.py`, no more FP fails. (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1928](https://github.com/RaRe-Technologies/gensim/pull/1928))
* Fix encoding in Lee corpus reader. (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1931](https://github.com/RaRe-Technologies/gensim/pull/1931))
* Fix OOV pairs counter in `WordEmbeddingsKeyedVectors.evaluate_word_pairs`. (__[@akutuzov](https://github.com/akutuzov)__, [#1934](https://github.com/RaRe-Technologies/gensim/pull/1934))


### :books: Tutorial and doc improvements:
* Fix example block for `gensim.models.Word2Vec` (__[@nzw0301](https://github.com/nzw0301)__, [#1870](https://github.com/RaRe-Technologies/gensim/pull/1876))
* Fix `doc2vec-lee` notebook (__[@numericlee](https://github.com/numericlee)__, [#1870](https://github.com/RaRe-Technologies/gensim/pull/1870))
* Store images from `README.md` directly in repository. Fix #1849 (__[@ibrahimsharaf](https://github.com/ibrahimsharaf)__, [#1861](https://github.com/RaRe-Technologies/gensim/pull/1861))
* Add windows venv activate command to `CONTRIBUTING.md` (__[@aneesh-joshi](https://github.com/aneesh-joshi)__, [#1880](https://github.com/RaRe-Technologies/gensim/pull/1880))
* Add anaconda-cloud badge. Partial fix #1901 (__[@sharanry](https://github.com/sharanry)__, [#1905](https://github.com/RaRe-Technologies/gensim/pull/1905))
* Fix docstrings for lsi-related code (__[@steremma](https://github.com/steremma)__, [#1892](https://github.com/RaRe-Technologies/gensim/pull/1892))
* Fix parameter description of  `sg` parameter for `gensim.models.word2vec` (__[@mdcclv](https://github.com/mdcclv)__, [#1919](https://github.com/RaRe-Technologies/gensim/pull/1919))
* Refactor documentation for `gensim.similarities.docsim` and `MmCorpus-related`. (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1910](https://github.com/RaRe-Technologies/gensim/pull/1910))
* Fix docstrings for `gensim.test.utils` (__[@yurkai](https://github.com/yurkai)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1904](https://github.com/RaRe-Technologies/gensim/pull/1904))
* Refactor docstrings for `gensim.scripts`. Partial fix #1665 (__[@yurkai](https://github.com/yurkai)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1792](https://github.com/RaRe-Technologies/gensim/pull/1792))
* Refactor API reference `gensim.corpora`. Partial fix #1671 (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1835](https://github.com/RaRe-Technologies/gensim/pull/1835))
* Fix documentation for `gensim.models.wrappers` (__[@kakshay21](https://github.com/kakshay21)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1859](https://github.com/RaRe-Technologies/gensim/pull/1859))
* Fix docstrings for `gensim.interfaces` (__[@yurkai](https://github.com/yurkai)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1913](https://github.com/RaRe-Technologies/gensim/pull/1913))


### :warning: Deprecations (will be removed in the next major release)
* Remove
    - `gensim.models.wrappers.fasttext` (obsoleted by the new native `gensim.models.fasttext` implementation)
    - `gensim.examples`
    - `gensim.nosy`
    - `gensim.scripts.word2vec_standalone`
    - `gensim.scripts.make_wiki_lemma`
    - `gensim.scripts.make_wiki_online`
    - `gensim.scripts.make_wiki_online_lemma`
    - `gensim.scripts.make_wiki_online_nodebug`
    - `gensim.scripts.make_wiki` (all of these obsoleted by the new native  `gensim.scripts.segment_wiki` implementation)
    - "deprecated" functions and attributes

* Move
    - `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
    - `gensim.summarization` ➡ `gensim.models.summarization`
    - `gensim.topic_coherence` ➡ `gensim.models._coherence`
    - `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
    - `gensim.parsing.*` ➡ `gensim.utils.text_utils`


## 3.3.0, 2018-02-02

:star2: New features:
* Re-designed all "*2vec" implementations (__[@manneshiva](https://github.com/manneshiva)__, [#1777](https://github.com/RaRe-Technologies/gensim/pull/1777))
    - Modular organization of `Word2Vec`, `Doc2Vec`, `FastText`, etc ..., making it easier to add new models in the future and re-use code
    - Fully backward compatible (even with loading models stored by a previous Gensim version)
    - [Detailed documentation for the *2vec refactoring project](https://github.com/manneshiva/gensim/wiki/Any2Vec-Refactoring-Summary)

* Improve `gensim.scripts.segment_wiki` by retaining interwiki links. Fix #1712
 (__[@steremma](https://github.com/steremma)__, [PR #1839](https://github.com/RaRe-Technologies/gensim/pull/1839))
    - Optionally extract interlinks from Wikipedia pages (use the `--include-interlinks` option). This will output one additional JSON dict for each article:
        ```
        {
            "interlinks": {
                "article title 1": "interlink text 1",
                "article title 2": "interlink text 2",
                ...
            }
        }
        ```

    - Example: extract the Wikipedia graph with article links as edges, from a raw Wikipedia dump:
        ```bash
        python -m gensim.scripts.segment_wiki --include-interlinks --file ~/Downloads/enwiki-latest-pages-articles.xml.bz2 --output ~/Desktop/enwiki-latest.jsonl.gz
        ```
        - Read this field from the `segment_wiki` output:

        ```python
        import json
        from smart_open import smart_open

        with smart_open("enwiki-latest.jsonl.gz") as infile:
            for doc in infile:
                doc = json.loads(doc)

                src_node = doc['title']
                dst_nodes = doc['interlinks'].keys()

                print(u"Source node: {}".format(src_node))
                print(u"Destination nodes: {}".format(u", ".join(dst_nodes)))
                break

        """
        OUTPUT:

        Source node: Anarchism
        Destination nodes: anarcha-feminist, Ivan Illich, Adolf Brand, Josiah Warren, will (philosophy), anarcha-feminism, Anarchism in Mexico, Lysander Spooner, English Civil War, G8, Sebastien Faure, Nihilist movement, Sébastien Faure, Left-wing politics, imamate, Pierre Joseph Proudhon, anarchist communism, Università popolare (Italian newspaper), 1848 Revolution, Synthesis anarchism, labour movement, anarchist communists, collectivist anarchism, polyamory, post-humanism, postcolonialism, anti war movement, State (polity), security culture, Catalan people, Stoicism, Progressive education, stateless society, Umberto I of Italy, German language, Anarchist schools of thought, NEFAC, Jacques Ellul, Spanish Communist Party, Crypto-anarchism, ruling class, non-violence, Platformist, The History of Sexuality, Revolutions of 1917–23, Federación Anarquista Ibérica, propaganda of the deed, William B. Greene, Platformism, mutually exclusive, Fraye Arbeter Shtime, Adolf Hitler, oxymoron, Paris Commune, Anarchism in Italy#Postwar years and today, Oranienburg, abstentionism, Free Society, Henry David Thoreau, privative alpha, George I of Greece, communards, Gustav Landauer, Lucifer the Lightbearer, Moses Harman, coercion, regicide, rationalist, Resistance during World War II, Christ (title), Bohemianism, individualism, Crass, black bloc, Spanish Revolution of 1936, Erich Mühsam, Empress Elisabeth of Austria, Free association (communism and anarchism), general strike, Francesc Ferrer i Guàrdia, Catalan anarchist pedagogue and free-thinker, veganarchism, Traditional knowledge, Japanese Anarchist Federation, Diogenes of Sinope, Hierarchy, sexual revolution, Naturism, Bavarian Soviet Republic, February Revolution, Eugene Varlin, Renaissance humanism, Mexican Liberal Party, Friedrich Engels, Fernando Tarrida del Mármol, Caliphate, Marxism, Jesus, John Cage, Umanita Nova, Anarcho-pacifism, Peter Kropotkin, Religious anarchism, Anselme Bellegarrigue, civilisation, moral obligation, hedonist, Free Territory (Ukraine), -ism, neo-liberalism, Austrian School, philosophy, freethought, Joseph Goebbels, Conservatism, anarchist economics, Cavalier, Maximilien de Robespierre, Comstockery, Dorothy Day, Anarchism in France, Fédération anarchiste, World Economic Forum, Amparo Poch y Gascón, Sex Pistols, women's rights, collectivisation, Taoism, common ownership, William Batchelder Greene, Collective farming, popular education, biphobia, targeted killings, Protestant Christianity, state socialism, Marie François Sadi Carnot, Stephen Pearl Andrews, World Trade Organization, Communist Party of Spain (main), Pluto Press, Levante, Spain, Alexander Berkman, Wilhelm Weitling, Kharijites, Bolshevik, Liberty (1881–1908), Anarchist Aragon, social democrats, Dielo Truda, Post-left anarchy, Age of Enlightenment, Blanquism, Walden, mutual aid (organization), Far-left politics, privative, revolutions of 1848, anarchism and nationalism, punk rock, Étienne de La Boétie, Max Stirner, Jacobin (politics), agriculture, anarchy, Confederacion General del Trabajo de España, toleration, reformism, International Anarchist Congress of Amsterdam, The Ego and Its Own, Ukraine, Civil Disobedience (Thoreau), Spanish Civil War, David Graeber, Anarchism and issues related to love and sex, James Guillaume, Insurrectionary anarchism, Political repression, International Workers' Association, Barcelona, Bulgaria, Voline, Zeno of Citium, anarcho-communists, organized religion, libertarianism, bisexuality, Ricardo Flores Magón, Henri Zisly, Eight-hour day, Freetown Christiania, heteronormativity, Mikhail Bakunin, Propagandaministerium, Ezra Heywood, individual reappropriation, Modern School (United States), archon, Confédération nationale du travail, socialist movement, History of Islam, Max Nettlau, Political Justice, Reichstag fire, Anti-Christianity, decentralised, Issues in anarchism#Communism, deschooling, Christian movement, squatter, Anarchism in Germany, Catalonia, Louise Michel, Solidarity Federation, What is Property?, European individualist anarchism, Pierre-Joseph Proudhon, Mexican Revolution, wikt:anarchism, Blackshirts, Jewish anarchism, Russian Civil War, property rights, anti-authoritarian, individual reclamation, propaganda by the deed, from each according to his ability, to each according to his need, Feminist movement, Confiscation, social anarchism, Anarchism in Russia, Daniel Guérin, Uruguayan Anarchist Federation, Anarcha-feminism, Enragés, Cynicism (philosophy), workers' council, The Word (free love), Allen Ginsberg, Campaign for Nuclear Disarmament, antimilitarism, Workers' self-management, Federación Obrera Regional Argentina, self-governance, free market, Carlos I of Portugal, Simon Critchley, Anti-clericalism, heterosexual, Layla AbdelRahim, Mexican Anarchist Federation, Anarchism and Marxism, October Revolution, Anti-nuclear movement, Joseph Déjacque, Bolsheviks, Luigi Fabbri, morality, Communist party, Sam Dolgoff, united front, Ammon Hennacy, social ecology, commune (intentional community), Oscar Wilde, French Revolution, egoist anarchism, Comintern, transphobia, anarchism without adjectives, social control, means of production, Michel Onfray, Anarchism in France#The Fourth Republic (1945–1958), syndicalism, Anarchism in Spain, Iberian Anarchist Federation, International of Anarchist Federations, Emma Goldman, Netherlands, anarchist free school, International Workingmen's Association, Queer anarchism, Cantonal Revolution, trade unionism, Karl Marx, LGBT community, humanism, Anti-fascism, Carrara, political philosophy, Anarcho-transhumanism, libertarian socialist, Russian Revolution (1917), Two Cheers for Anarchism: Six Easy Pieces on Autonomy, Dignity, and Meaningful Work and Play, Emile Armand, insurrectionary anarchism, individual, Zhuang Zhou, Free Territory, White movement, Greenwich Village, Virginia Bolten, transcendentalist, public choice theory, wikt:brigand, Issues in anarchism#Participation in statist democracy, free love, Mutualism (economic theory), Anarchist St. Imier International, censorship, federalist, 6 February 1934 crisis, biennio rosso, anti-clerical, centralism, Anarchism: A Documentary History of Libertarian Ideas, minarchism, James C. Scott, First International, homosexuality, political theology, spontaneous order, Oranienburg concentration camp, anarcho-communism, negative liberty, post-modernism, Anarchism in Italy, Leopold Kohr, union of egoists, counterculture, Miguel Gimenez Igualada, philosophical anarchism, International Libertarian Solidarity, homosexual, Counterculture of the 1960s, Errico Malatesta, strikebreaker, Workers' Party of Marxist Unification, Clifford Harper, Reification (fallacy), patriarchy, anarchist law, Apostle (Christian), market (economics), Summerhill School, positive liberty, socialism, feminism, Direct action, Melchor Rodríguez García, William Godwin, Nazi concentration camps, Synthesist anarchism, Margaret Anderson, Han Ryner, Federation of Organized Trades and Labor Unions, technology, Workers Solidarity Movement, Edmund Burke, Encyclopædia Britannica, state (polity), Herbert Read, Park Güell, utilitarian, far right leagues, Limited government, self-ownership, Pejorative, homophobia, Industrial Workers of the World, The Dispossessed, Hague Congress (1872), Stalinism, Reciprocity (cultural anthropology), Fernand Pelloutier, individualist anarchism in France, The False Principle of our Education, individualist anarchism, Pierre Monatte, Soviet Union, counter-economics, Rudolf Rocker, Anarchism and capitalism, Parma, Black Rose Books, lesbian, Arditi del Popolo, Emile Armand (1872–1962), who propounded the virtues of free love in the Parisian anarchist milieu of the early 20th century, collectivism, Development criticism, John Henry Mackay, Benoît Broutchoux, Illegalism, Laozi, feminist, Christiaan Cornelissen, Syndicalist Workers' Federation, anarcho-syndicalism, Andalusia, Renzo Novatore, trade union, autonomist marxism, dictatorship of the proletariat, Mujeres Libres, Voltairine de Cleyre, Post-anarchism, participatory economics, Confederación Nacional del Trabajo, Syncretic politics, direct democracy, Jean-Jacques Rousseau, Green anarchism, Surrealism, labour unions, A. S. Neill, christian anarchist, Bonnot Gang, Anti-capitalism, Anarchism in Brazil, simple living, enlightened self-interest, Confédération générale du travail, class conflict, International Workers' Day, Hébertists, Gerrard Winstanley, Francoism, anarcho-pacifist, Andrej Grubacic, individualist anarchist and social anarchist thinkers., April Carter, private property, penal colonies, Libertarian socialism, Camillo Berneri, Christian anarchism, transhumanism, Lucifer, the Light-Bearer, Edna St. Vincent Millay, unschooling, Leo Tolstoy, M. E. Lazarus, Spanish Anarchists, Buddhist anarchism, ideology, William McKinley, anarcho-primitivism, Francesc Pi i Margall, :Category:Anarchism by country, International Workers Association, Anarcho-capitalism, Lois Waisbrooker, wikt:Solidarity, Baja California, social revolution, Unione Sindacale Italiana, Lev Chernyi, Alex Comfort, Sonnenburg, Leon Czolgosz, Volin, utopian, Argentine Libertarian Federation, Nudism, Left-wing market anarchism, insurrection, definitional concerns in anarchist theory, infinitive, affinity group, World Trade Organization Ministerial Conference of 1999 protest activity, class struggle, nonviolence, John Zerzan, poststructuralist, Noam Chomsky, Second Fitna, Julian Beck, Philadelphes, League of Peace and Freedom, Fédération Anarchiste, Kronstadt rebellion, Cold War, André Breton, Silvio Gesell, libertarian anarchism, voluntary association, anti-globalisation movement, birth control, L. Susan Brown, anarcho-naturism, personal property, Roundhead, Harold Barclay, The Joy of Sex, Council communism, Lucía Sánchez Saornil, tyrannicide, Neopaganism, lois scélérates, Johann Most, Anarchist Catalonia, Albert Camus, Protests of 1968, Alexander II of Russia, Spain's economy, Federazione Anarchica Italiana, Cuba, German Revolution of 1918–1919, stirner, Property is theft, Situationist International, law and economics

        ```

* Add support for [SMART notation](https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html) for `TfidfModel`. Fix #1785 (__[@markroxor](https://github.com/markroxor)__, [#1791](https://github.com/RaRe-Technologies/gensim/pull/1791))
    - Natural extension of `TfidfModel` to allow different weighting and normalization schemes
        ```python
        from gensim.corpora import Dictionary
        from gensim.models import TfidfModel
        import gensim.downloader as api

        data = api.load("text8")
        dct = Dictionary(data)
        corpus = [dct.doc2bow(line) for line in data]

        # Train Tfidf model using the SMART notation, smartirs="ntc" where
        # 'n' - natural term frequency
        # 't' - idf document frequency
        # 'c' - cosine normalization
        #
        # More information about possible values available in documentation or https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html

        model = TfidfModel(corpus, id2word=dct, smartirs="ntc")
        vectorized_corpus = list(model[corpus])

        ```
    - [SMART Information Retrieval System (wiki)](https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System)

* Add CircleCI for building Gensim documentation. Fix #1807 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1822](https://github.com/RaRe-Technologies/gensim/pull/1822))
    - An easy way to preview the rendered documentation (especially, if don't use Linux)
        - Go to "Details" link of CircleCI in your PR, click on the "Artifacts" tab, choose the HTML file that you want to view; a new tab will open with the rendered HTML page
    - Integration with Github, to see the documentation directly from the pull request page
        - Install a user-script plugin: [greasemonkey (for firefox)](https://addons.mozilla.org/en-US/firefox/addon/greasemonkey/) or [tampermonkey (for chrome)](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo?hl=en)
        - Add [this user-script](https://gist.github.com/menshikh-iv/bfe9b8ef2db10e9511aa9fe5935a7289) to the plugin
        - Now you’ll see a new button "See CircleCI doc for this PR" in each PR in the Gensim repository. Click it to see the full rendered documentation.


:red_circle: Bug fixes:
* Fix import in `get_my_ip`. Fix #1771 (__[@darindf](https://github.com/darindf)__, [#1772](https://github.com/RaRe-Technologies/gensim/pull/1772))
* Fix tox.ini/setup.cfg configuration (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1815](https://github.com/RaRe-Technologies/gensim/pull/1815))
* Fix formula in `gensim.summarization.bm25`. Fix #1828 (__[@sj29-innovate](https://github.com/sj29-innovate)__, [#1833](https://github.com/RaRe-Technologies/gensim/pull/1833))
* Fix the train method of `TranslationMatrix` (__[@robotcator](https://github.com/robotcator)__, [#1838](https://github.com/RaRe-Technologies/gensim/pull/1838))
* Fix positional params used for `gensim.models.CoherenceModel` in `gensim.models.callbacks` (__[@Alexjmsherman](https://github.com/Alexjmsherman)__, [#1823](https://github.com/RaRe-Technologies/gensim/pull/1823))
* Fix parameter setting for `FastText.train`. Fix #1818 (__[@sj29-innovate](https://github.com/sj29-innovate)__, [#1837](https://github.com/RaRe-Technologies/gensim/pull/1837))
* Pin python2 explicitly for building documentation (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1840](https://github.com/RaRe-Technologies/gensim/pull/1840))
* Remove dispatcher deadlock for distributed LDA (__[@darindf](https://github.com/darindf)__, [#1817](https://github.com/RaRe-Technologies/gensim/pull/1817))
* Fix `score_function` from `LexicalEntailmentEvaluation`. Fix #1858 (__[@hachibaka](https://github.com/hachibaka)__, [#1863](https://github.com/RaRe-Technologies/gensim/pull/1863))
* Fix symmetrical case for hellinger distance. Fix #1854 (__[@caiyulun](https://github.com/caiyulun)__, [#1860](https://github.com/RaRe-Technologies/gensim/pull/1860))
* Remove wrong logging at import. Fix #1706 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1871](https://github.com/RaRe-Technologies/gensim/pull/1871))


:books: Tutorial and doc improvements:
* Refactor documentation API Reference for `gensim.summarization` (__[@yurkai](https://github.com/yurkai)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1709](https://github.com/RaRe-Technologies/gensim/pull/1709))
* Fix docstrings for `gensim.similarities.index`. Partial fix #1666 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1681](https://github.com/RaRe-Technologies/gensim/pull/1681))
* Fix docstrings for `gensim.models.translation_matrix` (__[@KokuKUSIAKU](https://github.com/KokuKUSIAKU)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1806](https://github.com/RaRe-Technologies/gensim/pull/1806))
* Fix docstrings for `gensim.models.rpmodel` (__[@jazzmuesli](https://github.com/jazzmuesli)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1802](https://github.com/RaRe-Technologies/gensim/pull/1802))
* Fix docstrings for `gensim.utils` (__[@kakshay21](https://github.com/kakshay21)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1797](https://github.com/RaRe-Technologies/gensim/pull/1797))
* Fix docstrings for `gensim.matutils` (__[@Cheukting](https://github.com/Cheukting)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1804](https://github.com/RaRe-Technologies/gensim/pull/1804))
* Fix docstrings for `gensim.models.logentropy_model` (__[@minggli](https://github.com/minggli)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1803](https://github.com/RaRe-Technologies/gensim/pull/1803))
* Fix docstrings for `gensim.models.normmodel` (__[@AustenLamacraft](https://github.com/AustenLamacraft)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1805](https://github.com/RaRe-Technologies/gensim/pull/1805))
* Refactor API reference `gensim.topic_coherence`. Fix #1669 (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1714](https://github.com/RaRe-Technologies/gensim/pull/1714))
* Fix documentation for `gensim.corpora.dictionary` and `gensim.corpora.hashdictionary`. Partial fix #1671 (__[@CLearERR](https://github.com/CLearERR)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1814](https://github.com/RaRe-Technologies/gensim/pull/1814))
* Fix documentation for `gensim.corpora`. Partial fix #1671 (__[@anotherbugmaster](https://github.com/anotherbugmaster)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1729](https://github.com/RaRe-Technologies/gensim/pull/1729))
* Update banner in doc pages (__[@piskvorky](https://github.com/piskvorky)__, [#1865](https://github.com/RaRe-Technologies/gensim/pull/1865))
* Fix errors in the doc2vec-lee notebook (__[@PeterHamilton](https://github.com/PeterHamilton)__, [#1841](https://github.com/RaRe-Technologies/gensim/pull/1841))
* Add wordnet mammal train file for Poincare notebook (__[@jayantj](https://github.com/jayantj)__, [#1781](https://github.com/RaRe-Technologies/gensim/pull/1781))
* Update Poincare notebooks (#1774) (__[@jayantj](https://github.com/jayantj)__, [#1774](https://github.com/RaRe-Technologies/gensim/pull/1774))
* Update contributing guide. Fix #1786 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1793](https://github.com/RaRe-Technologies/gensim/pull/1793))
* Add `model_to_dict` one-liner to word2vec notebook. Fix #1269 (__[@kakshay21](https://github.com/kakshay21)__, [#1776](https://github.com/RaRe-Technologies/gensim/pull/1776))
* Add word embedding viz to word2vec notebook. Fix #1419 (__[@markroxor](https://github.com/markroxor)__, [#1800](https://github.com/RaRe-Technologies/gensim/pull/1800))
* Fix description of `sg` parameter for `gensim.models.FastText` (__[@akutuzov](https://github.com/akutuzov)__, [#1801](https://github.com/RaRe-Technologies/gensim/pull/1801))
* Fix typo in `doc2vec-IMDB`. Fix #1788 (__[@apoorvaeternity](https://github.com/apoorvaeternity)__, [#1796](https://github.com/RaRe-Technologies/gensim/pull/1796))
* Remove outdated bz2 examples from tutorials[2] (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1868](https://github.com/RaRe-Technologies/gensim/pull/1868))
* Remove outdated `bz2` + `MmCorpus` examples from tutorials (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1867](https://github.com/RaRe-Technologies/gensim/pull/1867))



:+1: Improvements:
* Refactor tests for `gensim.corpora.WikiCorpus` (__[@steremma](https://github.com/steremma)__, [#1821](https://github.com/RaRe-Technologies/gensim/pull/1821))


:warning: Deprecations (will be removed in the next major release)
* Remove
    - `gensim.models.wrappers.fasttext` (obsoleted by the new native `gensim.models.fasttext` implementation)
    - `gensim.examples`
    - `gensim.nosy`
    - `gensim.scripts.word2vec_standalone`
    - `gensim.scripts.make_wiki_lemma`
    - `gensim.scripts.make_wiki_online`
    - `gensim.scripts.make_wiki_online_lemma`
    - `gensim.scripts.make_wiki_online_nodebug`
    - `gensim.scripts.make_wiki` (all of these obsoleted by the new native  `gensim.scripts.segment_wiki` implementation)
    - "deprecated" functions and attributes

* Move
    - `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
    - `gensim.summarization` ➡ `gensim.models.summarization`
    - `gensim.topic_coherence` ➡ `gensim.models._coherence`
    - `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
    - `gensim.parsing.*` ➡ `gensim.utils.text_utils`


## 3.2.0, 2017-12-09

:star2: New features:

* New download API for corpora and pre-trained models (__[@chaitaliSaini](https://github.com/chaitaliSaini)__ & __[@menshikh-iv](https://github.com/menshikh-iv)__, [#1705](https://github.com/RaRe-Technologies/gensim/pull/1705) & [#1632](https://github.com/RaRe-Technologies/gensim/pull/1632) & [#1492](https://github.com/RaRe-Technologies/gensim/pull/1492))
    - Download large NLP datasets in one line of Python, then use with memory-efficient data streaming:
        ```python
        import gensim.downloader as api

        for article in api.load("wiki-english-20171001"):
            pass

        ```
    - Don’t waste time searching for good word embeddings, use the curated ones we included:
        ```python
        import gensim.downloader as api

        model = api.load("glove-twitter-25")
        model.most_similar("engineer")

        # [('specialist', 0.957542896270752),
        #  ('developer', 0.9548177123069763),
        #  ('administrator', 0.9432312846183777),
        #  ('consultant', 0.93915855884552),
        #  ('technician', 0.9368376135826111),
        #  ('analyst', 0.9342101216316223),
        #  ('architect', 0.9257484674453735),
        #  ('engineering', 0.9159940481185913),
        #  ('systems', 0.9123805165290833),
        #  ('consulting', 0.9112802147865295)]
        ```
    - [Blog post](https://rare-technologies.com/new-api-for-pretrained-nlp-models-and-datasets-in-gensim/) introducing the API and design decisions.
  - [Notebook with examples](https://github.com/RaRe-Technologies/gensim/blob/be4500e4f0616ec2864c2ce70cb5d4db4b46512d/docs/notebooks/downloader_api_tutorial.ipynb)

* New model: Poincaré embeddings (__[@jayantj](https://github.com/jayantj)__, [#1696](https://github.com/RaRe-Technologies/gensim/pull/1696) & [#1700](https://github.com/RaRe-Technologies/gensim/pull/1700) & [#1757](https://github.com/RaRe-Technologies/gensim/pull/1757) & [#1734](https://github.com/RaRe-Technologies/gensim/pull/1734))
    - Embed a graph (taxonomy) in the same way as word2vec embeds words:
        ```python
        from gensim.models.poincare import PoincareRelations, PoincareModel
        from gensim.test.utils import datapath

        data = PoincareRelations(datapath('poincare_hypernyms.tsv'))
        model = PoincareModel(data)
        model.kv.most_similar("cat.n.01")

        # [('kangaroo.n.01', 0.010581353439700418),
        # ('gib.n.02', 0.011171531439892076),
        # ('striped_skunk.n.01', 0.012025106076442395),
        # ('metatherian.n.01', 0.01246679759214648),
        # ('mammal.n.01', 0.013281303506525968),
        # ('marsupial.n.01', 0.013941330203709653)]
        ```
    - [Tutorial notebook on Poincaré embeddings](https://github.com/RaRe-Technologies/gensim/blob/920c029ca97f961c8df264672c34936607876694/docs/notebooks/Poincare%20Tutorial.ipynb)
    - [Model introduction and the journey of its implementation](https://rare-technologies.com/implementing-poincare-embeddings/)
    - [Original paper](https://arxiv.org/abs/1705.08039) on arXiv

* Optimized FastText (__[@manneshiva](https://github.com/manneshiva)__, [#1742](https://github.com/RaRe-Technologies/gensim/pull/1742))
  - New fast multithreaded implementation of FastText, natively in Python/Cython. Deprecates the existing wrapper for Facebook’s C++ implementation.
    ```python
    import gensim.downloader as api
    from gensim.models import FastText

    model = FastText(api.load("text8"))
    model.most_similar("cat")

    # [('catnip', 0.8538144826889038),
    #  ('catwalk', 0.8136177062988281),
    #  ('catchy', 0.7828493118286133),
    #  ('caf', 0.7826495170593262),
    #  ('bobcat', 0.7745151519775391),
    #  ('tomcat', 0.7732658386230469),
    #  ('moat', 0.7728310823440552),
    #  ('caye', 0.7666271328926086),
    #  ('catv', 0.7651021480560303),
    #  ('caveat', 0.7643581628799438)]


    ```

* Binary pre-compiled wheels for Windows, OSX and Linux (__[@menshikh-iv](https://github.com/menshikh-iv)__, [MacPython/gensim-wheels/#7](https://github.com/MacPython/gensim-wheels/pull/7))
    - Users no longer need to have a C compiler for using the fast (Cythonized) version of word2vec, doc2vec, etc.
    - Faster Gensim pip installation

* Added `DeprecationWarnings` to deprecated methods and parameters, with a clear schedule for removal.

:+1: Improvements:
* Add Montemurro and Zanette's entropy based keyword extraction algorithm. Fix #665 (__[@PeteBleackley](https://github.com/PeteBleackley)__, [#1738](https://github.com/RaRe-Technologies/gensim/pull/1738))
* Fix flake8 E731, E402, refactor tests & sklearn API code. Partial fix #1644  (__[@horpto](https://github.com/horpto)__, [#1689](https://github.com/RaRe-Technologies/gensim/pull/1689))
* Reduce distribution size. Fix #1698 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1699](https://github.com/RaRe-Technologies/gensim/pull/1699))
* Improve `scan_vocab` speed, `build_vocab_from_freq` method (__[@jodevak](https://github.com/jodevak)__, [#1695](https://github.com/RaRe-Technologies/gensim/pull/1695))
* Improve `segment_wiki` script (__[@piskvorky](https://github.com/piskvorky)__, [#1707](https://github.com/RaRe-Technologies/gensim/pull/1707))
* Add custom `dtype` support for `LdaModel`. Partially fix #1576 (__[@xelez](https://github.com/xelez)__, [#1656](https://github.com/RaRe-Technologies/gensim/pull/1656))
* Add `doc2idx` method for `gensim.corpora.Dictionary`. Fix #1634 (__[@roopalgarg](https://github.com/roopalgarg)__, [#1720](https://github.com/RaRe-Technologies/gensim/pull/1720))
* Add tox and pytest to gensim, integration with Travis and Appveyor. Fix #1613, #1644 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1721](https://github.com/RaRe-Technologies/gensim/pull/1721))
* Add flag for hiding outdated data for `gensim.downloader.info` (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1736](https://github.com/RaRe-Technologies/gensim/pull/1736))
* Add reproducible order between python versions for `gensim.corpora.Dictionary` (__[@formi23](https://github.com/formi23)__, [#1715](https://github.com/RaRe-Technologies/gensim/pull/1715))
* Update `tox.ini`, `setup.cfg`, `README.md` (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1741](https://github.com/RaRe-Technologies/gensim/pull/1741))
* Add custom `logsumexp` for `LdaModel` (__[@arlenk](https://github.com/arlenk)__, [#1745](https://github.com/RaRe-Technologies/gensim/pull/1745))

:red_circle: Bug fixes:
* Fix ranking formula in `gensim.summarization.bm25`. Fix #1718 (__[@souravsingh](https://github.com/souravsingh)__, [#1726](https://github.com/RaRe-Technologies/gensim/pull/1726))
* Fixed incompatibility in persistence for `FastText` wrapper. Fix #1642 (__[@chinmayapancholi13](https://github.com/chinmayapancholi13)__, [#1723](https://github.com/RaRe-Technologies/gensim/pull/1723))
* Fix `gensim.sklearn_api` bug with `documents_columns` parameter. Fix #1676 (__[@chinmayapancholi13](https://github.com/chinmayapancholi13)__, [#1704](https://github.com/RaRe-Technologies/gensim/pull/1704))
* Fix slowdown of CI, remove pytest-cov (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1728](https://github.com/RaRe-Technologies/gensim/pull/1728))
* Replace outdated packages in Dockerfile (__[@rbahumi](https://github.com/rbahumi)__, [#1730](https://github.com/RaRe-Technologies/gensim/pull/1730))
* Replace `num_words` to `topn` in `LdaMallet.show_topics`. Fix #1747 (__[@apoorvaeternity](https://github.com/apoorvaeternity)__, [#1749](https://github.com/RaRe-Technologies/gensim/pull/1749))
* Fix `os.rename` from `gensim.downloader` when 'src' and 'dst' on different partitions (__[@anotherbugmaster](https://github.com/anotherbugmaster)__, [#1733](https://github.com/RaRe-Technologies/gensim/pull/1733))
* Fix `DeprecationWarning` from `logsumexp` (__[@dreamgonfly](https://github.com/dreamgonfly)__, [#1703](https://github.com/RaRe-Technologies/gensim/pull/1703))
* Fix backward compatibility problem in `Phrases.load`. Fix #1751 (__[@alexgarel](https://github.com/alexgarel)__, [#1758](https://github.com/RaRe-Technologies/gensim/pull/1758))
* Fix `load_word2vec_format` from `FastText`. Fix #1743 (__[@manneshiva](https://github.com/manneshiva)__, [#1755](https://github.com/RaRe-Technologies/gensim/pull/1755))
* Fix ipython kernel version in `Dockerfile`. Fix #1762 (__[@rbahumi](https://github.com/rbahumi)__, [#1764](https://github.com/RaRe-Technologies/gensim/pull/1764))
* Fix writing in `segment_wiki` (__[@horpto](https://github.com/horpto)__, [#1763](https://github.com/RaRe-Technologies/gensim/pull/1763))
* Fix write method of file requires byte-like object in `segment_wiki` (__[@horpto](https://github.com/horpto)__, [#1750](https://github.com/RaRe-Technologies/gensim/pull/1750))
* Fix incorrect vectors learned during online training for `FastText`. Fix #1752 (__[@manneshiva](https://github.com/manneshiva)__, [#1756](https://github.com/RaRe-Technologies/gensim/pull/1756))
* Fix `dtype` of `model.wv.syn0_vocab` on updating `vocab` for `FastText`. Fix  #1759 (__[@manneshiva](https://github.com/manneshiva)__, [#1760](https://github.com/RaRe-Technologies/gensim/pull/1760))
* Fix hashing-trick from `FastText.build_vocab`. Fix #1765 (__[@manneshiva](https://github.com/manneshiva)__, [#1768](https://github.com/RaRe-Technologies/gensim/pull/1768))
* Add explicit `DeprecationWarning` for all outdated stuff. Fix #1753 (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1769](https://github.com/RaRe-Technologies/gensim/pull/1769))
* Fix epsilon according to `dtype` in `LdaModel` (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1770](https://github.com/RaRe-Technologies/gensim/pull/1770))

:books: Tutorial and doc improvements:
* Update perf numbers of `segment_wiki` (__[@piskvorky](https://github.com/piskvorky)__, [#1708](https://github.com/RaRe-Technologies/gensim/pull/1708))
* Update docstring for `gensim.summarization.summarize`. Fix #1575 (__[@fbarrios](https://github.com/fbarrios)__, [#1702](https://github.com/RaRe-Technologies/gensim/pull/1702))
* Refactor API Reference for `gensim.parsing`. Fix #1664 (__[@CLearERR](https://github.com/CLearERR)__, [#1684](https://github.com/RaRe-Technologies/gensim/pull/1684))
* Fix typos in doc2vec-wikipedia notebook (__[@youqad](https://github.com/youqad)__, [#1727](https://github.com/RaRe-Technologies/gensim/pull/1727))
* Fix PyPI long description rendering (__[@edigaryev](https://github.com/edigaryev)__, [#1739](https://github.com/RaRe-Technologies/gensim/pull/1739))
* Fix twitter badge src  (__[@menshikh-iv](https://github.com/menshikh-iv)__)
* Fix maillist badge color (__[@menshikh-iv](https://github.com/menshikh-iv)__)

:warning: Deprecations (will be removed in the next major release)
* Remove
    - `gensim.examples`
    - `gensim.nosy`
    - `gensim.scripts.word2vec_standalone`
    - `gensim.scripts.make_wiki_lemma`
    - `gensim.scripts.make_wiki_online`
    - `gensim.scripts.make_wiki_online_lemma`
    - `gensim.scripts.make_wiki_online_nodebug`
    - `gensim.scripts.make_wiki`

* Move
    - `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
    - `gensim.summarization` ➡ `gensim.models.summarization`
    - `gensim.topic_coherence` ➡ `gensim.models._coherence`
    - `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
    - `gensim.parsing.*` ➡ `gensim.utils.text_utils`


## 3.1.0, 2017-11-06


:star2: New features:
* Massive optimizations to LSI model training (__[@isamaru](https://github.com/isamaru)__, [#1620](https://github.com/RaRe-Technologies/gensim/pull/1620) & [#1622](https://github.com/RaRe-Technologies/gensim/pull/1622))
  - LSI model allows use of single precision (float32), to consume  *40% less memory* while being *40% faster*.
  - LSI model can now also accept CSC matrix as input, for further memory and speed boost.
  - Overall, if your entire corpus fits in RAM: 3x faster LSI training (SVD) in 4x less memory!
    ```python
    # just an example; the corpus stream is up to you
    streaming_corpus = gensim.corpora.MmCorpus("my_tfidf_corpus.mm.gz")

    # convert your corpus to a CSC sparse matrix (assumes the entire corpus fits in RAM)
    in_memory_csc_matrix = gensim.matutils.corpus2csc(streaming_corpus, dtype=np.float32)

    # then pass the CSC to LsiModel directly
    model = LsiModel(corpus=in_memory_csc_matrix, num_topics=500, dtype=np.float32)
    ```
  - Even if you continue to use streaming corpora (your training dataset is too large for RAM), you should see significantly faster processing times and a lower memory footprint. In our experiments with a very large LSI model, we saw a drop from 29 GB peak RAM and 38 minutes (before) to 19 GB peak RAM and 26 minutes (now):
    ```python
    model = LsiModel(corpus=streaming_corpus, num_topics=500, dtype=np.float32)
    ```
* Add common terms to Phrases. Fix #1258 (__[@alexgarel](https://github.com/alexgarel)__, [#1568](https://github.com/RaRe-Technologies/gensim/pull/1568))
  - Phrases allows to use common terms in bigrams. Before, if you are searching to reveal ngrams like `car_with_driver` and `car_without_driver`, you can either remove stop words before processing, but you will only find `car_driver`, or you won't find any of those forms (because they have three words, but also because high frequency of with will avoid them to be scored correctly), inspired by [ES common grams token filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-common-grams-tokenfilter.html).
    ```python
    phr_old = Phrases(corpus)
    phr_new = Phrases(corpus, common_terms=stopwords.words('en'))

    print(phr_old[["we", "provide", "car", "with", "driver"]])  # ["we", "provide", "car_with", "driver"]
    print(phr_new[["we", "provide", "car", "with", "driver"]])  # ["we", "provide", "car_with_driver"]
    ```
* New [segment_wiki.py](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/scripts/segment_wiki.py) script (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1483](https://github.com/RaRe-Technologies/gensim/pull/1483) & [#1694](https://github.com/RaRe-Technologies/gensim/pull/1694))
  - CLI script for processing a raw Wikipedia dump (the xml.bz2 format provided by WikiMedia) to extract its articles in a plain text format. It extracts each article's title, section names and section content and saves them as json-line:
    ```bash
    python -m gensim.scripts.segment_wiki -f enwiki-latest-pages-articles.xml.bz2 | gzip > enwiki-latest-pages-articles.json.gz
    ```
       Processing the entire English Wikipedia dump (13.5 GB, link [here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)) takes about 2.5 hours (i7-6700HQ, SSD).

       The output format is one article per line, serialized into JSON:
       ```python
          for line in smart_open('enwiki-latest-pages-articles.json.gz'):  # read the file we just created
              article = json.loads(line)
              print("Article title: %s" % article['title'])
              for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                  print("Section title: %s" % section_title)
                  print("Section text: %s" % section_text)
        ```

:+1: Improvements:
* Speedup FastText tests (__[@horpto](https://github.com/horpto)__, [#1686](https://github.com/RaRe-Technologies/gensim/pull/1686))
* Add optimization for `SlicedCorpus.__len__` (__[@horpto](https://github.com/horpto)__, [#1679](https://github.com/RaRe-Technologies/gensim/pull/1679))
* Make `word_vec` return immutable vector. Fix #1651 (__[@CLearERR](https://github.com/CLearERR)__, [#1662](https://github.com/RaRe-Technologies/gensim/pull/1662))
* Drop Win x32 support & add rolling builds (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1652](https://github.com/RaRe-Technologies/gensim/pull/1652))
* Fix scoring function in Phrases. Fix #1533, #1635 (__[@michaelwsherman](https://github.com/michaelwsherman)__, [#1573](https://github.com/RaRe-Technologies/gensim/pull/1573))
* Add configuration for flake8 to setup.cfg (__[@mcobzarenco](https://github.com/mcobzarenco)__, [#1636](https://github.com/RaRe-Technologies/gensim/pull/1636))
* Add `build_vocab_from_freq` to Word2Vec, speedup scan\_vocab (__[@jodevak](https://github.com/jodevak)__, [#1599](https://github.com/RaRe-Technologies/gensim/pull/1599))
* Add `most_similar_to_given` method for KeyedVectors (__[@TheMathMajor](https://github.com/TheMathMajor)__, [#1582](https://github.com/RaRe-Technologies/gensim/pull/1582))
* Add `__getitem__` method to Sparse2Corpus to allow direct queries (__[@isamaru](https://github.com/isamaru)__, [#1621](https://github.com/RaRe-Technologies/gensim/pull/1621))

:red_circle: Bug fixes:
* Add single core mode to CoherenceModel. Fix #1683 (__[@horpto](https://github.com/horpto)__, [#1685](https://github.com/RaRe-Technologies/gensim/pull/1685))
* Fix ResourceWarnings in tests. Partially fix #1519 (__[@horpto](https://github.com/horpto)__, [#1660](https://github.com/RaRe-Technologies/gensim/pull/1660))
* Fix DeprecationWarnings generated by deprecated assertEquals. Partial fix #1519 (__[@poornagurram](https://github.com/poornagurram)__, [#1658](https://github.com/RaRe-Technologies/gensim/pull/1658))
* Fix DeprecationWarnings for regex string literals. Fix #1646 (__[@franklsf95](https://github.com/franklsf95)__, [#1649](https://github.com/RaRe-Technologies/gensim/pull/1649))
* Fix pagerank algorithm. Fix #805 (__[@xelez](https://github.com/xelez)__, [#1653](https://github.com/RaRe-Technologies/gensim/pull/1653))
* Fix FastText inconsistent dtype. Fix #1637 (__[@mcobzarenco](https://github.com/mcobzarenco)__, [#1638](https://github.com/RaRe-Technologies/gensim/pull/1638))
* Fix `test_filename_filtering` test (__[@nehaljwani](https://github.com/nehaljwani)__, [#1647](https://github.com/RaRe-Technologies/gensim/pull/1647))

:books: Tutorial and doc improvements:
* Fix code/docstring style (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1650](https://github.com/RaRe-Technologies/gensim/pull/1650))
* Update error message for supervised FastText. Fix #1498 (__[@ElSaico](https://github.com/ElSaico)__, [#1645](https://github.com/RaRe-Technologies/gensim/pull/1645))
* Add "DOI badge" to README. Fix #1610 (__[@dphov](https://github.com/dphov)__, [#1639](https://github.com/RaRe-Technologies/gensim/pull/1639))
* Remove duplicate annoy notebook. Fix #1415 (__[@Karamax](https://github.com/Karamax)__, [#1640](https://github.com/RaRe-Technologies/gensim/pull/1640))
* Fix duplication and wrong markup in docs (__[@horpto](https://github.com/horpto)__, [#1633](https://github.com/RaRe-Technologies/gensim/pull/1633))
* Refactor dendrogram & topic network notebooks (__[@parulsethi](https://github.com/parulsethi)__, [#1571](https://github.com/RaRe-Technologies/gensim/pull/1571))
* Fix release badge (__[@menshikh-iv](https://github.com/menshikh-iv)__, [#1631](https://github.com/RaRe-Technologies/gensim/pull/1631))

:warning: Deprecation part (will come into force in the next major release)
* Remove
	- `gensim.examples`
	- `gensim.nosy`
	- `gensim.scripts.word2vec_standalone`
	- `gensim.scripts.make_wiki_lemma`
	- `gensim.scripts.make_wiki_online`
	- `gensim.scripts.make_wiki_online_lemma`
	- `gensim.scripts.make_wiki_online_nodebug`
	- `gensim.scripts.make_wiki`

* Move
	- `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
	- `gensim.summarization` ➡ `gensim.models.summarization`
	- `gensim.topic_coherence` ➡ `gensim.models._coherence`
	- `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
	- `gensim.parsing.*` ➡ `gensim.utils.text_utils`

Also, we'll create `experimental` subpackage for unstable models. Specific lists will be available in the next major release.


## 3.0.1, 2017-10-12


:red_circle: Bug fixes:
* Fix Keras import, speedup importing time. Fix #1614 (@menshikh-v, [#1615](https://github.com/RaRe-Technologies/gensim/pull/1615))
* Fix Sphinx warnings and retreive all missing .rst (@anotherbugmaster and @menshikh-iv, [#1612](https://github.com/RaRe-Technologies/gensim/pull/1612))
* Fix logger message in lsi_dispatcher (@lorosanu, [#1603](https://github.com/RaRe-Technologies/gensim/pull/1603))


:books: Tutorial and doc improvements:
* Fix spelling (@jberkel, [#1625](https://github.com/RaRe-Technologies/gensim/pull/1625))

:warning: Deprecation part (will come into force in the next release)
* Remove
	- `gensim.examples`
	- `gensim.nosy`
	- `gensim.scripts.word2vec_standalone`
	- `gensim.scripts.make_wiki_lemma`
	- `gensim.scripts.make_wiki_online`
	- `gensim.scripts.make_wiki_online_lemma`
	- `gensim.scripts.make_wiki_online_nodebug`
	- `gensim.scripts.make_wiki`

* Move
	- `gensim.scripts.make_wikicorpus` ➡ `gensim.scripts.make_wiki.py`
	- `gensim.summarization` ➡ `gensim.models.summarization`
	- `gensim.topic_coherence` ➡ `gensim.models._coherence`
	- `gensim.utils` ➡ `gensim.utils.utils` (old imports will continue to work)
	- `gensim.parsing.*` ➡ `gensim.utils.text_utils`

Also, we'll create `experimental` subpackage for unstable models. Specific lists will be available in the next release.


## 3.0.0, 2017-09-27


:star2: New features:
* Add unsupervised FastText to Gensim (@chinmayapancholi13, [#1525](https://github.com/RaRe-Technologies/gensim/pull/1525))
* Add sklearn API for gensim models (@chinmayapancholi13, [#1462](https://github.com/RaRe-Technologies/gensim/pull/1462))
* Add callback metrics for LdaModel and integration with Visdom (@parulsethi, [#1399](https://github.com/RaRe-Technologies/gensim/pull/1399))
* Add TranslationMatrix model (@robotcator, [#1434](https://github.com/RaRe-Technologies/gensim/pull/1434))
* Add word2vec-based coherence. Fix #1380 (@macks22, [#1530](https://github.com/RaRe-Technologies/gensim/pull/1530))


:+1: Improvements:
* Add 'diagonal' parameter for LdaModel.diff (@parulsethi, [#1448](https://github.com/RaRe-Technologies/gensim/pull/1448))
* Add 'score' function for SklLdaModel (@chinmayapancholi13, [#1445](https://github.com/RaRe-Technologies/gensim/pull/1445))
* Update sklearn API for gensim models (@chinmayapancholi13, [#1473](https://github.com/RaRe-Technologies/gensim/pull/1473)) [:warning: breaks backward compatibility]
* Add CoherenceModel to LdaModel.top_topics. Fix #1128 (@macks22, [#1427](https://github.com/RaRe-Technologies/gensim/pull/1427))
* Add dendrogram viz for topics and JS metric (@parulsethi, [#1484](https://github.com/RaRe-Technologies/gensim/pull/1484))
* Add topic network viz (@parulsethi, [#1536](https://github.com/RaRe-Technologies/gensim/pull/1536))
* Replace viewitems to iteritems. Fix #1495 (@HodorTheCoder, [#1508](https://github.com/RaRe-Technologies/gensim/pull/1508))
* Fix Travis config and add style-checking for Ipython Notebooks. Fix #1518, #1520 (@menshikh-iv, [#1522](https://github.com/RaRe-Technologies/gensim/pull/1522))
* Remove mutable args from definitions. Fix #1561 (@zsef123, [#1562](https://github.com/RaRe-Technologies/gensim/pull/1562))
* Add Appveyour for all PRs. Fix #1565 (@menshikh-iv, [#1565](https://github.com/RaRe-Technologies/gensim/pull/1565))
* Refactor code by PEP8. Partially fix #1521 (@zsef123, [#1550](https://github.com/RaRe-Technologies/gensim/pull/1550))
* Refactor code by PEP8 with additional limitations. Fix #1521 (@menshikh-iv, [#1569](https://github.com/RaRe-Technologies/gensim/pull/1569))
* Update FastTextKeyedVectors.\_\_contains\_\_ (@ELind77, [#1499](https://github.com/RaRe-Technologies/gensim/pull/1499))
* Update WikiCorpus tokenization. Fix #1534 (@roopalgarg, [#1537](https://github.com/RaRe-Technologies/gensim/pull/1537))


:red_circle: Bug fixes:
* Remove round in LdaSeqModel.print_topic. Fix #1480 (@menshikh-iv, [#1547](https://github.com/RaRe-Technologies/gensim/pull/1547))
* Fix TextCorpus.samle_text (@menshikh-iv, [#1548](https://github.com/RaRe-Technologies/gensim/pull/1548))
* Fix Mallet wrapper and tests for HDPTransform (@menshikh-iv, [#1555](https://github.com/RaRe-Technologies/gensim/pull/1555))
* Fix incorrect initialization ShardedCorpus with a generator. Fix #1511 (@karkkainenk1, [#1512](https://github.com/RaRe-Technologies/gensim/pull/1512))
* Add verification when summarize_corpus returns null. Fix #1531 (@fbarrios, [#1570](https://github.com/RaRe-Technologies/gensim/pull/1570))
* Fix doctag unicode problem. Fix 1543 (@englhardt, [#1544](https://github.com/RaRe-Technologies/gensim/pull/1544))
* Fix Translation Matrix (@robotcator, [#1594](https://github.com/RaRe-Technologies/gensim/pull/1594))
* Add trainable flag to KeyedVectors.get_embedding_layer. Fix #1557 (@zsef123, [#1558](https://github.com/RaRe-Technologies/gensim/pull/1558))


:books: Tutorial and doc improvements:
* Update exception text in TextCorpus.samle_text. Partial fix #308 (@vlejd, [#1444](https://github.com/RaRe-Technologies/gensim/pull/1444))
* Remove extra filter_token from tutorial (@VorontsovIE, [#1502](https://github.com/RaRe-Technologies/gensim/pull/1502))
* Update Doc2Vec-IMDB notebook (@pahdo, [#1476](https://github.com/RaRe-Technologies/gensim/pull/1476))
* Add Google Tag Manager for site (@yardos, [#1556](https://github.com/RaRe-Technologies/gensim/pull/1556))
* Update docstring explaining lack of multistream support in WikiCopus. Fix #1496 (@polm and @menshikh-iv, [#1515](https://github.com/RaRe-Technologies/gensim/pull/1515))
* Fix PathLineSentences docstring (@gojomo)
* Fix typos from Translation Matrix notebook (@robotcator, [#1598](https://github.com/RaRe-Technologies/gensim/pull/1598))


## 2.3.0, 2017-07-25


:star2: New features:
* Add Dockerfile for gensim with external wrappers (@parulsethi, [#1368](https://github.com/RaRe-Technologies/gensim/pull/1368))
* Add sklearn wrapper for Word2Vec (@chinmayapancholi13, [#1437](https://github.com/RaRe-Technologies/gensim/pull/1437))
* Add loss function for Word2Vec. Fix #999 (@chinmayapancholi13, [#1201](https://github.com/RaRe-Technologies/gensim/pull/1201))
* Add sklearn wrapper for AuthorTopic model (@chinmayapancholi13, [#1403](https://github.com/RaRe-Technologies/gensim/pull/1403))


:+1: Improvements:
* Remove unittest2 (@souravsingh, [#1490](https://github.com/RaRe-Technologies/gensim/pull/1490))
* Add multiple scoring methods for Phrases. Partial fix #1363 (@michaelwsherman, [#1464](https://github.com/RaRe-Technologies/gensim/pull/1464))
* Add WordRank wrapper to Dockerfile (@parulsethi, [#1460](https://github.com/RaRe-Technologies/gensim/pull/1460))
* Add PathLineSentences. Fix #1364 (@michaelwsherman, [#1423](https://github.com/RaRe-Technologies/gensim/pull/1423))
* Add TextDirectoryCorpus and refactor TextCorpus. Fix #1387 (@macks22, [#1459](https://github.com/RaRe-Technologies/gensim/pull/1459))
* Add sparse input support with topn parameter in any2sparse. Fix #1294 (@manneshiva, [#1321](https://github.com/RaRe-Technologies/gensim/pull/1321))
* Add seed and length for sample_text. Partial fix #308 (@vlejd, [#1422](https://github.com/RaRe-Technologies/gensim/pull/1422))
* Add word_ngram parameter to FastText (@fsonntag, [#1432](https://github.com/RaRe-Technologies/gensim/pull/1432))


:red_circle: Bug fixes:
* Fix fastText loading from .bin file. Fix #1236 (@prakhar2b, [#1341](https://github.com/RaRe-Technologies/gensim/pull/1341))
* Fix paths in WordRank and running gensim version in Dockerfile (@parulsethi, [#1503](https://github.com/RaRe-Technologies/gensim/pull/1503))
* Fix commit version for gensim in Dockerfile (@parulsethi, [#1491](https://github.com/RaRe-Technologies/gensim/pull/1491))
* Fix encoding problems with tests on windows. Fix #1441 (@menshikh-iv, [#1469](https://github.com/RaRe-Technologies/gensim/pull/1469))
* Fix parameters in score_cbow_pair (@jmhessel, [#1468](https://github.com/RaRe-Technologies/gensim/pull/1468))
* Fix parameters in score_sentence_cbow (@jmhessel, [#1467](https://github.com/RaRe-Technologies/gensim/pull/1467))
* Fix TextDirectoryCorpus on windows (@macks22, [#1463](https://github.com/RaRe-Technologies/gensim/pull/1463))
* Fix gensim version in Dockerfile (@parulsethi, [#1456](https://github.com/RaRe-Technologies/gensim/pull/1456))
* Fix WordOccurenceAccumulator on windows. Fix #1441 (@macks22, [#1449](https://github.com/RaRe-Technologies/gensim/pull/1449))
* Fix scipy/numpy requirements (downgrade). Fix #1450 (@menshikh-iv, [#1450](https://github.com/RaRe-Technologies/gensim/pull/1450))


:books: Tutorial and doc improvements:
* Fix links and spaces in quick start guide (@iamsanten, [#1500](https://github.com/RaRe-Technologies/gensim/pull/1500))
* Fix error of ConcatedDoc2Vec in doc2vec-imdb notebook (@robocator, [#1377](https://github.com/RaRe-Technologies/gensim/pull/1377))
* Fix Sphinx warnings. Fix #1192 (@prerna135, [#1442](https://github.com/RaRe-Technologies/gensim/pull/1442))
* Fix typo in LdaModel.diff method (@parulsethi, [#1461](https://github.com/RaRe-Technologies/gensim/pull/1461))
* Add Tensorboard visualization for LDA (@parulsethi, [#1396](https://github.com/RaRe-Technologies/gensim/pull/1396))
* Update old and add new notebook with CoherenceModel (@macks22, [#1431](https://github.com/RaRe-Technologies/gensim/pull/1431))



## 2.2.0, 2017-06-21


:star2: New features:
* Add sklearn wrapper for RpModel (@chinmayapancholi13, [#1395](https://github.com/RaRe-Technologies/gensim/pull/1395))
* Add sklearn wrappers for LdaModel and LsiModel (@chinmayapancholi13, [#1398](https://github.com/RaRe-Technologies/gensim/pull/1398))
* Add sklearn wrapper for LdaSeq (@chinmayapancholi13, [#1405](https://github.com/RaRe-Technologies/gensim/pull/1405))
* Add keras wrapper for Word2Vec model (@chinmayapancholi13, [#1248](https://github.com/RaRe-Technologies/gensim/pull/1248))
* Add LdaModel.diff method (@menshikh-iv, [#1334](https://github.com/RaRe-Technologies/gensim/pull/1334))
* Allow use of truncated Dictionary for coherence measures. Fix #1342 (@macks22, [#1349](https://github.com/RaRe-Technologies/gensim/pull/1349))


:+1: Improvements:
* Fix save_as_text/load_as_text for Dictionary (@vlejd, [#1402](https://github.com/RaRe-Technologies/gensim/pull/1402))
* Add sampling support for corpus. Fix #308 (@vlejd, [#1408](https://github.com/RaRe-Technologies/gensim/pull/1408))
* Add napoleon extension to sphinx (@rasto2211, [#1411](https://github.com/RaRe-Technologies/gensim/pull/1411))
* Add KeyedVectors support to AnnoyIndexer (@quole, [#1318](https://github.com/RaRe-Technologies/gensim/pull/1318))
* Add BaseSklearnWrapper (@chinmayapancholi13, [#1383](https://github.com/RaRe-Technologies/gensim/pull/1383))
* Replace num_words to topn in model for unification. Fix #1198 (@prakhar2b, [#1200](https://github.com/RaRe-Technologies/gensim/pull/1200))
* Rename out_path to out_name & add logging for WordRank model. Fix #1310 (@parulsethi, [#1332](https://github.com/RaRe-Technologies/gensim/pull/1332))
* Remove multiple iterations of corpus in p_boolean_document (@danielchamberlain, [#1325](https://github.com/RaRe-Technologies/gensim/pull/1325))
* Fix codestyle in TfIdf (@piskvorky, [#1313](https://github.com/RaRe-Technologies/gensim/pull/1313))
* Fix warnings from Sphinx. Partial fix #1192 (@souravsingh, [#1330](https://github.com/RaRe-Technologies/gensim/pull/1330))
* Add test_env to setup.py (@menshikh-iv, [#1336](https://github.com/RaRe-Technologies/gensim/pull/1336))


:red_circle: Bug fixes:
* Add cleanup in annoy test (@prakhar2b, [#1420](https://github.com/RaRe-Technologies/gensim/pull/1420))
* Add cleanup in lda backprop test (@prakhar2b, [#1417](https://github.com/RaRe-Technologies/gensim/pull/1417))
* Fix out-of-vocab in FastText (@jayantj, [#1409](https://github.com/RaRe-Technologies/gensim/pull/1409))
* Add cleanup in WordRank test (@parulsethi, [#1410](https://github.com/RaRe-Technologies/gensim/pull/1410))
* Fix rest requirements in Travis. Partial fix #1393 (@ibrahimsharaf, @menshikh-iv, [#1400](https://github.com/RaRe-Technologies/gensim/pull/1400))
* Fix morfessor exception. Partial fix #1324 (@souravsingh, [#1406](https://github.com/RaRe-Technologies/gensim/pull/1406))
* Fix test for FastText (@prakhar2b, [#1371](https://github.com/RaRe-Technologies/gensim/pull/1371))
* Fix WikiCorpus (@alekol, [#1333](https://github.com/RaRe-Technologies/gensim/pull/1333))
* Fix backward incompatibility for LdaModel (@chinmayapancholi13, [#1327](https://github.com/RaRe-Technologies/gensim/pull/1327))
* Fix support for old and new FastText model format. Fix #1301 (@prakhar2b, [#1319](https://github.com/RaRe-Technologies/gensim/pull/1319))
* Fix wrapper tests. Fix #1323 (@shubhamjain74, [#1359](https://github.com/RaRe-Technologies/gensim/pull/1359))
* Update export_phrases method. Fix #794 (@toumorokoshi, [#1362](https://github.com/RaRe-Technologies/gensim/pull/1362))
* Fix sklearn exception in test (@souravsingh, [#1350](https://github.com/RaRe-Technologies/gensim/pull/1350))


:books: Tutorial and doc improvements:
* Fix incorrect link in tutorials (@aneesh-joshi, [#1426](https://github.com/RaRe-Technologies/gensim/pull/1426))
* Add notebook with sklearn wrapper examples (@chinmayapancholi13, [#1428](https://github.com/RaRe-Technologies/gensim/pull/1428))
* Replace absolute pathes to relative in notebooks (@vochicong, [#1414](https://github.com/RaRe-Technologies/gensim/pull/1414))
* Fix code-style in keras notebook (@chinmayapancholi13, [#1394](https://github.com/RaRe-Technologies/gensim/pull/1394))
* Replace absolute pathes to relative in notebooks (@vochicong, [#1407](https://github.com/RaRe-Technologies/gensim/pull/1407))
* Fix typo in quickstart guide (@vochicong, [#1404](https://github.com/RaRe-Technologies/gensim/pull/1404))
* Update docstring for WordRank. Fix #1384 (@parulsethi, [#1378](https://github.com/RaRe-Technologies/gensim/pull/1378))
* Update docstring for SkLdaModel (@chinmayapancholi13, [#1382](https://github.com/RaRe-Technologies/gensim/pull/1382))
* Update logic for updatetype in LdaModel (@chinmayapancholi13, [#1389](https://github.com/RaRe-Technologies/gensim/pull/1389))
* Update docstring for Doc2Vec (@jstol, [#1379](https://github.com/RaRe-Technologies/gensim/pull/1379))
* Fix docstring for KL-distance (@viciousstar, [#1373](https://github.com/RaRe-Technologies/gensim/pull/1373))
* Update Corpora_and_Vector_Spaces tutorial (@charliejharrison, [#1308](https://github.com/RaRe-Technologies/gensim/pull/1308))
* Add visualization for difference between LdaModel (@menshikh-iv, [#1374](https://github.com/RaRe-Technologies/gensim/pull/1374))
* Fix punctuation & typo in changelog (@piskvorky, @menshikh-iv, [#1366](https://github.com/RaRe-Technologies/gensim/pull/1366))
* Fix PEP8 & typo in several PRs (@menshikh-iv, [#1369](https://github.com/RaRe-Technologies/gensim/pull/1369))
* Update docstrings connected with backward compability in for LdaModel (@chinmayapancholi13, [#1365](https://github.com/RaRe-Technologies/gensim/pull/1365))
* Update Corpora_and_Vector_Spaces tutorial (@schuyler1d, [#1360](https://github.com/RaRe-Technologies/gensim/pull/1360))
* Fix typo in Doc2Vec doctsring (@fujiyuu75, [#1356](https://github.com/RaRe-Technologies/gensim/pull/1356))
* Update Annoy tutorial (@pmbaumgartner, [#1355](https://github.com/RaRe-Technologies/gensim/pull/1355))
* Update temp folder in tutorials (@yl2526, [#1352](https://github.com/RaRe-Technologies/gensim/pull/1352))
* Remove spaces after print in Topics_and_Transformation tutorial (@gsimore, [#1354](https://github.com/RaRe-Technologies/gensim/pull/1354))
* Update Dictionary docstring (@oonska, [#1347](https://github.com/RaRe-Technologies/gensim/pull/1347))
* Add section headings in word2vec notebook (@MikeTheReader, [#1348](https://github.com/RaRe-Technologies/gensim/pull/1348))
* Fix broken urls in starter tutorials (@ka7eh, [#1346](https://github.com/RaRe-Technologies/gensim/pull/1346))
* Update quick start notebook (@yardsale8, [#1345](https://github.com/RaRe-Technologies/gensim/pull/1345))
* Fix typo in quick start notebook (@MikeTheReader, [#1344](https://github.com/RaRe-Technologies/gensim/pull/1344))
* Fix docstring in keyedvectors (@chinmayapancholi13, [#1337](https://github.com/RaRe-Technologies/gensim/pull/1337))



## 2.1.0, 2017-05-12

:star2: New features:
* Add modified save_word2vec_format for Doc2Vec, to save document vectors. (@parulsethi, [#1256](https://github.com/RaRe-Technologies/gensim/pull/1256))


:+1: Improvements:
* Add automatic code style check limited only to the code modified in PR (@tmylk, [#1287](https://github.com/RaRe-Technologies/gensim/pull/1287))
* Replace `logger.warn` by `logger.warning` (@chinmayapancholi13, [#1295](https://github.com/RaRe-Technologies/gensim/pull/1295))
* Docs word2vec docstring improvement, deprecation labels (@shubhvachher, [#1274](https://github.com/RaRe-Technologies/gensim/pull/1274))
* Stop passing 'sentences' as parameter to Doc2Vec. Fix #511 (@gogokaradjov, [#1306](https://github.com/RaRe-Technologies/gensim/pull/1306))


:red_circle: Bug fixes:
* Allow indexing with np.int64 in doc2vec. Fix #1231 (@bogdanteleaga, [#1254](https://github.com/RaRe-Technologies/gensim/pull/1254))
* Update Doc2Vec docstring. Fix #1302 (@datapythonista, [#1307](https://github.com/RaRe-Technologies/gensim/pull/1307))
* Ignore rst and ipynb file in Travis flake8 validations (@datapythonista, [#1309](https://github.com/RaRe-Technologies/gensim/pull/1309))


:books: Tutorial and doc improvements:
* Update Tensorboard Doc2Vec notebook (@parulsethi, [#1286](https://github.com/RaRe-Technologies/gensim/pull/1286))
* Update Doc2Vec IMDB Notebook, replace codesc to smart_open (@robotcator, [#1278](https://github.com/RaRe-Technologies/gensim/pull/1278))
* Add explanation of `size` to Word2Vec Notebook (@jbcoe, [#1305](https://github.com/RaRe-Technologies/gensim/pull/1305))
* Add extra param to WordRank notebook. Fix #1276 (@parulsethi, [#1300](https://github.com/RaRe-Technologies/gensim/pull/1300))
* Update warning message in WordRank (@parulsethi, [#1299](https://github.com/RaRe-Technologies/gensim/pull/1299))


## 2.0.0, 2017-04-10

Breaking changes:

Any direct calls to method train() of Word2Vec/Doc2Vec now require an explicit epochs parameter and explicit estimate of corpus size. The most usual way to call `train` is `vec_model.train(sentences, total_examples=self.corpus_count, epochs=self.iter)`
See the [method documentation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L766) for more information.


* Explicit epochs and corpus size in word2vec train(). (@gojomo, @robotcator, [#1139](https://github.com/RaRe-Technologies/gensim/pull/1139), [#1237](https://github.com/RaRe-Technologies/gensim/pull/1237))

New features:
* Add output word prediction in word2vec. Only for negative sampling scheme. See [ipynb]( https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb) (@chinmayapancholi13, [#1209](https://github.com/RaRe-Technologies/gensim/pull/1209))
* scikit_learn wrapper for LSI Model in Gensim (@chinmayapancholi13, [#1244](https://github.com/RaRe-Technologies/gensim/pull/1244))
* Add the 'keep_tokens' parameter to 'filter_extremes'. (@toliwa, [#1210](https://github.com/RaRe-Technologies/gensim/pull/1210))
* Load FastText models with specified encoding (@jayantj, [#1210](https://github.com/RaRe-Technologies/gensim/pull/1189))


Improvements:
* Fix loading large FastText models on Mac. (@jaksmid, [#1196](https://github.com/RaRe-Technologies/gensim/pull/1214))
* Sklearn LDA wrapper now works in sklearn pipeline (@kris-singh, [#1213](https://github.com/RaRe-Technologies/gensim/pull/1213))
* glove2word2vec conversion script refactoring (@parulsethi, [#1247](https://github.com/RaRe-Technologies/gensim/pull/1247))
* Word2vec error message when update called before train . Fix #1162 (@hemavakade, [#1205](https://github.com/RaRe-Technologies/gensim/pull/1205))
* Allow training if model is not modified by "_minimize_model". Add deprecation warning. (@chinmayapancholi13, [#1207](https://github.com/RaRe-Technologies/gensim/pull/1207))
* Update the warning text when building vocab on a trained w2v model (@prakhar2b, [#1190](https://github.com/RaRe-Technologies/gensim/pull/1190))

Bug fixes:

*  Fix word2vec reset_from bug in v1.0.1 Fix #1230. (@Kreiswolke, [#1234](https://github.com/RaRe-Technologies/gensim/pull/1234))

* Distributed LDA: checking the length of docs instead of the boolean value, plus int index conversion (@saparina, [#1191](https://github.com/RaRe-Technologies/gensim/pull/1191))

* syn0_lockf initialised with zero in intersect_word2vec_format() (@KiddoZhu, [#1267](https://github.com/RaRe-Technologies/gensim/pull/1267))

* Fix wordrank max_iter_dump calculation. Fix #1216 (@ajkl, [#1217](https://github.com/RaRe-Technologies/gensim/pull/1217))

* Make SgNegative test use skip-gram (@shubhvachher, [#1252](https://github.com/RaRe-Technologies/gensim/pull/1252))

* pep8/pycodestyle fixes for hanging indents in Summarization module (@SamriddhiJain, [#1202](https://github.com/RaRe-Technologies/gensim/pull/1202))

* WordRank and Mallet wrappers single vs double quote issue in windows. (@prakhar2b, [#1208](https://github.com/RaRe-Technologies/gensim/pull/1208))


* Fix #824 : no corpus in init, but trim_rule in init (@prakhar2b, [#1186](https://github.com/RaRe-Technologies/gensim/pull/1186))

* Hardcode version number. Fix #1138. (@tmylk, [#1138](https://github.com/RaRe-Technologies/gensim/pull/1138))

Tutorial and doc improvements:

* Color dictionary according to topic notebook update (@bhargavvader, [#1164](https://github.com/RaRe-Technologies/gensim/pull/1164))

* Fix hdp show_topic/s docstring (@parulsethi, [#1264](https://github.com/RaRe-Technologies/gensim/pull/1264))

* Add docstrings for word2vec.py forwarding functions (@shubhvachher, [#1251](https://github.com/RaRe-Technologies/gensim/pull/1251))

* updated description for worker_loop function used in score function (@chinmayapancholi13, [#1206](https://github.com/RaRe-Technologies/gensim/pull/1206))

## 1.0.1, 2017-03-03

* Rebuild cumulative table on load. Fix #1180. (@tmylk, [#1181](https://github.com/RaRe-Technologies/gensim/pull/893))
* most_similar_cosmul bug fix (@dkim010, [#1177](https://github.com/RaRe-Technologies/gensim/pull/1177))
* Fix loading old word2vec models pre-1.0.0  (@jayantj, [#1179](https://github.com/RaRe-Technologies/gensim/pull/1179))
* Load utf-8 words in fasttext  (@jayantj, [#1176](https://github.com/RaRe-Technologies/gensim/pull/1176))


## 1.0.0, 2017-02-24

New features:
* Add Author-topic modeling (@olavurmortensen, [#893](https://github.com/RaRe-Technologies/gensim/pull/893))
* Add FastText word embedding wrapper (@Jayantj, [#847](https://github.com/RaRe-Technologies/gensim/pull/847))
* Add WordRank word embedding  wrapper (@parulsethi, [#1066](https://github.com/RaRe-Technologies/gensim/pull/1066), [#1125](https://github.com/RaRe-Technologies/gensim/pull/1125))
* Add VarEmbed word embedding wrapper (@anmol01gulati, [#1067](https://github.com/RaRe-Technologies/gensim/pull/1067)))
* Add sklearn wrapper for LDAModel (@AadityaJ, [#932](https://github.com/RaRe-Technologies/gensim/pull/932))

Deprecated features:

* Move `load_word2vec_format` and `save_word2vec_format` out of Word2Vec class to KeyedVectors (@tmylk, [#1107](https://github.com/RaRe-Technologies/gensim/pull/1107))
* Move properties `syn0norm`, `syn0`, `vocab`, `index2word` from Word2Vec class to KeyedVectors (@tmylk,[#1147](https://github.com/RaRe-Technologies/gensim/pull/1147))
* Remove support for Python 2.6, 3.3 and 3.4 (@tmylk,[#1145](https://github.com/RaRe-Technologies/gensim/pull/1145))


Improvements:

* Python 3.6 support (@tmylk [#1077](https://github.com/RaRe-Technologies/gensim/pull/1077))
* Phrases and Phraser allow a generator corpus (ELind77 [#1099](https://github.com/RaRe-Technologies/gensim/pull/1099))
* Ignore DocvecsArray.doctag_syn0norm in save. Fix #789 (@accraze, [#1053](https://github.com/RaRe-Technologies/gensim/pull/1053))
* Fix bug in LsiModel that occurs when id2word is a Python 3 dictionary. (@cvangysel, [#1103](https://github.com/RaRe-Technologies/gensim/pull/1103)
* Fix broken link to paper in readme (@bhargavvader, [#1101](https://github.com/RaRe-Technologies/gensim/pull/1101))
* Lazy formatting in evaluate_word_pairs (@akutuzov, [#1084](https://github.com/RaRe-Technologies/gensim/pull/1084))
* Deacc option to keywords pre-processing (@bhargavvader, [#1076](https://github.com/RaRe-Technologies/gensim/pull/1076))
* Generate Deprecated exception when using Word2Vec.load_word2vec_format (@tmylk, [#1165](https://github.com/RaRe-Technologies/gensim/pull/1165))
* Fix hdpmodel constructor docstring for print_topics (#1152) (@toliwa, [#1152](https://github.com/RaRe-Technologies/gensim/pull/1152))
* Default to per_word_topics=False in LDA get_item for performance (@menshikh-iv, [#1154](https://github.com/RaRe-Technologies/gensim/pull/1154))
* Fix bound computation in Author Topic models. (@olavurmortensen, [#1156](https://github.com/RaRe-Technologies/gensim/pull/1156))
* Write UTF-8 byte strings in tensorboard conversion (@tmylk, [#1144](https://github.com/RaRe-Technologies/gensim/pull/1144))
* Make top_topics and sparse2full compatible with numpy 1.12 strictly int idexing (@tmylk, [#1146](https://github.com/RaRe-Technologies/gensim/pull/1146))

Tutorial and doc improvements:

* Clarifying comment in is_corpus func in utils.py (@greninja, [#1109](https://github.com/RaRe-Technologies/gensim/pull/1109))
* Tutorial Topics_and_Transformations fix markdown and add references (@lgmoneda, [#1120](https://github.com/RaRe-Technologies/gensim/pull/1120))
* Fix doc2vec-lee.ipynb results to match previous behavior (@bahbbc, [#1119](https://github.com/RaRe-Technologies/gensim/pull/1119))
* Remove Pattern lib dependency in News Classification tutorial (@luizcavalcanti, [#1118](https://github.com/RaRe-Technologies/gensim/pull/1118))
* Corpora_and_Vector_Spaces tutorial text clarification (@lgmoneda, [#1116](https://github.com/RaRe-Technologies/gensim/pull/1116))
* Update Transformation and Topics link from quick start notebook (@mariana393, [#1115](https://github.com/RaRe-Technologies/gensim/pull/1115))
* Quick Start Text clarification and typo correction (@luizcavalcanti, [#1114](https://github.com/RaRe-Technologies/gensim/pull/1114))
* Fix typos in Author-topic tutorial (@Fil, [#1102](https://github.com/RaRe-Technologies/gensim/pull/1102))
* Address benchmark inconsistencies in Annoy tutorial (@droudy, [#1113](https://github.com/RaRe-Technologies/gensim/pull/1113))
* Add note about Annoy speed depending on numpy BLAS setup in annoytutorial.ipynb (@greninja, [#1137](https://github.com/RaRe-Technologies/gensim/pull/1137))
* Fix dependencies description on doc2vec-IMDB notebook (@luizcavalcanti, [#1132](https://github.com/RaRe-Technologies/gensim/pull/1132))
* Add documentation for WikiCorpus metadata. (@kirit93, [#1163](https://github.com/RaRe-Technologies/gensim/pull/1163))


## 1.0.0RC2, 2017-02-16

* Add note about Annoy speed depending on numpy BLAS setup in annoytutorial.ipynb (@greninja, [#1137](https://github.com/RaRe-Technologies/gensim/pull/1137))
* Remove direct access to properties moved to KeyedVectors (@tmylk, [#1147](https://github.com/RaRe-Technologies/gensim/pull/1147))
* Remove support for Python 2.6, 3.3 and 3.4 (@tmylk, [#1145](https://github.com/RaRe-Technologies/gensim/pull/1145))
* Write UTF-8 byte strings in tensorboard conversion (@tmylk, [#1144](https://github.com/RaRe-Technologies/gensim/pull/1144))
* Make top_topics and sparse2full compatible with numpy 1.12 strictly int idexing (@tmylk, [#1146](https://github.com/RaRe-Technologies/gensim/pull/1146))

## 1.0.0RC1, 2017-01-31

New features:
* Add Author-topic modeling (@olavurmortensen, [#893](https://github.com/RaRe-Technologies/gensim/pull/893))
* Add FastText word embedding wrapper (@Jayantj, [#847](https://github.com/RaRe-Technologies/gensim/pull/847))
* Add WordRank word embedding  wrapper (@parulsethi, [#1066](https://github.com/RaRe-Technologies/gensim/pull/1066), [#1125](https://github.com/RaRe-Technologies/gensim/pull/1125))
* Add sklearn wrapper for LDAModel (@AadityaJ, [#932](https://github.com/RaRe-Technologies/gensim/pull/932))

Improvements:
* Python 3.6 support (@tmylk [#1077](https://github.com/RaRe-Technologies/gensim/pull/1077))
* Phrases and Phraser allow a generator corpus (ELind77 [#1099](https://github.com/RaRe-Technologies/gensim/pull/1099))
* Ignore DocvecsArray.doctag_syn0norm in save. Fix #789 (@accraze, [#1053](https://github.com/RaRe-Technologies/gensim/pull/1053))
* Move load and save word2vec_format out of word2vec class to KeyedVectors  (@tmylk, [#1107](https://github.com/RaRe-Technologies/gensim/pull/1107))
* Fix bug in LsiModel that occurs when id2word is a Python 3 dictionary. (@cvangysel, [#1103](https://github.com/RaRe-Technologies/gensim/pull/1103)
* Fix broken link to paper in readme (@bhargavvader, [#1101](https://github.com/RaRe-Technologies/gensim/pull/1101))
* Lazy formatting in evaluate_word_pairs (@akutuzov, [#1084](https://github.com/RaRe-Technologies/gensim/pull/1084))
* Deacc option to keywords pre-processing (@bhargavvader, [#1076](https://github.com/RaRe-Technologies/gensim/pull/1076))

Tutorial and doc improvements:

* Clarifying comment in is_corpus func in utils.py (@greninja, [#1109](https://github.com/RaRe-Technologies/gensim/pull/1109))
* Tutorial Topics_and_Transformations fix markdown and add references (@lgmoneda, [#1120](https://github.com/RaRe-Technologies/gensim/pull/1120))
* Fix doc2vec-lee.ipynb results to match previous behavior (@bahbbc, [#1119](https://github.com/RaRe-Technologies/gensim/pull/1119))
* Remove Pattern lib dependency in News Classification tutorial (@luizcavalcanti, [#1118](https://github.com/RaRe-Technologies/gensim/pull/1118))
* Corpora_and_Vector_Spaces tutorial text clarification (@lgmoneda, [#1116](https://github.com/RaRe-Technologies/gensim/pull/1116))
* Update Transformation and Topics link from quick start notebook (@mariana393, [#1115](https://github.com/RaRe-Technologies/gensim/pull/1115))
* Quick Start Text clarification and typo correction (@luizcavalcanti, [#1114](https://github.com/RaRe-Technologies/gensim/pull/1114))
* Fix typos in Author-topic tutorial (@Fil, [#1102](https://github.com/RaRe-Technologies/gensim/pull/1102))
* Address benchmark inconsistencies in Annoy tutorial (@droudy, [#1113](https://github.com/RaRe-Technologies/gensim/pull/1113))


## 0.13.4.1, 2017-01-04

* Disable direct access warnings on save and load of Word2vec/Doc2vec (@tmylk, [#1072](https://github.com/RaRe-Technologies/gensim/pull/1072))
* Making Default hs error explicit (@accraze, [#1054](https://github.com/RaRe-Technologies/gensim/pull/1054))
* Removed unnecessary numpy imports (@bhargavvader, [#1065](https://github.com/RaRe-Technologies/gensim/pull/1065))
* Utils and Matutils changes (@bhargavvader, [#1062](https://github.com/RaRe-Technologies/gensim/pull/1062))
* Tests for the evaluate_word_pairs function (@akutuzov, [#1061](https://github.com/RaRe-Technologies/gensim/pull/1061))

## 0.13.4, 2016-12-22

* Added suggested lda model method and print methods to HDP class (@bhargavvader, [#1055](https://github.com/RaRe-Technologies/gensim/pull/1055))
* New class KeyedVectors to store embedding separate from training code (@anmol01gulati and @droudy, [#980](https://github.com/RaRe-Technologies/gensim/pull/980))
* Evaluation of word2vec models against semantic similarity datasets like SimLex-999 (@akutuzov, [#1047](https://github.com/RaRe-Technologies/gensim/pull/1047))
* TensorBoard word embedding visualisation of Gensim Word2vec format (@loretoparisi, [#1051](https://github.com/RaRe-Technologies/gensim/pull/1051))
* Throw exception if load() is called on instance rather than the class in word2vec and doc2vec (@dust0x, [#889](https://github.com/RaRe-Technologies/gensim/pull/889))
* Loading and Saving LDA Models across Python 2 and 3. Fix #853 (@anmolgulati, [#913](https://github.com/RaRe-Technologies/gensim/pull/913), [#1093](https://github.com/RaRe-Technologies/gensim/pull/1093))
* Fix automatic learning of eta (prior over words) in LDA (@olavurmortensen, [#1024](https://github.com/RaRe-Technologies/gensim/pull/1024)).
    * eta should have dimensionality V (size of vocab) not K (number of topics). eta with shape K x V is still allowed, as the user may want to impose specific prior information to each topic.
    * eta is no longer allowed the "asymmetric" option. Asymmetric priors over words in general are fine (learned or user defined).
    * As a result, the eta update (`update_eta`) was simplified some. It also no longer logs eta when updated, because it is too large for that.
    * Unit tests were updated accordingly. The unit tests expect a different shape than before; some unit tests were redundant after the change; `eta='asymmetric'` now should raise an error.
* Optimise show_topics to only call get_lambda once. Fix #1006. (@bhargavvader, [#1028](https://github.com/RaRe-Technologies/gensim/pull/1028))
* HdpModel doc improvement. Inference and print_topics (@dsquareindia, [#1029](https://github.com/RaRe-Technologies/gensim/pull/1029))
* Removing Doc2Vec defaults so that it won't override Word2Vec defaults. Fix #795. (@markroxor, [#929](https://github.com/RaRe-Technologies/gensim/pull/929))
* Remove warning on gensim import "pattern not installed". Fix #1009 (@shashankg7, [#1018](https://github.com/RaRe-Technologies/gensim/pull/1018))
* Add delete_temporary_training_data() function to word2vec and doc2vec models. (@deepmipt-VladZhukov, [#987](https://github.com/RaRe-Technologies/gensim/pull/987))
* Documentation improvements (@IrinaGoloshchapova, [#1010](https://github.com/RaRe-Technologies/gensim/pull/1010), [#1011](https://github.com/RaRe-Technologies/gensim/pull/1011))
* LDA tutorial by Olavur, tips and tricks (@olavurmortensen, [#779](https://github.com/RaRe-Technologies/gensim/pull/779))
* Add double quote in commmand line to run on Windows (@akarazeev, [#1005](https://github.com/RaRe-Technologies/gensim/pull/1005))
* Fix directory names in notebooks to be OS-independent (@mamamot, [#1004](https://github.com/RaRe-Technologies/gensim/pull/1004))
* Respect clip_start, clip_end in most_similar. Fix #601. (@parulsethi, [#994](https://github.com/RaRe-Technologies/gensim/pull/994))
* Replace Python sigmoid function with scipy in word2vec & doc2vec (@markroxor, [#989](https://github.com/RaRe-Technologies/gensim/pull/989))
* WMD to return 0 instead of inf for sentences that contain a single word (@rbahumi, [#986](https://github.com/RaRe-Technologies/gensim/pull/986))
* Pass all the params through the apply call in lda.get_document_topics(), test case to use the per_word_topics through the corpus in test_ldamodel (@parthoiiitm, [#978](https://github.com/RaRe-Technologies/gensim/pull/978))
* Pyro annotations for lsi_worker (@markroxor, [#968](https://github.com/RaRe-Technologies/gensim/pull/968))


## 0.13.3, 2016-10-20

* Add vocabulary expansion feature to word2vec. (@isohyt, [#900](https://github.com/RaRe-Technologies/gensim/pull/900))
* Tutorial: Reproducing Doc2vec paper result on wikipedia. (@isohyt, [#654](https://github.com/RaRe-Technologies/gensim/pull/654))
* Add Save/Load interface to AnnoyIndexer for index persistence (@fortiema, [#845](https://github.com/RaRe-Technologies/gensim/pull/845))
* Fixed issue [#938](https://github.com/RaRe-Technologies/gensim/issues/938),Creating a unified base class for all topic models. ([@markroxor](https://github.com/markroxor), [#946](https://github.com/RaRe-Technologies/gensim/pull/946))
    -  _breaking change in HdpTopicFormatter.show_\__topics_
* Add Phraser for Phrases optimization. ( @gojomo & @anujkhare , [#837](https://github.com/RaRe-Technologies/gensim/pull/837))
* Fix issue #743, in word2vec's n_similarity method if at least one empty list is passed ZeroDivisionError is raised (@pranay360, [#883](https://github.com/RaRe-Technologies/gensim/pull/883))
* Change export_phrases in Phrases model. Fix issue #794 (@AadityaJ, [#879](https://github.com/RaRe-Technologies/gensim/pull/879))
    - bigram construction can now support multiple bigrams within one sentence
* Fix issue [#838](https://github.com/RaRe-Technologies/gensim/issues/838), RuntimeWarning: overflow encountered in exp ([@markroxor](https://github.com/markroxor), [#895](https://github.com/RaRe-Technologies/gensim/pull/895))
*  Change some log messages to warnings as suggested in issue #828. (@rhnvrm, [#884](https://github.com/RaRe-Technologies/gensim/pull/884))
*  Fix issue #851, In summarizer.py, RunTimeError is raised if single sentence input is provided to avoid ZeroDivionError. (@metalaman, #887)
* Fix issue [#791](https://github.com/RaRe-Technologies/gensim/issues/791), correct logic for iterating over SimilarityABC interface. ([@MridulS](https://github.com/MridulS), [#839](https://github.com/RaRe-Technologies/gensim/pull/839))
* Fix RP model loading for large Fortran-order arrays (@piskvorky, [#605](https://github.com/RaRe-Technologies/gensim/issues/938))
* Remove ShardedCorpus from init because of Theano dependency (@tmylk, [#919](https://github.com/RaRe-Technologies/gensim/pull/919))
* Documentation improvements ( @dsquareindia & @tmylk, [#914](https://github.com/RaRe-Technologies/gensim/pull/914), [#906](https://github.com/RaRe-Technologies/gensim/pull/906) )
* Add Annoy memory-mapping example (@harshul1610, [#899](https://github.com/RaRe-Technologies/gensim/pull/899))
* Fixed issue [#601](https://github.com/RaRe-Technologies/gensim/issues/601), correct docID in most_similar for clip range (@parulsethi, [#994](https://github.com/RaRe-Technologies/gensim/pull/994))

## 0.13.2, 2016-08-19

* wordtopics has changed to word_topics in ldamallet, and fixed issue #764. (@bhargavvader, [#771](https://github.com/RaRe-Technologies/gensim/pull/771))
  - assigning wordtopics value of word_topics to keep backward compatibility, for now
* topics, topn parameters changed to num_topics and num_words in show_topics() and print_topics() (@droudy, [#755](https://github.com/RaRe-Technologies/gensim/pull/755))
  - In hdpmodel and dtmmodel
  - NOT BACKWARDS COMPATIBLE!
* Added random_state parameter to LdaState initializer and check_random_state() (@droudy, [#113](https://github.com/RaRe-Technologies/gensim/pull/113))
* Topic coherence update with `c_uci`, `c_npmi` measures.  LdaMallet, LdaVowpalWabbit support. Add `topics` parameter to coherencemodel. Can now provide tokenized topics to calculate coherence value. Faster backtracking. (@dsquareindia, [#750](https://github.com/RaRe-Technologies/gensim/pull/750), [#793](https://github.com/RaRe-Technologies/gensim/pull/793))
* Added a check for empty (no words) documents before starting to run the DTM wrapper if model = "fixed" is used (DIM model) as this    causes the an error when such documents are reached in training. (@eickho, [#806](https://github.com/RaRe-Technologies/gensim/pull/806))
* New parameters `limit`, `datatype` for load_word2vec_format(); `lockf` for intersect_word2vec_format (@gojomo, [#817](https://github.com/RaRe-Technologies/gensim/pull/817))
* Changed `use_lowercase` option in word2vec accuracy to `case_insensitive` to account for case variations in training vocabulary (@jayantj, [#804](https://github.com/RaRe-Technologies/gensim/pull/804)
* Link to Doc2Vec on airline tweets example in tutorials page (@544895340, [#823](https://github.com/RaRe-Technologies/gensim/pull/823))
* Small error on Doc2vec notebook tutorial (@charlessutton, [#816](https://github.com/RaRe-Technologies/gensim/pull/816))
* Bugfix: Full2sparse clipped to use abs value (@tmylk, [#811](https://github.com/RaRe-Technologies/gensim/pull/811))
* WMD docstring: add tutorial link and query example (@tmylk, [#813](https://github.com/RaRe-Technologies/gensim/pull/813))
* Annoy integration to speed word2vec and doc2vec similarity. Tutorial update (@droudy, [#799](https://github.com/RaRe-Technologies/gensim/pull/799),[#792](https://github.com/RaRe-Technologies/gensim/pull/799) )
* Add converter of LDA model between Mallet, Vowpal Wabit and gensim (@dsquareindia, [#798](https://github.com/RaRe-Technologies/gensim/pull/798), [#766](https://github.com/RaRe-Technologies/gensim/pull/766))
* Distributed LDA in different network segments without broadcast (@menshikh-iv, [#782](https://github.com/RaRe-Technologies/gensim/pull/782))
* Update Corpora_and_Vector_Spaces.ipynb (@megansquire, [#772](https://github.com/RaRe-Technologies/gensim/pull/772))
* DTM wrapper bug fixes caused by renaming num_words in #755 (@bhargavvader, [#770](https://github.com/RaRe-Technologies/gensim/pull/770))
* Add LsiModel.docs_processed attribute (@hobson, [#763](https://github.com/RaRe-Technologies/gensim/pull/763))
* Dynamic Topic Modelling in Python. Google Summer of Code 2016 project. (@bhargavvader, [#739](https://github.com/RaRe-Technologies/gensim/pull/739), [#831](https://github.com/RaRe-Technologies/gensim/pull/831))

## 0.13.1, 2016-06-22

* Topic coherence C_v and U_mass (@dsquareindia, #710)

## 0.13.0, 2016-06-21

* Added Distance Metrics to matutils.pt (@bhargavvader, #656)
* Tutorials migrated from website to ipynb (@j9chan, #721), (@jesford, #733), (@jesford, #725), (@jesford, #716)
* New doc2vec intro tutorial (@seanlaw, #730)
* Gensim Quick Start Tutorial (@andrewjlm, #727)
* Add export_phrases(sentences) to model Phrases (hanabi1224 #588)
* SparseMatrixSimilarity returns a sparse matrix if `maintain_sparsity` is True (@davechallis, #590)
* added functionality for Topics of Words in document - i.e, dynamic topics. (@bhargavvader, #704)
  - also included tutorial which explains new functionalities, and document word-topic colring.
* Made normalization an explicit transformation. Added 'l1' norm support (@dsquareindia, #649)
* added term-topics API for most probable topic for word in vocab. (@bhargavvader, #706)
* build_vocab takes progress_per parameter for smaller output (@zer0n, #624)
* Control whether to use lowercase for computing word2vec accuracy. (@alantian, #607)
* Easy import of GloVe vectors using Gensim (Manas Ranjan Kar, #625)
  - Allow easy port of GloVe vectors into Gensim
  - Standalone script with command line arguments, compatible with Python>=2.6
  - Usage: python -m gensim.scripts.glove2word2vec -i glove_vectors.txt -o output_word2vec_compatible.txt
* Add `similar_by_word()` and `similar_by_vector()` to word2vec (@isohyt, #381)
* Convenience method for similarity of two out of training sentences to doc2vec (@ellolo, #707)
* Dynamic Topic Modelling Tutorial updated with Dynamic Influence Model (@bhargavvader, #689)
* Added function to filter 'n' most frequent words from the dictionary (@abhinavchawla, #718)
* Raise warnings if vocab is single character elements and if alpha is increased in word2vec/doc2vec (@dsquareindia, #705)
* Tests for wikidump (@jonmcoe, #723)
* Mallet wrapper sparse format support (@RishabGoel, #664)
* Doc2vec pre-processing script translated from bash to Python (@andrewjlm, #720)


## 0.12.4, 2016-01-29

* Better internal handling of job batching in word2vec (#535)
  - up to 300% speed up when training on very short documents (~tweets)
* Word2vec CLI in line with original word2vec.c (Andrey Kutuzov, #538)
  - Same default values. See diff https://github.com/akutuzov/gensim/commit/6456cbcd75e6f8720451766ba31cc046b4463ae2
  - Standalone script with command line arguments matching those of original C tool.
  - Usage: python -m gensim.scripts.word2vec_standalone -train data.txt -output trained_vec.txt -size 200 -window 2 -sample 1e-4
* Improved load_word2vec_format() performance (@svenkreiss, #555)
  - Remove `init_sims()` call for performance improvements when normalized vectors are not needed.
  - Remove `norm_only` parameter (API change). Call `init_sims(replace=True)` after the `load_word2vec_format()` call for the old `norm_only=True` behavior.
* Word2vec allows non-strict unicode error handling (ignore or replace) (Gordon Mohr, #466)
* Doc2Vec `model.docvecs[key]` now raises KeyError for unknown keys (Gordon Mohr, #520)
* Fix `DocvecsArray.index_to_doctag` so `most_similar()` returns string doctags (Gordon Mohr, #560)
* On-demand loading of the `pattern` library in utils.lemmatize (Jan Zikes, #461)
  - `utils.HAS_PATTERN` flag moved to `utils.has_pattern()`
* Threadsafe Word2Vec/Doc2Vec finish-check to avoid hang/unending Word2Vec/Doc2Vec training (Gordon Mohr, #571)
* Tuned `TestWord2VecModel.test_cbow_hs()` against random failures (Gordon Mohr, #531)
* Prevent ZeroDivisionError when `default_timer()` indicate no elapsed time (Gordon Mohr, #518)
* Forwards compatibility for NumPy > 1.10 (Matti Lyra, #494, #513)
  - LdaModel and LdaMulticore produce a large number of DeprecationWarnings from
    .inference() because the term ids in each chunk returned from utils.grouper
    are floats. This behaviour has been changed so that the term IDs are now ints.
  - utils.grouper returns a python list instead of a numpy array in .update() when
    LdaModel is called in non distributed mode
  - in distributed mode .update() will still call utils.grouper with as_numpy=True
    to save memory
  - LdaModel.update and LdaMulticore.update have a new keyword parameter
    chunks_as_numpy=True/False (defaults to False) that allows controlling
    this behaviour

## 0.12.3, 2015-11-05

* Make show_topics return value consistent across models (Christopher Corley, #448)
  - All models with the `show_topics` method should return a list of
    `(topic_number, topic)` tuples, where `topic` is a list of
    `(word, probability)` tuples.
  - This is a breaking change that affects users of the `LsiModel`, `LdaModel`,
  and `LdaMulticore` that may be reliant on the old tuple layout of
  `(probability, word)`.
* Mixed integer & string document-tags (keys to doc-vectors) will work (Gordon Mohr, #491)
  - DocvecsArray's `index2doctag` list is renamed/reinterpreted as `offset2doctag`
  - `offset2doctag` entries map to `doctag_syn0` indexes *after* last plain-int doctag (if any)
  - (If using only string doctags, `offset2doctag` may be interpreted same as `index2doctag`.)
* New Tutorials on Dynamic Topic Modelling and Classification via Word2Vec (@arttii #471, @mataddy #500)
* Auto-learning for the eta parameter on the LdaModel (Christopher Corley, #479)
* Python 3.5 support
* Speed improvements to keyword and summarisation methods (@erbas #441)
* OSX wheels (#504)
* Win build (#492)

## 0.12.2, 2015-09-19

* tutorial on text summarization (Ólavur Mortensen, #436)
* more flexible vocabulary construction in word2vec & doc2vec (Philipp Dowling, #434)
* added support for sliced TransformedCorpus objects, so that after applying (for instance) TfidfModel the returned corpus remains randomly indexable. (Matti Lyra, #425)
* changed the LdaModel.save so that a custom `ignore` list can be passed in (Matti Lyra, #331)
* added support for NumPy style fancy indexing to corpus objects (Matti Lyra, #414)
* py3k fix in distributed LSI (spacecowboy, #433)
* Windows fix for setup.py (#428)
* fix compatibility for scipy 0.16.0 (#415)

## 0.12.1, 2015-07-20

* improvements to testing, switch to Travis CI containers
* support for loading old word2vec models (<=0.11.1) in 0.12+ (Gordon Mohr, #405)
* various bug fixes to word2vec, doc2vec (Gordon Mohr, #393, #386, #404)
* TextSummatization support for very short texts (Federico Barrios, #390)
* support for word2vec[['word1', 'word2'...]] convenience API calls (Satish Palaniappan, #395)
* MatrixSimilarity supports indexing generator corpora (single pass)

## 0.12.0, 2015-07-06

* complete API, performance, memory overhaul of doc2vec (Gordon Mohr, #356, #373, #380, #384)
  - fast infer_vector(); optional memory-mapped doc vectors; memory savings with int doc IDs
  - 'dbow_words' for combined DBOW & word skip-gram training; new 'dm_concat' mode
  - multithreading & negative-sampling optimizations (also benefitting word2vec)
  - API NOTE: doc vectors must now be accessed/compared through model's 'docvecs' field
    (eg: "model.docvecs['my_ID']" or "model.docvecs.most_similar('my_ID')")
  - https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
* new "text summarization" module (PR #324: Federico Lopez, Federico Barrios)
  - https://github.com/summanlp/docs/raw/master/articulo/articulo-en.pdf
* new matutils.argsort with partial sort
  - performance speedups to all similarity queries (word2vec, Similarity classes...)
* word2vec can compute likelihood scores for classification (Mat Addy, #358)
  - http://arxiv.org/abs/1504.07295
  - http://nbviewer.ipython.org/github/taddylab/deepir/blob/master/w2v-inversion.ipynb
* word2vec supports "encoding" parameter when loading from C format, for non-utf8 models
* more memory-efficient word2vec training (#385)
* fixes to Python3 compatibility (Pavel Kalaidin #330, S-Eugene #369)
* enhancements to save/load format (Liang Bo Wang #363, Gordon Mohr #356)
  - pickle defaults to protocol=2 for better py3 compatibility
* fixes and improvements to wiki parsing (Lukas Elmer #357, Excellent5 #333)
* fix to phrases scoring (Ikuya Yamada, #353)
* speed up of phrases generation (Dave Challis, #349)
* changes to multipass LDA training (Christopher Corley, #298)
* various doc improvements and fixes (Matti Lyra #331, Hongjoo Lee #334)
* fixes and improvements to LDA (Christopher Corley #323)

## 0.11.0 = 0.11.1 = 0.11.1-1, 2015-04-10

* added "topic ranking" to sort topics by coherence in LdaModel (jtmcmc, #311)
* new fast ShardedCorpus out-of-core corpus (Jan Hajic jr., #284)
* utils.smart_open now uses the smart_open package (#316)
* new wrapper for LDA in Vowpal Wabbit (Dave Challis, #304)
* improvements to the DtmModel wrapper (Yang Han, #272, #277)
* move wrappers for external modeling programs into a submodule (Christopher Corley, #295)
* allow transparent compression of NumPy files in save/load (Christopher Corley, #248)
* save/load methods now accept file handles, in addition to file names (macks22, #292)
* fixes to LdaMulticore on Windows (Feng Mai, #305)
* lots of small fixes & py3k compatibility improvements (Chyi-Kwei Yau, Daniel Nouri, Timothy Emerick, Juarez Bochi, Christopher Corley, Chirag Nagpal, Jan Hajic jr., Flávio Codeço Coelho)
* re-released as 0.11.1 and 0.11.1-1 because of a packaging bug

## 0.10.3, 2014-11-17

* added streamed phrases = collocation detection (Miguel Cabrera, #258)
* added param for multiple word2vec epochs (sebastienj, #243)
* added doc2vec (=paragraph2vec = extension of word2vec) model (Timothy Emerick, #231)
* initialize word2vec deterministically, for increased experiment reproducibility (KCzar, #240)
* all indexed corpora now allow full Python slicing syntax (Christopher Corley, #246)
* update distributed code for new Pyro4 API and py3k (Michael Brooks, Marco Bonzanini, #255, #249)
* fixes to six module version (Lars Buitinck, #259)
* fixes to setup.py (Maxim Avanov and Christopher Corley, #260, #251)
* ...and lots of minor fixes & updates all around

## 0.10.2, 2014-09-18

* new parallelized, LdaMulticore implementation (Jan Zikes, #232)
* Dynamic Topic Models (DTM) wrapper (Arttii, #205)
* word2vec compiled from bundled C file at install time: no more pyximport (#233)
* standardize show_/print_topics in LdaMallet (Benjamin Bray, #223)
* add new word2vec multiplicative objective (3CosMul) of Levy & Goldberg (Gordon Mohr, #224)
* preserve case in MALLET wrapper (mcburton, #222)
* support for matrix-valued topic/word prior eta in LdaModel (mjwillson, #208)
* py3k fix to SparseCorpus (Andreas Madsen, #234)
* fix to LowCorpus when switching dictionaries (Christopher Corley, #237)

## 0.10.1, 2014-07-22

* word2vec: new n_similarity method for comparing two sets of words (François Scharffe, #219)
* make LDA print/show topics parameters consistent with LSI (Bram Vandekerckhove, #201)
* add option for efficient word2vec subsampling (Gordon Mohr, #206)
* fix length calculation for corpora on empty files (Christopher Corley, #209)
* improve file cleanup of unit tests (Christopher Corley)
* more unit tests
* unicode now stored everywhere in gensim internally; accepted input stays either utf8 or unicode
* various fixes to the py3k ported code
* allow any dict-like input in Dictionary.from_corpus (Andreas Madsen)
* error checking improvements to the MALLET wrapper
* ignore non-articles during wiki parsig
* utils.lemmatize now (optionally) ignores stopwords

## 0.10.0 (aka "PY3K port"), 2014-06-04

* full Python 3 support (targeting 3.3+, #196)
* all internal methods now expect & store unicode, instead of utf8
* new optimized word2vec functionality: negative sampling, cbow (sebastien-j, #162)
* allow by-frequency sort in Dictionary.save_as_text (Renaud Richardet, #192)
* add topic printing to HDP model (Tiepes, #190)
* new gensim_addons package = optional install-time Cython compilations (Björn Esser, #197)
* added py3.3 and 3.4 to Travis CI tests
* fix a cbow word2vec bug (Liang-Chi Hsieh)

## 0.9.1, 2014-04-12

* MmCorpus fix for Windows
* LdaMallet support for printing/showing topics
* fix LdaMallet bug when user specified a file prefix (Victor, #184)
* fix LdaMallet output when input is single vector (Suvir)
* added LdaMallet unit tests
* more py3k fixes (Lars Buitinck)
* change order of LDA topic printing (Fayimora Femi-Balogun, #188)

## 0.9.0, 2014-03-16

* save/load automatically single out large arrays + allow mmap
* allow .gz/.bz2 corpus filenames => transparently (de)compressed I/O
* CBOW model for word2vec (Sébastien Jean, #176)
* new API for storing corpus metadata (Joseph Chang, #169)
* new LdaMallet class = train LDA using wrapped Mallet
* new MalletCorpus class for corpora in Mallet format (Christopher Corley, #179)
* better Wikipedia article parsing (Joseph Chang, #170)
* word2vec load_word2vec_format uses less memory (Yves Raimond, #164)
* load/store vocabulary files for word2vec C format (Yves Raimond, #172)
* HDP estimation on new documents (Elliot Kulakow, #153)
* store labels in SvmLight corpus (Ritesh, #152)
* fix word2vec binary load on Windows (Stephanus van Schalkwyk)
* replace numpy.svd with scipy.svd for more stability (Sven Döring, #159)
* parametrize LDA constructor (Christopher Corley, #174)
* steps toward py3k compatibility (Lars Buitinck, #154)

## 0.8.9, 2013-12-26

* use travis-ci for continuous integration
* auto-optimize LDA asymmetric prior (Ben Trahan)
* update for new word2vec binary format (Daren Race)
* doc rendering fix (Dan Foreman-Mackey)
* better LDA perplexity logging
* fix Pyro thread leak in distributed algos (Brian Feeny)
* optimizations in word2vec (Bryan Rink)
* allow compressed input in LineSentence corpus (Eric Moyer)
* upgrade ez_setup, doc improvements, minor fixes etc.

## 0.8.8 (aka "word2vec release"), 2013-11-03

* python3 port by Parikshit Samant: https://github.com/samantp/gensimPy3
* massive optimizations to word2vec (cython, BLAS, multithreading): ~20x-300x speedup
* new word2vec functionality (thx to Ghassen Hamrouni, PR #124)
* new CSV corpus class (thx to Zygmunt Zając)
* corpus serialization checks to prevent overwriting (by Ian Langmore, PR #125)
* add context manager support for older Python<=2.6 for gzip and bz2
* added unittests for word2vec

## 0.8.7, 2013-09-18

* initial version of word2vec, a neural network deep learning algo
* make distributed gensim compatible with the new Pyro
* allow merging dictionaries (by Florent Chandelier)
* new design for the gensim website!
* speed up handling of corner cases when returning top-n most similar
* make Random Projections compatible with new scipy (andrewjOc360, PR #110)
* allow "light" (faster) word lemmatization (by Karsten Jeschkies)
* save/load directly from bzip2 files (by Luis Pedro Coelho, PR #101)
* Blei corpus now tries harder to find its vocabulary file (by Luis Pedro Coelho, PR #100)
* sparse vector elements can now be a list (was: only a 2-tuple)
* simple_preprocess now optionally deaccents letters (ř/š/ú=>r/s/u etc.)
* better serialization of numpy corpora
* print_topics() returns the topics, in addition to printing/logging
* fixes for more robust Windows multiprocessing
* lots of small fixes, data checks and documentation updates

## 0.8.6, 2012-09-15

* added HashDictionary (by Homer Strong)
* support for adding target classes in SVMlight format (by Corrado Monti)
* fixed problems with global lemmatizer object when running in parallel on Windows
* parallelization of Wikipedia processing + added script version that lemmatizes the input documents
* added class method to initialize Dictionary from an existing corpus (by Marko Burjek)

## 0.8.5, 2012-07-22

* improved performance of sharding (similarity queries)
* better Wikipedia parsing (thx to Alejandro Weinstein and Lars Buitinck)
* faster Porter stemmer (thx to Lars Buitinck)
* several minor fixes (in HDP model thx to Greg Ver Steeg)
* improvements to documentation

## 0.8.4, 2012-03-09

* better support for Pandas series input (thx to JT Bates)
* a new corpus format: UCI bag-of-words (thx to Jonathan Esterhazy)
* a new model, non-parametric bayes: HDP (thx to Jonathan Esterhazy; based on Chong Wang's code)
* improved support for new scipy versions (thx to Skipper Seabold)
* lemmatizer support for wikipedia parsing (via the `pattern` python package)
* extended the lemmatizer for multi-core processing, to improve its performance

## 0.8.3, 2011-12-02

* fixed Similarity sharding bug (issue #65, thx to Paul Rudin)
* improved LDA code (clarity & memory footprint)
* optimized efficiency of Similarity sharding

## 0.8.2, 2011-10-31

* improved gensim landing page
* improved accuracy of SVD (Latent Semantic Analysis) (thx to Mark Tygert)
* changed interpretation of LDA topics: github issue #57
* took out similarity server code introduced in 0.8.1 (will become a separate project)
* started using `tox` for testing
* + several smaller fixes and optimizations

## 0.8.1, 2011-10-10

* transactional similarity server: see docs/simserver.html
* website moved from university hosting to radimrehurek.com
* much improved speed of lsi[corpus] transformation:
* accuracy tests of incremental svd: test/svd_error.py and http://groups.google.com/group/gensim/browse_thread/thread/4b605b72f8062770
* further improvements to memory-efficiency of LDA and LSA
* improved wiki preprocessing (thx to Luca de Alfaro)
* model.print_topics() debug fncs now support std output, in addition to logging (thx to Homer Strong)
* several smaller fixes and improvements

## 0.8.0 (Armageddon), 2011-06-28

* changed all variable and function names to comply with PEP8 (numTopics->num_topics): BREAKS BACKWARD COMPATIBILITY!
* added support for similarity querying more documents at once (index[query_documents] in addition to index[query_document]; much faster)
* rewrote Similarity so that it is more efficient and scalable (using disk-based mmap'ed shards)
* simplified directory structure (src/gensim/ is now only gensim/)
* several small fixes and optimizations

## 0.7.8, 2011-03-26

* added `corpora.IndexedCorpus`, a base class for corpus serializers (thx to Dieter Plaetinck). This allows corpus formats that inherit from it (MmCorpus, SvmLightCorpus, BleiCorpus etc.) to retrieve individual documents by their id in O(1), e.g. `corpus[14]` returns document #14.
* merged new code from the LarKC.eu team (`corpora.textcorpus`, `models.logentropy_model`, lots of unit tests etc.)
* fixed a bug in `lda[bow]` transformation (was returning gamma distribution instead of theta). LDA model generation was not affected, only transforming new vectors.
* several small fixes and documentation updates

## 0.7.7, 2011-02-13

* new LDA implementation after Hoffman et al.: Online Learning for Latent Dirichlet Allocation
* distributed LDA
* updated LDA docs (wiki experiments, distributed tutorial)
* matrixmarket header now uses capital 'M's: MatrixMarket. (André Lynum reported than Matlab has trouble processing the lowercase version)
* moved code to github
* started gensim Google group

## 0.7.6, 2011-01-10

* added workaround for a bug in numpy: pickling a fortran-order array (e.g. LSA model) and then loading it back and using it results in segfault (thx to Brian Merrel)
* bundled a new version of ez_setup.py: old failed with Python2.6 when setuptools were missing (thx to Alan Salmoni).

## 0.7.5, 2010-11-03

* further optimization to LSA; this is the version used in my NIPS workshop paper
* got rid of SVDLIBC dependency (one-pass LSA now uses stochastic algo for base-base decompositions)

## 0.7.4

* sped up Latent Dirichlet ~10x (through scipy.weave, optional)
* finally, distributed LDA! scales almost linearly, but no tutorial yet. see the tutorial on distributed LSI, everything's completely analogous.
* several minor fixes and improvements; one nasty bug fixed (lsi[corpus] didn't work; thx to Danilo Spinelli)

## 0.7.3

* added stochastic SVD decomposition (faster than the current one-pass LSI algo, but needs two passes over the input corpus)
* published gensim on mloss.org

## 0.7.2

* added workaround for a numpy bug where SVD sometimes fails to converge for no good reason
* changed content of gensims's PyPi title page
* completed HTML tutorial on distributed LSA

## 0.7.1

* fixed a bug in LSA that occurred when the number of features was smaller than the number of topics (thx to Richard Berendsen)

## 0.7.0

* optimized vocabulary generation in gensim.corpora.dictionary (faster and less memory-intense)
* MmCorpus accepts compressed input (file-like objects such as GzipFile, BZ2File; to save disk space)
* changed sparse solver to SVDLIBC (sparsesvd on PyPi) for large document chunks
* added distributed LSA, updated tutorials (still experimental though)
* several minor bug fixes

## 0.6.0

* added option for online LSI training (yay!). the transformation can now be
  used after any amount of training, and training can be continued at any time
  with more data.
* optimized the tf-idf transformation, so that it is a strictly one-pass algorithm in all cases  (thx to Brian Merrell).
* fixed Windows-specific bug in handling binary files (thx to Sutee Sudprasert)
* fixed 1-based feature counting bug in SVMlight format (thx to Richard Berendsen)
* added 'Topic :: Text Processing :: Linguistic' to gensim's pypi classifiers
* change of sphinx documentation css and layout

## 0.5.0

* finished all tutorials, stable version

## 0.4.7

* tutorial on transformations

## 0.4.6

* added Random Projections (aka Random Indexing), as another transformation model.
* several DML-CZ specific updates

## 0.4.5

* updated documentation
* further memory optimizations in SVD (LSI)

## 0.4.4

* added missing test files to MANIFEST.in

## 0.4.3

* documentation changes
* added gensim reference to Wikipedia articles (SVD, LSI, LDA, TFIDF, ...)

## 0.4.2

* finally, a tutorial!
* similarity queries got their own package

## 0.4.1

* pdf documentation
* removed dependency on python2.5 (theoretically, gensim now runs on 2.6 and 2.7 as well).

## 0.4.0

* support for ``python setup.py test``
* fixing package metadata
* documentation clean-up

## 0.2.0

* First version
