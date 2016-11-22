Changes
=======
0.13.5, 2016-11-12
* Add delete_temporary_training_data() function to word2vec and doc2vec models. (@deepmipt-VladZhukov, [#987](https://github.com/RaRe-Technologies/gensim/pull/987))

0.13.4, 2016-10-25
* Passed all the params through the apply call in lda.get_document_topics(), test case to use the per_word_topics through the corpus in test_ldamodel (@parthoiiitm, [#978](https://github.com/RaRe-Technologies/gensim/pull/978))

0.13.3, 2016-10-20

* Add vocabulary expansion feature to word2vec. (@isohyt, [#900](https://github.com/RaRe-Technologies/gensim/pull/900))
* Tutorial: Reproducing Doc2vec paper result on wikipedia. (@isohyt, [#654](https://github.com/RaRe-Technologies/gensim/pull/654))
* Add Save/Load interface to AnnoyIndexer for index persistence (@fortiema, [#845](https://github.com/RaRe-Technologies/gensim/pull/845))
* Fixed issue [#938](https://github.com/RaRe-Technologies/gensim/issues/938),Creating a unified base class for all topic models. ([@markroxor](https://github.com/markroxor), [#946](https://github.com/RaRe-Technologies/gensim/pull/946))
    -  _breaking change in HdpTopicFormatter.show_\__topics_
* Add Phraser for Phrases optimization. ( @gojomo & @anujkhare , [#837](https://github.com/RaRe-Technologies/gensim/pull/837))
* Fix issue #743, in word2vec's n_similarity method if at least one empty list is passed ZeroDivisionError is raised (@pranay360, [#883](https://github.com/RaRe-Technologies/gensim/pull/883))
* Change export_phrases in Phrases model. Fix issue #794 (@AadityaJ, [#879](https://github.com/RaRe-Technologies/gensim/pull/879))
    - bigram construction can now support multiple bigrams within one sentence
* Fix issue [#838](https://github.com/RaRe-Technologies/gensim/issues/838), RuntimeWarning: overflow encountered in exp  ([@markroxor](https://github.com/markroxor), [#895](https://github.com/RaRe-Technologies/gensim/pull/895))
*  Change some log messages to warnings as suggested in issue #828. (@rhnvrm, [#884](https://github.com/RaRe-Technologies/gensim/pull/884))
*  Fix issue #851, In summarizer.py, RunTimeError is raised if single sentence input is provided to avoid ZeroDivionError. (@metalaman, #887)
* Fix issue [#791](https://github.com/RaRe-Technologies/gensim/issues/791), correct logic for iterating over SimilarityABC interface. ([@MridulS](https://github.com/MridulS), [#839](https://github.com/RaRe-Technologies/gensim/pull/839))
* Fix RP model loading for large Fortran-order arrays (@piskvorky, [#605](https://github.com/RaRe-Technologies/gensim/issues/938))
* Remove ShardedCorpus from init because of Theano dependency (@tmylk, [#919](https://github.com/RaRe-Technologies/gensim/pull/919))
* Documentation improvements ( @dsquareindia & @tmylk, [#914](https://github.com/RaRe-Technologies/gensim/pull/914), [#906](https://github.com/RaRe-Technologies/gensim/pull/906) )
* Add Annoy memory-mapping example (@harshul1610, [#899](https://github.com/RaRe-Technologies/gensim/pull/899))
* Fixed issue [#601](https://github.com/RaRe-Technologies/gensim/issues/601), correct docID in most_similar for clip range (@parulsethi, [#994](https://github.com/RaRe-Technologies/gensim/pull/994))

0.13.2, 2016-08-19

* wordtopics has changed to word_topics in ldamallet, and fixed issue #764. (@bhargavvader, [#771](https://github.com/RaRe-Technologies/gensim/pull/771))
  - assigning wordtopics value of word_topics to keep backward compatibility, for now
* topics, topn parameters changed to num_topics and num_words in show_topics() and print_topics()(@droudy, [#755](https://github.com/RaRe-Technologies/gensim/pull/755))
  - In hdpmodel and dtmmodel
  - NOT BACKWARDS COMPATIBLE!
* Added random_state parameter to LdaState initializer and check_random_state() (@droudy, [#113](https://github.com/RaRe-Technologies/gensim/pull/113))
* Topic coherence update with `c_uci`, `c_npmi` measures.  LdaMallet, LdaVowpalWabbit support. Add `topics` parameter to coherencemodel. Can now provide tokenized topics to calculate coherence value. Faster backtracking. (@dsquareindia, [#750](https://github.com/RaRe-Technologies/gensim/pull/750), [#793](https://github.com/RaRe-Technologies/gensim/pull/793))
* Added a check for empty (no words) documents before starting to run the DTM wrapper if model = "fixed" is used (DIM model) as this    causes the an error when such documents are reached in training. (@eickho, [#806](https://github.com/RaRe-Technologies/gensim/pull/806))
* New parameters `limit`, `datatype` for load_word2vec_format(); `lockf` for intersect_word2vec_format (@gojomo, [#817](https://github.com/RaRe-Technologies/gensim/pull/817))
* Changed `use_lowercase` option in word2vec accuracy to `case_insensitive` to account for case variations in training vocabulary (@jayantj, [#804](https://github.com/RaRe-Technologies/gensim/pull/804)
* Link to Doc2Vec on airline tweets example in tutorials page (@544895340 , [#823](https://github.com/RaRe-Technologies/gensim/pull/823))
* Small error on Doc2vec notebook tutorial (@charlessutton, [#816](https://github.com/RaRe-Technologies/gensim/pull/816))
* Bugfix: Full2sparse clipped to use abs value (@tmylk, [#811](https://github.com/RaRe-Technologies/gensim/pull/811))
* WMD docstring: add tutorial link and query example (@tmylk, [#813](https://github.com/RaRe-Technologies/gensim/pull/813))
* Annoy integration to speed word2vec and doc2vec similarity. Tutorial update (@droudy, [#799](https://github.com/RaRe-Technologies/gensim/pull/799),[#792](https://github.com/RaRe-Technologies/gensim/pull/799) )
* Add converter of LDA model between Mallet, Vowpal Wabit and gensim (@dsquareindia, [#798](https://github.com/RaRe-Technologies/gensim/pull/798), [#766](https://github.com/RaRe-Technologies/gensim/pull/766))
* Distributed LDA in different network segments without broadcast (@menshikh-iv , [#782](https://github.com/RaRe-Technologies/gensim/pull/782))
* Update Corpora_and_Vector_Spaces.ipynb (@megansquire, [#772](https://github.com/RaRe-Technologies/gensim/pull/772))
* DTM wrapper bug fixes caused by renaming num_words in #755 (@bhargavvader,  [#770](https://github.com/RaRe-Technologies/gensim/pull/770))
* Add LsiModel.docs_processed attribute (@hobson, [#763](https://github.com/RaRe-Technologies/gensim/pull/763))
* Dynamic Topic Modelling in Python. Google Summer of Code 2016 project. (@bhargavvader, [#739, #831](https://github.com/RaRe-Technologies/gensim/pull/739))

0.13.1, 2016-06-22

* Topic coherence C_v and U_mass (@dsquareindia, #710)

0.13.0, 2016-06-21

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


0.12.4, 2016-01-29

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

0.12.3, 2015-11-05

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

0.12.2, 2015-09-19

* tutorial on text summarization (Ólavur Mortensen, #436)
* more flexible vocabulary construction in word2vec & doc2vec (Philipp Dowling, #434)
* added support for sliced TransformedCorpus objects, so that after applying (for instance) TfidfModel the returned corpus remains randomly indexable. (Matti Lyra, #425)
* changed the LdaModel.save so that a custom `ignore` list can be passed in (Matti Lyra, #331)
* added support for NumPy style fancy indexing to corpus objects (Matti Lyra, #414)
* py3k fix in distributed LSI (spacecowboy, #433)
* Windows fix for setup.py (#428)
* fix compatibility for scipy 0.16.0 (#415)

0.12.1, 2015-07-20

* improvements to testing, switch to Travis CI containers
* support for loading old word2vec models (<=0.11.1) in 0.12+ (Gordon Mohr, #405)
* various bug fixes to word2vec, doc2vec (Gordon Mohr, #393, #386, #404)
* TextSummatization support for very short texts (Federico Barrios, #390)
* support for word2vec[['word1', 'word2'...]] convenience API calls (Satish Palaniappan, #395)
* MatrixSimilarity supports indexing generator corpora (single pass)

0.12.0, 2015-07-06

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

0.11.0 = 0.11.1 = 0.11.1-1, 2015-04-10

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

0.10.3, 2014-11-17

* added streamed phrases = collocation detection (Miguel Cabrera, #258)
* added param for multiple word2vec epochs (sebastienj, #243)
* added doc2vec (=paragraph2vec = extension of word2vec) model (Timothy Emerick, #231)
* initialize word2vec deterministically, for increased experiment reproducibility (KCzar, #240)
* all indexed corpora now allow full Python slicing syntax (Christopher Corley, #246)
* update distributed code for new Pyro4 API and py3k (Michael Brooks, Marco Bonzanini, #255, #249)
* fixes to six module version (Lars Buitinck, #259)
* fixes to setup.py (Maxim Avanov and Christopher Corley, #260, #251)
* ...and lots of minor fixes & updates all around

0.10.2, 2014-09-18

* new parallelized, LdaMulticore implementation (Jan Zikes, #232)
* Dynamic Topic Models (DTM) wrapper (Arttii, #205)
* word2vec compiled from bundled C file at install time: no more pyximport (#233)
* standardize show_/print_topics in LdaMallet (Benjamin Bray, #223)
* add new word2vec multiplicative objective (3CosMul) of Levy & Goldberg (Gordon Mohr, #224)
* preserve case in MALLET wrapper (mcburton, #222)
* support for matrix-valued topic/word prior eta in LdaModel (mjwillson, #208)
* py3k fix to SparseCorpus (Andreas Madsen, #234)
* fix to LowCorpus when switching dictionaries (Christopher Corley, #237)

0.10.1, 2014-07-22

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

0.10.0 (aka "PY3K port"), 2014-06-04

* full Python 3 support (targeting 3.3+, #196)
* all internal methods now expect & store unicode, instead of utf8
* new optimized word2vec functionality: negative sampling, cbow (sebastien-j, #162)
* allow by-frequency sort in Dictionary.save_as_text (Renaud Richardet, #192)
* add topic printing to HDP model (Tiepes, #190)
* new gensim_addons package = optional install-time Cython compilations (Björn Esser, #197)
* added py3.3 and 3.4 to Travis CI tests
* fix a cbow word2vec bug (Liang-Chi Hsieh)

0.9.1, 2014-04-12

* MmCorpus fix for Windows
* LdaMallet support for printing/showing topics
* fix LdaMallet bug when user specified a file prefix (Victor, #184)
* fix LdaMallet output when input is single vector (Suvir)
* added LdaMallet unit tests
* more py3k fixes (Lars Buitinck)
* change order of LDA topic printing (Fayimora Femi-Balogun, #188)

0.9.0, 2014-03-16

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

0.8.9, 2013-12-26

* use travis-ci for continuous integration
* auto-optimize LDA asymmetric prior (Ben Trahan)
* update for new word2vec binary format (Daren Race)
* doc rendering fix (Dan Foreman-Mackey)
* better LDA perplexity logging
* fix Pyro thread leak in distributed algos (Brian Feeny)
* optimizations in word2vec (Bryan Rink)
* allow compressed input in LineSentence corpus (Eric Moyer)
* upgrade ez_setup, doc improvements, minor fixes etc.

0.8.8 (aka "word2vec release"), 2013-11-03

* python3 port by Parikshit Samant: https://github.com/samantp/gensimPy3
* massive optimizations to word2vec (cython, BLAS, multithreading): ~20x-300x speedup
* new word2vec functionality (thx to Ghassen Hamrouni, PR #124)
* new CSV corpus class (thx to Zygmunt Zając)
* corpus serialization checks to prevent overwriting (by Ian Langmore, PR #125)
* add context manager support for older Python<=2.6 for gzip and bz2
* added unittests for word2vec

0.8.7, 2013-09-18

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

0.8.6, 2012-09-15

* added HashDictionary (by Homer Strong)
* support for adding target classes in SVMlight format (by Corrado Monti)
* fixed problems with global lemmatizer object when running in parallel on Windows
* parallelization of Wikipedia processing + added script version that lemmatizes the input documents
* added class method to initialize Dictionary from an existing corpus (by Marko Burjek)

0.8.5, 2012-07-22

* improved performance of sharding (similarity queries)
* better Wikipedia parsing (thx to Alejandro Weinstein and Lars Buitinck)
* faster Porter stemmer (thx to Lars Buitinck)
* several minor fixes (in HDP model thx to Greg Ver Steeg)
* improvements to documentation

0.8.4, 2012-03-09

* better support for Pandas series input (thx to JT Bates)
* a new corpus format: UCI bag-of-words (thx to Jonathan Esterhazy)
* a new model, non-parametric bayes: HDP (thx to Jonathan Esterhazy; based on Chong Wang's code)
* improved support for new scipy versions (thx to Skipper Seabold)
* lemmatizer support for wikipedia parsing (via the `pattern` python package)
* extended the lemmatizer for multi-core processing, to improve its performance

0.8.3, 2011-12-02

* fixed Similarity sharding bug (issue #65, thx to Paul Rudin)
* improved LDA code (clarity & memory footprint)
* optimized efficiency of Similarity sharding

0.8.2, 2011-10-31

* improved gensim landing page
* improved accuracy of SVD (Latent Semantic Analysis) (thx to Mark Tygert)
* changed interpretation of LDA topics: github issue #57
* took out similarity server code introduced in 0.8.1 (will become a separate project)
* started using `tox` for testing
* + several smaller fixes and optimizations

0.8.1, 2011-10-10

* transactional similarity server: see docs/simserver.html
* website moved from university hosting to radimrehurek.com
* much improved speed of lsi[corpus] transformation:
* accuracy tests of incremental svd: test/svd_error.py and http://groups.google.com/group/gensim/browse_thread/thread/4b605b72f8062770
* further improvements to memory-efficiency of LDA and LSA
* improved wiki preprocessing (thx to Luca de Alfaro)
* model.print_topics() debug fncs now support std output, in addition to logging (thx to Homer Strong)
* several smaller fixes and improvements

0.8.0 (Armageddon), 2011-06-28

* changed all variable and function names to comply with PEP8 (numTopics->num_topics): BREAKS BACKWARD COMPATIBILITY!
* added support for similarity querying more documents at once (index[query_documents] in addition to index[query_document]; much faster)
* rewrote Similarity so that it is more efficient and scalable (using disk-based mmap'ed shards)
* simplified directory structure (src/gensim/ is now only gensim/)
* several small fixes and optimizations

0.7.8, 2011-03-26

* added `corpora.IndexedCorpus`, a base class for corpus serializers (thx to Dieter Plaetinck). This allows corpus formats that inherit from it (MmCorpus, SvmLightCorpus, BleiCorpus etc.) to retrieve individual documents by their id in O(1), e.g. `corpus[14]` returns document #14.
* merged new code from the LarKC.eu team (`corpora.textcorpus`, `models.logentropy_model`, lots of unit tests etc.)
* fixed a bug in `lda[bow]` transformation (was returning gamma distribution instead of theta). LDA model generation was not affected, only transforming new vectors.
* several small fixes and documentation updates

0.7.7, 2011-02-13

* new LDA implementation after Hoffman et al.: Online Learning for Latent Dirichlet Allocation
* distributed LDA
* updated LDA docs (wiki experiments, distributed tutorial)
* matrixmarket header now uses capital 'M's: MatrixMarket. (André Lynum reported than Matlab has trouble processing the lowercase version)
* moved code to github
* started gensim Google group

0.7.6, 2011-01-10

* added workaround for a bug in numpy: pickling a fortran-order array (e.g. LSA model) and then loading it back and using it results in segfault (thx to Brian Merrel)
* bundled a new version of ez_setup.py: old failed with Python2.6 when setuptools were missing (thx to Alan Salmoni).

0.7.5, 2010-11-03

* further optimization to LSA; this is the version used in my NIPS workshop paper
* got rid of SVDLIBC dependency (one-pass LSA now uses stochastic algo for base-base decompositions)

0.7.4

* sped up Latent Dirichlet ~10x (through scipy.weave, optional)
* finally, distributed LDA! scales almost linearly, but no tutorial yet. see the tutorial on distributed LSI, everything's completely analogous.
* several minor fixes and improvements; one nasty bug fixed (lsi[corpus] didn't work; thx to Danilo Spinelli)

0.7.3

* added stochastic SVD decomposition (faster than the current one-pass LSI algo, but needs two passes over the input corpus)
* published gensim on mloss.org

0.7.2

* added workaround for a numpy bug where SVD sometimes fails to converge for no good reason
* changed content of gensims's PyPi title page
* completed HTML tutorial on distributed LSA

0.7.1

* fixed a bug in LSA that occurred when the number of features was smaller than the number of topics (thx to Richard Berendsen)

0.7.0

* optimized vocabulary generation in gensim.corpora.dictionary (faster and less memory-intense)
* MmCorpus accepts compressed input (file-like objects such as GzipFile, BZ2File; to save disk space)
* changed sparse solver to SVDLIBC (sparsesvd on PyPi) for large document chunks
* added distributed LSA, updated tutorials (still experimental though)
* several minor bug fixes

0.6.0

* added option for online LSI training (yay!). the transformation can now be
  used after any amount of training, and training can be continued at any time
  with more data.
* optimized the tf-idf transformation, so that it is a strictly one-pass algorithm in all cases  (thx to Brian Merrell).
* fixed Windows-specific bug in handling binary files (thx to Sutee Sudprasert)
* fixed 1-based feature counting bug in SVMlight format (thx to Richard Berendsen)
* added 'Topic :: Text Processing :: Linguistic' to gensim's pypi classifiers
* change of sphinx documentation css and layout

0.5.0

* finished all tutorials, stable version

0.4.7

* tutorial on transformations

0.4.6

* added Random Projections (aka Random Indexing), as another transformation model.
* several DML-CZ specific updates

0.4.5

* updated documentation
* further memory optimizations in SVD (LSI)

0.4.4

* added missing test files to MANIFEST.in

0.4.3

* documentation changes
* added gensim reference to Wikipedia articles (SVD, LSI, LDA, TFIDF, ...)

0.4.2

* finally, a tutorial!
* similarity queries got their own package

0.4.1

* pdf documentation
* removed dependency on python2.5 (theoretically, gensim now runs on 2.6 and 2.7 as well).

0.4.0

* support for ``python setup.py test``
* fixing package metadata
* documentation clean-up

0.2.0

* First version
