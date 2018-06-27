=============
API Reference
=============

.. note::
    This is the new style of API reference. :ref:`The old version <apiref_legacy>`.

.. _models_ref:

:mod:`gensim.models`: Models
============================

.. automodule:: gensim.models
    :no-members:
    :no-inherited-members:

Document ➔ Vector
------------------
.. currentmodule:: gensim.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ldamodel.LdaModel
    ldamulticore.LdaMulticore
    ldaseqmodel.LdaSeqModel
    tfidfmodel.TfidfModel
    rpmodel.RpModel
    hdpmodel.HdpModel
    atmodel.AuthorTopicModel
    doc2vec.Doc2Vec

Word ➔ Vector
--------------
.. currentmodule:: gensim.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    word2vec.Word2Vec
    fasttext.FastText

Utils
-----
.. currentmodule:: gensim.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    keyedvectors.KeyedVectors
    phrases.SentenceAnalyzer
    coherencemodel.CoherenceModel
    basemodel.BaseTopicModel

Distributed
-----------

.. currentmodule:: gensim.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    lsi_worker.Worker
    lsi_dispatcher.Dispatcher
    lda_worker.Worker
    lda_dispatcher.Dispatcher

Wrappers
--------
.. currentmodule:: gensim.models.wrappers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ldamallet.LdaMallet
    dtmmodel.DtmModel
    ldavowpalwabbit.LdaVowpalWabbit
    wordrank.Wordrank
    varembed.VarEmbed
    fasttext.FastText

.. _downloader_ref:

:mod:`gensim.downloader`: Downloader
====================================

.. automodule:: gensim.downloader
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`downloader` section for further details.

Functions
---------
.. currentmodule:: gensim.downloader

.. autosummary::
    :toctree: generated/
    :template: function.rst

    info
    load

.. _scripts_ref:

:mod:`gensim.scripts`: Scripts
==============================

.. automodule:: gensim.scripts
    :no-members:
    :no-inherited-members:

Scripts
-------
.. currentmodule:: gensim.scripts

.. autosummary::
    :toctree: generated/
    :template: module.rst

    glove2word2vec

.. command-output:: python -m gensim.scripts.glove2word2vec --help
    :ellipsis: 0, -5

.. autosummary::
    :toctree: generated/
    :template: module.rst

    word2vec2tensor

.. command-output:: python -m gensim.scripts.word2vec2tensor --help
    :ellipsis: 0, -7

.. autosummary::
    :toctree: generated/
    :template: module.rst

    segment_wiki

.. command-output:: python -m gensim.scripts.segment_wiki --help
    :ellipsis: 0, -10

.. _corpora_ref:

:mod:`gensim.corpora`: Corpora
==============================

.. automodule:: gensim.corpora
    :no-members:
    :no-inherited-members:

Corpora
-------
.. currentmodule:: gensim.corpora

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bleicorpus.BleiCorpus
    csvcorpus.CsvCorpus
    indexedcorpus.IndexedCorpus
    lowcorpus.LowCorpus
    malletcorpus.MalletCorpus
    mmcorpus.MmCorpus
    sharded_corpus.ShardedCorpus
    svmlightcorpus.SvmLightCorpus
    textcorpus.TextCorpus
    ucicorpus.UciCorpus
    wikicorpus.WikiCorpus

Dictionaries
------------
.. currentmodule:: gensim.corpora

.. autosummary::
    :toctree: generated/
    :template: class.rst

    dictionary.Dictionary
    hashdictionary.HashDictionary

.. _similarities_ref:

:mod:`gensim.similarities`: Similarities
========================================

.. automodule:: gensim.similarities
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.similarities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    docsim.Shard
    docsim.Similarity
    docsim.MatrixSimilarity
    docsim.SoftCosineSimilarity
    docsim.SparseMatrixSimilarity
    index.AnnoyIndexer

.. _parsing_ref:

:mod:`gensim.parsing`: Parsing
==============================

.. automodule:: gensim.parsing
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.parsing

.. autosummary::
    :toctree: generated/
    :template: class.rst

    porter.PorterStemmer

Functions
---------
.. currentmodule:: gensim.parsing

.. autosummary::
    :toctree: generated/
    :template: function.rst

    preprocessing.remove_stopwords
    preprocessing.strip_punctuation
    preprocessing.strip_tags
    preprocessing.strip_short
    preprocessing.strip_numeric
    preprocessing.strip_non_alphanum
    preprocessing.strip_multiple_whitespaces
    preprocessing.split_alphanum
    preprocessing.stem_text
    preprocessing.preprocess_string
    preprocessing.preprocess_documents
    preprocessing.read_file
    preprocessing.read_files

.. _summarization_ref:

:mod:`gensim.summarization`: Summarization
==========================================

.. automodule:: gensim.summarization
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.summarization

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bm25.BM25

Functions
---------
.. currentmodule:: gensim.summarization

.. autosummary::
    :toctree: generated/
    :template: function.rst

    summarizer.summarize
    summarizer.summarize_corpus
    mz_entropy.mz_keywords
    bm25.get_bm25_weights
    keywords.keywords

.. _sklearn_api_ref:

:mod:`gensim.sklearn_api`: Sklearn API
======================================

.. automodule:: gensim.sklearn_api
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.sklearn_api

.. autosummary::
    :toctree: generated/
    :template: class.rst

    atmodel.AuthorTopicTransformer
    d2vmodel.D2VTransformer
    hdp.HdpTransformer
    ldamodel.LdaTransformer
    ldaseqmodel.LdaSeqTransformer
    lsimodel.LsiTransformer
    phrases.PhrasesTransformer
    rpmodel.RpTransformer
    text2bow.Text2BowTransformer
    tfidf.TfIdfTransformer
    w2vmodel.W2VTransformer

.. _viz_ref:

:mod:`gensim.viz`: Visualization
================================

.. automodule:: gensim.viz
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gensim.viz

.. autosummary::
    :toctree: generated/
    :template: function.rst

    poincare.poincare_2d_visualization
    poincare.poincare_distance_heatmap

.. _interfaces_ref:

:mod:`gensim.interfaces`: Interfaces
====================================

.. automodule:: gensim.interfaces
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.interfaces

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CorpusABC
    TransformedCorpus
    TransformationABC
    SimilarityABC

.. _utils_ref:

:mod:`gensim.utils`: Utils
==========================

.. automodule:: gensim.utils
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.utils

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SaveLoad
    FakeDict
    RepeatCorpusNTimes
    ClippedCorpus
    SlicedCorpus
    InputQueue

Functions
---------
.. currentmodule:: gensim.utils

.. autosummary::
    :toctree: generated/
    :template: function.rst

    get_random_state
    synchronous
    file_or_filename
    deaccent
    copytree_hardlink
    tokenize
    simple_tokenize
    simple_preprocess
    any2utf8
    any2unicode
    call_on_class_only
    identity
    get_max_id
    dict_from_corpus
    is_corpus
    get_my_ip
    safe_unichr
    decode_htmlentities
    chunkize_serial
    chunkize
    smart_extension
    pickle
    unpickle
    revdict
    deprecated
    randfname
    upload_chunked
    getNS
    pyro_daemon
    has_pattern
    lemmatize
    mock_data_row
    mock_data
    prune_vocab
    qsize
    keep_vocab_item
    check_output
    sample_dict
    strided_windows
    iter_windows
    flatten
    lazy_flatten

.. _matutils_ref:

:mod:`gensim.matutils`: Matutils
================================

.. automodule:: gensim.matutils
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.matutils

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Scipy2Corpus
    Dense2Corpus
    Sparse2Corpus
    MmWriter
    MmReader

Functions
---------
.. currentmodule:: gensim.matutils

.. autosummary::
    :toctree: generated/
    :template: function.rst

    blas
    argsort
    corpus2csc
    pad
    zeros_aligned
    ismatrix
    any2sparse
    scipy2scipy_clipped
    scipy2sparse
    sparse2full
    full2sparse
    full2sparse_clipped
    corpus2dense
    veclen
    ret_normalized_vec
    ret_log_normalize_vec
    unitvec
    cossim
    isbow
    kullback_leibler
    jensen_shannon
    hellinger
    jaccard
    jaccard_distance
    dirichlet_expectation
    qr_destroy

.. _c_matutils_ref:

:mod:`gensim._matutils`: C Matutils
===================================

.. automodule:: gensim._matutils
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gensim._matutils

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mean_absolute_difference
    logsumexp
    dirichlet_expectation
    dirichlet_expectation_1d
    dirichlet_expectation_2d
