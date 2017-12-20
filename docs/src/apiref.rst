=============
API Reference
=============

Modules:

.. toctree::
    :maxdepth: 0

    utils
    matutils
    downloader

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

.. _corpora_ref:

:mod:`gensim.corpora`: Corpora
==============================

.. automodule:: gensim.corpora
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.corpora

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bleicorpus.BleiCorpus
    csvcorpus.CsvCorpus
    dictionary.Dictionary
    hashdictionary.HashDictionary
    indexedcorpus.IndexedCorpus
    lowcorpus.LowCorpus
    malletcorpus.MalletCorpus
    mmcorpus.MmCorpus
    sharded_corpus.ShardedCorpus
    svmlightcorpus.SvmLightCorpus
    textcorpus.TextCorpus
    ucicorpus.UciCorpus
    wikicorpus.WikiCorpus

.. _models_ref:

:mod:`gensim.models`: Models
============================

.. automodule:: gensim.models
    :no-members:
    :no-inherited-members:

Models
------
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
    logentropy_model.LogEntropyModel
    normmodel.NormModel
    translation_matrix.TranslationMatrix
    lsi_dispatcher.Dispatcher
    lsi_worker.Worker
    lda_dispatcher.Dispatcher
    lda_worker.Worker
    atmodel.AuthorTopicModel
    word2vec.Word2Vec
    keyedvectors.KeyedVectors
    doc2vec.Doc2Vec
    fasttext.FastText
    phrases.SentenceAnalyzer
    phrases.Phrases
    phrases.Phraser
    coherencemodel.CoherenceModel
    basemodel.BaseTopicModel
    callbacks.Callback

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
    docsim.SparseMatrixSimilarity
    index.AnnoyIndexer

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

.. _topic_coherence_ref:

:mod:`gensim.topic_coherence`: Topic Coherence
==============================================

.. automodule:: gensim.topic_coherence
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gensim.topic_coherence

.. autosummary::
    :toctree: generated/
    :template: class.rst

    indirect_confirmation_measure.ContextVectorComputer
    text_analysis.BaseAnalyzer
    text_analysis.UsesDictionary
    text_analysis.InvertedIndexBased
    text_analysis.CorpusAccumulator
    text_analysis.WindowedTextsAnalyzer
    text_analysis.InvertedIndexAccumulator
    text_analysis.WordOccurrenceAccumulator
    text_analysis.PatchedWordOccurrenceAccumulator
    text_analysis.ParallelWordOccurrenceAccumulator
    text_analysis.AccumulatingWorker
    text_analysis.WordVectorsAccumulator

Functions
---------
.. currentmodule:: gensim.topic_coherence

.. autosummary::
    :toctree: generated/
    :template: function.rst

    aggregation.arithmetic_mean
    direct_confirmation_measure.log_conditional_probability
    direct_confirmation_measure.aggregate_segment_sims
    direct_confirmation_measure.log_ratio_measure
    indirect_confirmation_measure.word2vec_similarity
    indirect_confirmation_measure.cosine_similarity
    probability_estimation.p_boolean_document
    probability_estimation.p_boolean_sliding_window
    probability_estimation.p_word2vec
    probability_estimation.unique_ids_from_segments
    segmentation.s_one_pre
    segmentation.s_one_one
    segmentation.s_one_set

.. _scripts_ref:

:mod:`gensim.scripts`: Scripts
==============================

.. automodule:: gensim.scripts
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gensim.scripts

.. autosummary::
    :toctree: generated/
    :template: function.rst

    glove2word2vec.get_glove_info
    glove2word2vec.glove2word2vec
    word2vec2tensor.word2vec2tensor
    segment_wiki.segment_all_articles
    segment_wiki.segment_and_write_all_articles
    segment_wiki.extract_page_xmls
    segment_wiki.segment

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
