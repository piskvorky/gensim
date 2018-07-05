.. _downloader:

==========
Downloader
==========

Give information about available models/datasets::

    >>> import gensim.downloader as api
    >>>
    >>> api.info()  # return dict with info about available models/datasets
    >>> api.info("text8")  # return dict with info about "text8" dataset

Model example::

    >>> import gensim.downloader as api
    >>>
    >>> model = api.load("glove-twitter-25")  # load glove vectors
    >>> model.most_similar("cat")  # show words that similar to word 'cat'

Dataset example::

    >>> import gensim.downloader as api
    >>> from gensim.models import Word2Vec
    >>>
    >>> dataset = api.load("text8")  # load dataset as iterable
    >>> model = Word2Vec(dataset)  # train w2v model

Also, this API available via CLI::

    python -m gensim.downloader --info <dataname> # same as api.info(dataname)
    python -m gensim.downloader --download <dataname> # same as api.load(dataname, return_path=True)
