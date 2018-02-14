#!/usr/bin/env python
# encoding: utf-8

"""This module contains common utilities used in automated code tests for Gensim
modules. It includes functions to get full paths to test data and files in
temporary directories. Also it contains context manager which automatically
delete temporary directory (and included files) after use. For your quick
tests you may use tiny toy dataset, its dictionary and corpus which were
used in Gensim tutorial.

Attributes
----------
module_path : str
    Full path to this module directory.

common_texts : list of list of str
    Toy dataset.

common_dictionary : :class:`~gensim.corpora.Dictionary`
    Dictionary of toy dataset.

common_corpus : list of list of (int, int)
    Corpus of toy dataset.

Examples
--------
It's easy to keep objects in temporary folder and reuse'em if needed:

>>> from gensim.models import word2vec
>>> from gensim.test.utils import get_tmpfile, common_texts
>>> model = word2vec.Word2Vec(common_texts, min_count=1)
>>> temp_path = get_tmpfile('toy_w2v')
>>> model.save(temp_path)
>>> new_model = word2vec.Word2Vec.load(temp_path)
>>> new_model.wv.most_similar("human", topn=1)
[('user', 0.159118190407753)]

Let's print first document in toy dataset and then recreate it using its corpus
and dictionary. Order of elements might differ:

>>> from gensim.test.utils import common_texts, common_dictionary, common_corpus
>>> print(common_texts[0])
['human', 'interface', 'computer']
>>> print([common_dictionary[i] for i, _ in common_corpus[0]])
['computer', 'human', 'interface']

We can find our toy set in test data directory. Again, order of words might
be different.

>>> from gensim.test.utils import datapath, common_texts
>>> with open(datapath("testcorpus.txt")) as f:
...     texts = [line.strip().split() for line in f]
>>> print(texts[-1], common_texts[-1])
['survey', 'graph', 'minors'] ['graph', 'minors', 'survey']

If you don't need to keep temporary objects on disk use :func:`~gensim.test.utils.temporary_file`:

>>> from gensim.test.utils import temporary_file, common_corpus
>>> with temporary_file("temp.txt") as tf:
>>>     lda = LdaModel(common_corpus[:-1], num_topics=5)
>>>     lda.save(tf)
>>>     lda_updated = LdaModel.load(tf)
>>>     lda_updated.update(common_corpus[:-1])
>>>     print(lda[common_corpus[-2]])
>>>     print(lda_updated[common_corpus[-2]])
[(0, 0.050019536), (1, 0.35821638), (2, 0.050089914), (3, 0.4916386), (4, 0.050035536)]
[(0, 0.05001102), (1, 0.29562944), (2, 0.050081477), (3, 0.5542563), (4, 0.050021794)]

"""

import contextlib
import tempfile
import os
import shutil

from gensim.corpora import Dictionary

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    """Returns full path for file `fname` in test data directory placed in this
    module directory. Usually used to place corpus to test_data directory.

    Parameters
    ----------
    fname : str
        Filename.

    Returns
    -------
    str
        Full path to `fname` in test_data folder.

    Example
    -------
    Let's get path of test GloVe data file and check if it exits.

    >>> import os
    >>> from gensim.test.utils import datapath
    >>> os.path.exists(datapath("test_glove.txt"))
    True
    
    """

    return os.path.join(module_path, 'test_data', fname)


def get_tmpfile(suffix):
    """Returns full path to file `suffix` in temporary folder. This function
    doesn't creates file. Also it may return different paths in consecutive
    calling.

    Parameters
    ----------
    suffix : str
        Filename.

    Returns
    -------
    str
        Path to `suffix` file in temporary folder.

    Examples
    --------
    Using this function we may get path to temporary file and use it, for
    example, to store temporary model.

    >>> from gensim.models import LsiModel
    >>> from gensim.test.utils import get_tmpfile, common_dictionary, common_corpus
    >>> tmp_f = get_tmpfile("toy_lsi_model")
    >>> model = LsiModel(common_corpus, id2word=common_dictionary)
    >>> model.save(tmp_f)
    >>> loaded_model = LsiModel.load(tmp_f)

    """
    return os.path.join(tempfile.gettempdir(), suffix)


@contextlib.contextmanager
def temporary_file(name=""):
    """This context manager creates file `name` in temporary directory and
    returns its full path. Temporary directory with included files will deleted
    at the end of context. Note, it won't create file.

    Parameters
    ----------
    name : str
        Filename

    Yields
    ------
    str
        Path to file `name` in temporary directory.

    Examples
    --------
    This example demonstrates that created temporary directory (and included
    files) will deleted at the end of context.

    >>> import os
    >>> from gensim.test.utils import temporary_file
    >>> with temporary_file("temp.txt") as tf:
    ...     with open(tf, "w") as f:
    ...         f.write("my extremely useful information")
    ...     print("Is this file exists?", os.path.exists(tf))
    Is this file exists? True
    >>> print("Is file's folder exists now?", os.path.exists(os.path.dirname(tf)))
    But is file's folder exists now? False

    """

    # note : when dropping python2.7 support, we can use tempfile.TemporaryDirectory
    tmp = tempfile.mkdtemp()
    try:
        yield os.path.join(tmp, name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# set up vars used in testing ("Deerwester" from the web tutorial)
common_texts = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]

common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
