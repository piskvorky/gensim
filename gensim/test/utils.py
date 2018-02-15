#!/usr/bin/env python
# encoding: utf-8

"""Module contains common utilities used in automated code tests for Gensim modules.

Attributes:
-----------
module_path : str
    Full path to this module directory.

common_texts : list of list of str
    Toy dataset.

common_dictionary : :class:`~gensim.corpora.Dictionary`
    Dictionary of toy dataset.

common_corpus : list of list of (int, int)
    Corpus of toy dataset.


Examples:
---------
It's easy to keep objects in temporary folder and reuse'em if needed:

>>> from gensim.models import word2vec
>>> from gensim.test.utils import get_tmpfile, common_texts
>>>
>>> model = word2vec.Word2Vec(common_texts, min_count=1)
>>> temp_path = get_tmpfile('toy_w2v')
>>> model.save(temp_path)
>>>
>>> new_model = word2vec.Word2Vec.load(temp_path)
>>> result = new_model.wv.most_similar("human", topn=1)

Let's print first document in toy dataset and then recreate it using its corpus and dictionary.

>>> from gensim.test.utils import common_texts, common_dictionary, common_corpus
>>> print(common_texts[0])
['human', 'interface', 'computer']
>>> assert common_dictionary.doc2bow(common_texts[0]) == common_corpus[0]

We can find our toy set in test data directory.

>>> from gensim.test.utils import datapath
>>>
>>> with open(datapath("testcorpus.txt")) as f:
...     texts = [line.strip().split() for line in f]
>>> print(texts[0])
['computer', 'human', 'interface']

If you don't need to keep temporary objects on disk use :func:`~gensim.test.utils.temporary_file`:

>>> from gensim.test.utils import temporary_file, common_corpus, common_dictionary
>>> from gensim.models import LdaModel
>>>
>>> with temporary_file("temp.txt") as tf:
...     lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=3)
...     lda.save(tf)

"""

import contextlib
import tempfile
import os
import shutil

from gensim.corpora import Dictionary

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    """Get full path for file `fname` in test data directory placed in this module directory.
    Usually used to place corpus to test_data directory.

    Parameters
    ----------
    fname : str
        Name of file.

    Returns
    -------
    str
        Full path to `fname` in test_data folder.

    Example
    -------
    Let's get path of test GloVe data file and check if it exits.

    >>> from gensim.corpora import MmCorpus
    >>> from gensim.test.utils import datapath
    >>>
    >>> corpus = MmCorpus(datapath("testcorpus.mm"))
    >>> for document in corpus:
    ...     pass


    """
    return os.path.join(module_path, 'test_data', fname)


def get_tmpfile(suffix):
    """Get full path to file `suffix` in temporary folder.
    This function doesn't creates file (only generate unique name).
    Also, it may return different paths in consecutive calling.

    Parameters
    ----------
    suffix : str
        Suffix of file.

    Returns
    -------
    str
        Path to `suffix` file in temporary folder.

    Examples
    --------
    Using this function we may get path to temporary file and use it, for example, to store temporary model.

    >>> from gensim.models import LsiModel
    >>> from gensim.test.utils import get_tmpfile, common_dictionary, common_corpus
    >>>
    >>> tmp_f = get_tmpfile("toy_lsi_model")
    >>>
    >>> model = LsiModel(common_corpus, id2word=common_dictionary)
    >>> model.save(tmp_f)
    >>>
    >>> loaded_model = LsiModel.load(tmp_f)

    """
    return os.path.join(tempfile.gettempdir(), suffix)


@contextlib.contextmanager
def temporary_file(name=""):
    """This context manager creates file `name` in temporary directory and returns its full path.
    Temporary directory with included files will deleted at the end of context. Note, it won't create file.

    Parameters
    ----------
    name : str
        Filename.

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
    >>> with temporary_file("temp.txt") as tf, open(tf, 'w') as outfile:
    ...     outfile.write("my extremely useful information")
    ...     print("Is this file exists? {}".format(os.path.exists(tf)))
    ...     print("Is this folder exists? {}".format(os.path.exists(os.path.dirname(tf))))
    Is this file exists? True
    Is this folder exists? True
    >>>
    >>> print("Is this file exists? {}".format(os.path.exists(tf)))
    Is this file exists? False
    >>> print("Is this folder exists? {}".format(os.path.exists(os.path.dirname(tf))))
    Is this folder exists? False

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
