#!/usr/bin/env python
# encoding: utf-8


"""
Common utils for tests
"""
import contextlib
import tempfile
import os
import shutil

from gensim.corpora import Dictionary

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    """Return full path to the pre created file with test data (basically corpus)."""
    return os.path.join(module_path, 'test_data', fname)


def get_tmpfile(suffix):
    """
    Return full path to temporary file with required suffix.

    Function doesn't create file. Double calling with the same suffix can return different paths.
    """
    return os.path.join(tempfile.gettempdir(), suffix)


@contextlib.contextmanager
def temporary_file(name=""):
    """create a temporary directory and return a path to "name" in that directory

    At the end of the context, the directory is removed.

    The function doesn't create the file.
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
