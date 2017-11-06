#!/usr/bin/env python
# encoding: utf-8


"""
Common utils for tests
"""
import tempfile
import os

from gensim.corpora import Dictionary

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    return os.path.join(module_path, 'test_data', fname)


def get_tmpfile(suffix):
    return os.path.join(tempfile.gettempdir(), suffix)


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
