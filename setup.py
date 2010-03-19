#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Run with:
python ./setup.py install
"""

import os
import sys

if sys.version_info[:2] < (2, 5):
    raise Exception('This version of gensim needs Python 2.5 or later. ')

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages



long_desc = """
Gensim is a Python framework for unsupervised learning from raw, unstructured digital texts.
It provides a framework for learning hidden (*latent*) corpus structure.
Once found, documents can be succinctly expressed in terms of this structure, queried 
for topical similarity and so on.

Gensim includes the following features:
* Memory independence -- there is no need for the whole text corpus (or any 
  intermediate term-document matrices) to reside fully in RAM at any one time.
* Provides implementations for several popular topic inference algorithms, 
  including Latent Semantic Analysis (LSA, LSI) and Latent Dirichlet Allocation (LDA), 
  and makes adding new ones simple.
* Contains I/O wrappers and converters around several popular data formats.
* Allows similarity queries across documents in their latent, topical representation.

The principal design objectives behind gensim are:
1. Straightforward interfaces and low API learning curve for developers, 
   facilitating modifications and rapid prototyping.
2. Memory independence with respect to the size of the input corpus; all intermediate 
   steps and algorithms operate in a streaming fashion, processing one document 
   at a time.
"""


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

long_desc = read('README.txt')



setup(
    name = 'gensim',
    version = '0.4',
    description = 'Python Framework for Topic Modeling',
    long_description = long_desc,
    
    package_dir = {'': 'src'},
    packages = find_packages('src'),

    # there is a bug in python2.5, preventing distutils from using non-ascii characters :(
    author = 'Radim Rehurek', 
    # author = u'Radim Řehůřek', # <- should really be this.. see http://bugs.python.org/issue2562
    author_email = 'radimrehurek@seznam.cz',
    url = 'http://nlp.fi.muni.cz/projekty/gensim',
    download_url = 'http://pypi.python.org/pypi/gensim',
    license = 'LGPL',
    zip_safe = False,
    
    classifiers = [ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    
    test_suite = "gensim.test",

    install_requires=[
        'numpy >= 1.2.0',
        'scipy >= 0.7.0',
    ],
    
    include_package_data = True,

    entry_points = {},

)
