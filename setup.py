#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Run with:

sudo python ./setup.py install
"""

import os
import sys

if sys.version_info[:2] < (2, 5):
    raise Exception('This version of gensim needs Python 2.5 or later. ')

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()



setup(
    name = 'gensim',
    version = '0.8.4',
    description = 'Python framework for fast Vector Space Modelling',
    long_description = read('README.rst'),

    packages = find_packages(),

    # there is a bug in python2.5, preventing distutils from using any non-ascii characters :( http://bugs.python.org/issue2562
    author = 'Radim Rehurek', # u'Radim Řehůřek', # <- should really be this...
    author_email = 'radimrehurek@seznam.cz',

    url = 'http://radimrehurek.com/gensim',
    download_url = 'http://pypi.python.org/pypi/gensim',

    keywords = 'Singular Value Decomposition, Latent Semantic Indexing, SVD, LSA, '
    'LSI, LDA, Latent Dirichlet Allocation, VSM, Random Projections, TFIDF',

    license = 'LGPL',
    platforms = 'any',

    zip_safe = False,

    classifiers = [ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    test_suite = "gensim.test",

    install_requires = [
        'scipy >= 0.7.0',
    ],

    extras_require = {
        'distributed': ['Pyro4 >= 4.8'],
    },

    include_package_data = True,

    # lines below are commented out to avoid installing system-wide scripts
    # scripts can be run by running `python -m module_name` instead: less
    # flexible but more explicit and imo cleaner.
#    entry_points = {
#        'console_scripts': [
#            'lsi_worker = gensim.models.lsi_worker:main',
#            'lsi_dispatcher = gensim.models.lsi_dispatcher:main',
#        ],
#    },

)
