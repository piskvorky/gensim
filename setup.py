#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
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
    version = '0.7.0rc1',
    description = 'Python framework for extracting semantic representations from raw texts via fast, distributed Vector Space Modelling',
    long_description = read('README.txt'),
    
    package_dir = {'': 'src'},
    packages = find_packages('src'),

    # there is a bug in python2.5, preventing distutils from using any non-ascii characters :( http://bugs.python.org/issue2562
    author = 'Radim Rehurek', # u'Radim Řehůřek', # <- should really be this...
    author_email = 'radimrehurek@seznam.cz',
    
    url = 'http://nlp.fi.muni.cz/projekty/gensim',
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
        'scipy >= 0.6.0',
        'sparsesvd >= 0.1',
    ],
    
    setup_requires = [
        'sparsesvd >= 0.1',
    ],
    
    extras_require = {
        'distributed': ['Pyro >= 4.1'],
    },

    include_package_data = True,

    # avoid installing system-wide scripts; scripts shall be run by
    # running `python -m module_name` instead. less flexible but more explicit
    # and cleaner
#    entry_points = {
#        'console_scripts': [
#            'lsi_worker = gensim.models.lsi_worker:main',
#            'lsi_dispatcher = gensim.models.lsi_dispatcher:main',
#        ],
#    },

)
