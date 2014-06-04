#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
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
from setuptools import setup, find_packages, Extension


# Commonly used information
pkg_name = 'gensim'
pkg_ver = '0.10.0'
pkg_desc = 'Python framework for fast Vector Space Modelling'

# there is a bug in python2.5, preventing distutils from using any non-ascii characters :( http://bugs.python.org/issue2562
pkg_author = 'Radim Rehurek' # u'Radim Řehůřek', # <- should really be this...
pkg_author_email = 'radimrehurek@seznam.cz'
pkg_url = 'http://radimrehurek.com/gensim'
pkg_download_url = 'http://pypi.python.org/pypi/gensim'

pkg_keywords = 'Singular Value Decomposition, SVD, Latent Semantic Indexing, '
'LSA, LSI, Latent Dirichlet Allocation, LDA, '
'Hierarchical Dirichlet Process, HDP, Random Projections, '
'TFIDF, word2vec'

pkg_classifiers = [ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 3.3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
]

pkg_license = 'LGPL'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

native_ext = False

setup(
    name = pkg_name,
    version = pkg_ver,
    description = pkg_desc,
    long_description = read('README.rst'),

    packages = find_packages(exclude=[ pkg_name + '_addons', pkg_name + '_addons.*']),

    author = pkg_author,
    author_email = pkg_author_email,

    url = pkg_url,
    download_url = pkg_download_url,

    keywords = pkg_keywords,

    license = pkg_license,
    platforms = 'any',

    zip_safe = False,

    classifiers = pkg_classifiers,

    test_suite = "gensim.test",

    install_requires = [
        'scipy >= 0.7.0',
        'six >= 1.2.0',
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

# Here comes the setup for cythonized native addon-extension.

try:
    from Cython.Distutils import build_ext
    import numpy
    models_dir = os.path.join(os.path.dirname(__file__), 'gensim', 'models')

    ext_modules = [
        Extension('gensim_addons.models.word2vec_inner',
        ['gensim_addons/models/word2vec_inner.pyx'],
        include_dirs = [models_dir, numpy.get_include()])
    ]

    native_ext = True

except ImportError:
    sys.stderr.write('''
=========================================================

  Please install Cython (http://cython.org/), if you
  want to use the highly optimized version of word2vec.

  Usually you can install it (optional) using:

  pip install -U cython

    or

  easy_install -U cython

    or

  the package-management of your distribution.

  If you install Cython *after* installing gensim, the
  optimized version of word2vec will still be automatically
  generated, on the first use of word2vec.

=========================================================
''')

if native_ext:

    setup(

        name = pkg_name + '_addons',
        version = pkg_ver,
        description = pkg_desc,
        long_description = read('README.rst'),

        packages = find_packages(exclude=[ pkg_name, pkg_name + '.*']),

        author = pkg_author,
        author_email = pkg_author_email,

        url = pkg_url,
        download_url = pkg_download_url,

        keywords = pkg_keywords,

        license = pkg_license,
        platforms = 'any',

        zip_safe = False,

        classifiers = pkg_classifiers,

        install_requires = [
            'gensim == ' + pkg_ver,
        ],

        include_package_data = True,

        cmdclass = {
            'build_ext': build_ext
        },

        ext_modules = ext_modules,
    )
