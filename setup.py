#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Run with::

    python ./setup.py install
"""

import distutils.cmd
import distutils.log
import itertools
import os
import platform
import shutil
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

PY2 = sys.version_info[:2] == (2, 7)
if sys.version_info[:2] < (3, 5) and not PY2:
    raise Exception('This version of gensim requires Py2.7, or Py3.5 or greater')

c_extensions = {
    'gensim.models.word2vec_inner': 'gensim/models/word2vec_inner.c',
    'gensim.corpora._mmreader': 'gensim/corpora/_mmreader.c',
    'gensim.models.fasttext_inner': 'gensim/models/fasttext_inner.c',
    'gensim.models._utils_any2vec': 'gensim/models/_utils_any2vec.c',
    'gensim._matutils': 'gensim/_matutils.c',
    'gensim.models.nmf_pgd': 'gensim/models/nmf_pgd.c',
}

cpp_extensions = {
    'gensim.models.doc2vec_inner': 'gensim/models/doc2vec_inner.cpp',
    'gensim.models.word2vec_corpusfile': 'gensim/models/word2vec_corpusfile.cpp',
    'gensim.models.fasttext_corpusfile': 'gensim/models/fasttext_corpusfile.cpp',
    'gensim.models.doc2vec_corpusfile': 'gensim/models/doc2vec_corpusfile.cpp',
}

if os.name == 'nt' and PY2:
    #
    # The corpusfile extensions don't work under Py2.7 on Win, so disable them.
    #
    cpp_extensions = {
        'gensim.models.doc2vec_inner': 'gensim/models/doc2vec_inner.cpp',
    }


def need_cython():
    """Return True if we need Cython to translate any of the extensions.

    If the extensions have already been translated to C/C++, then we don't need
    to install Cython and perform the translation."""
    expected = list(c_extensions.values()) + list(cpp_extensions.values())
    return any([not os.path.isfile(f) for f in expected])


def make_c_ext(use_cython=False):
    for module, source in c_extensions.items():
        if use_cython:
            source = source.replace('.c', '.pyx')
        yield Extension(module, sources=[source], language='c')


def make_cpp_ext(use_cython=False):
    extra_args = []
    system = platform.system()

    if system == 'Linux':
        extra_args.append('-std=c++11')
    elif system == 'Darwin':
        extra_args.extend(['-stdlib=libc++', '-std=c++11'])

    for module, source in cpp_extensions.items():
        if use_cython:
            source = source.replace('.cpp', '.pyx')
        yield Extension(
            module,
            sources=[source],
            language='c++',
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
        )


#
# We use use_cython=False here for two reasons:
#
# 1. Cython may not be available at this stage
# 2. The actual translation from Cython to C/C++ happens inside CustomBuildExt
#
ext_modules = list(itertools.chain(make_c_ext(use_cython=False), make_cpp_ext(use_cython=False)))


class CustomBuildExt(build_ext):
    """Custom build_ext action with bootstrapping.

    We need this in order to use numpy and Cython in this script without
    importing them at module level, because they may not be available yet.
    """
    #
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    #
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())

        if need_cython():
            import Cython.Build
            Cython.Build.cythonize(list(make_c_ext(use_cython=True)))
            Cython.Build.cythonize(list(make_cpp_ext(use_cython=True)))


class CleanExt(distutils.cmd.Command):
    description = 'Remove C sources, C++ sources and binaries for gensim extensions'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root, dirs, files in os.walk('gensim'):
            files = [
                os.path.join(root, f)
                for f in files
                if os.path.splitext(f)[1] in ('.c', '.cpp', '.so')
            ]
            for f in files:
                self.announce('removing %s' % f, level=distutils.log.INFO)
                os.unlink(f)

        if os.path.isdir('build'):
            self.announce('recursively removing build', level=distutils.log.INFO)
            shutil.rmtree('build')


cmdclass = {'build_ext': CustomBuildExt, 'clean_ext': CleanExt}

WHEELHOUSE_UPLOADER_COMMANDS = {'fetch_artifacts', 'upload_all'}
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd
    cmdclass.update(vars(wheelhouse_uploader.cmd))


LONG_DESCRIPTION = u"""
==============================================
gensim -- Topic Modelling in Python
==============================================

|Travis|_
|Wheel|_

.. |Travis| image:: https://img.shields.io/travis/RaRe-Technologies/gensim/develop.svg
.. |Wheel| image:: https://img.shields.io/pypi/wheel/gensim.svg

.. _Travis: https://travis-ci.org/RaRe-Technologies/gensim
.. _Downloads: https://pypi.python.org/pypi/gensim
.. _License: http://radimrehurek.com/gensim/about.html
.. _Wheel: https://pypi.python.org/pypi/gensim

Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
Target audience is the *natural language processing* (NLP) and *information retrieval* (IR) community.

Features
---------

* All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM, streamed, out-of-core),
* **Intuitive interfaces**

  * easy to plug in your own input corpus/datastream (simple streaming API)
  * easy to extend with other Vector Space algorithms (simple transformation API)

* Efficient multicore implementations of popular algorithms, such as online **Latent Semantic Analysis (LSA/LSI/SVD)**,
  **Latent Dirichlet Allocation (LDA)**, **Random Projections (RP)**, **Hierarchical Dirichlet Process (HDP)** or **word2vec deep learning**.
* **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers.
* Extensive `documentation and Jupyter Notebook tutorials <https://github.com/RaRe-Technologies/gensim/#documentation>`_.


If this feature list left you scratching your head, you can first read more about the `Vector
Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised
document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ on Wikipedia.

Installation
------------

This software depends on `NumPy and Scipy <http://www.scipy.org/Download>`_, two Python packages for scientific computing.
You must have them installed prior to installing `gensim`.

It is also recommended you install a fast BLAS library before installing NumPy. This is optional, but using an optimized BLAS such as `ATLAS <http://math-atlas.sourceforge.net/>`_ or `OpenBLAS <http://xianyi.github.io/OpenBLAS/>`_ is known to improve performance by as much as an order of magnitude. On OS X, NumPy picks up the BLAS that comes with it automatically, so you don't need to do anything special.

Install the latest version of gensim::

    pip install --upgrade gensim

Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/gensim>`_ package::

    python setup.py install


For alternative modes of installation, see the `documentation <http://radimrehurek.com/gensim/install.html>`_.

Gensim is being `continuously tested <https://travis-ci.org/RaRe-Technologies/gensim>`_ under Python 3.5, 3.6, 3.7 and 3.8.
Support for Python 2.7 was dropped in gensim 4.0.0 – install gensim 3.8.3 if you must use Python 2.7.


How come gensim is so fast and memory efficient? Isn't it pure Python, and isn't Python slow and greedy?
--------------------------------------------------------------------------------------------------------

Many scientific algorithms can be expressed in terms of large matrix operations (see the BLAS note above). Gensim taps into these low-level BLAS libraries, by means of its dependency on NumPy. So while gensim-the-top-level-code is pure Python, it actually executes highly optimized Fortran/C under the hood, including multithreading (if your BLAS is so configured).

Memory-wise, gensim makes heavy use of Python's built-in generators and iterators for streamed data processing. Memory efficiency was one of gensim's `design goals <http://radimrehurek.com/gensim/about.html>`_, and is a central feature of gensim, rather than something bolted on as an afterthought.

Documentation
-------------
* `QuickStart`_
* `Tutorials`_
* `Tutorial Videos`_
* `Official Documentation and Walkthrough`_

Citing gensim
-------------

When `citing gensim in academic papers and theses <https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC>`_, please use this BibTeX entry::

  @inproceedings{rehurek_lrec,
        title = {{Software Framework for Topic Modelling with Large Corpora}},
        author = {Radim {\\v R}eh{\\r u}{\\v r}ek and Petr Sojka},
        booktitle = {{Proceedings of the LREC 2010 Workshop on New
             Challenges for NLP Frameworks}},
        pages = {45--50},
        year = 2010,
        month = May,
        day = 22,
        publisher = {ELRA},
        address = {Valletta, Malta},
        language={English}
  }

----------------

Gensim is open source software released under the `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.
Copyright (c) 2009-now Radim Rehurek

|Analytics|_

.. |Analytics| image:: https://ga-beacon.appspot.com/UA-24066335-5/your-repo/page-name
.. _Analytics: https://github.com/igrigorik/ga-beacon
.. _Official Documentation and Walkthrough: http://radimrehurek.com/gensim/
.. _Tutorials: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials
.. _Tutorial Videos: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#videos
.. _QuickStart: https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_core_concepts.html

"""

distributed_env = ['Pyro4 >= 4.27']

win_testenv = [
    'pytest',
    'pytest-rerunfailures',
    'mock',
    'cython',
    'nmslib',
    'pyemd',
    'testfixtures',
    'Morfessor==2.0.2a4',
    'python-Levenshtein >= 0.10.2',
    'visdom >= 0.1.8, != 0.1.8.7',
    'scikit-learn',
    # The following packages are commented out because they don't install on Windows. So skip the
    # related tests in AppVeyor. We still test them in Linux via Travis, see linux_testenv below.
    # See https://github.com/RaRe-Technologies/gensim/pull/2814
    # 'tensorflow',
    # 'keras',
]

# This list partially duplicates requirements_docs.txt.
# The main difference is that we don't include version pins here unless
# absolutely necessary, whereas requirements_docs.txt includes pins for
# everything, by design.
#
# For more info about the difference between the two:
#
#   https://packaging.python.org/discussions/install-requires-vs-requirements/
#
docs_testenv = win_testenv + distributed_env + [
    'sphinx <= 2.4.4',  # avoid `sphinx >= 3.0` that breaks the build
    'sphinx-gallery',
    'sphinxcontrib.programoutput',
    'sphinxcontrib-napoleon',
    'matplotlib',  # expected by sphinx-gallery
    'plotly',
    #
    # Pattern is a PITA to install, it requires mysqlclient, which in turn
    # requires MySQL dev tools be installed. We don't need it for building
    # documentation.
    #
    # 'Pattern==3.6',  # Need 3.6 or later for Py3 support
    'memory_profiler',
    'annoy',
    'Pyro4',
    'scikit-learn',
    'nltk',
    'testfixtures',
    'statsmodels',
    'pyemd',
    'pandas',
]

# Add additional requirements for testing on Linux. We skip some tests on Windows,
# because the libraries below are too tricky to install there.
linux_testenv = win_testenv[:]
if sys.version_info >= (3, 7):
    # HACK: Installing tensorflow causes a segfault in Travis on py3.6. Other Pythons work – a mystery.
    # See https://github.com/RaRe-Technologies/gensim/pull/2814#issuecomment-621477948
    linux_testenv += [
        'tensorflow',
        'keras',
    ]

#
# smart_open >= 1.11 is py3+ only.
# TODO: Remove the pin once we drop py2.7 from gensim too.
#
if PY2:
    #
    # https://www.scipy.org/scipylib/faq.html#python-version-support
    #
    NUMPY_STR = 'numpy <= 1.16.1'
    SCIPY_STR = 'scipy <= 1.2.3'
    SO_STR = 'smart_open >= 1.8.1, < 1.11'
else:
    NUMPY_STR = 'numpy >= 1.11.3'
    SCIPY_STR = 'scipy >= 0.18.1'
    SO_STR = 'smart_open >= 1.8.1'

#
# We pin the Cython version for reproducibility.  We expect our extensions
# to build with any sane version of Cython, so we should update this pin
# periodically.
#
CYTHON_STR = 'Cython==0.29.14'

install_requires = [
    NUMPY_STR,
    SCIPY_STR,
    'six >= 1.5.0',
    SO_STR,
]

setup_requires = [NUMPY_STR]

if need_cython():
    install_requires.append(CYTHON_STR)
    setup_requires.append(CYTHON_STR)

setup(
    name='gensim',
    version='3.8.3',
    description='Python framework for fast Vector Space Modelling',
    long_description=LONG_DESCRIPTION,

    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),

    author=u'Radim Rehurek',
    author_email='me@radimrehurek.com',

    url='http://radimrehurek.com/gensim',
    download_url='http://pypi.python.org/pypi/gensim',

    license='LGPLv2.1',

    keywords='Singular Value Decomposition, SVD, Latent Semantic Indexing, '
        'LSA, LSI, Latent Dirichlet Allocation, LDA, '
        'Hierarchical Dirichlet Process, HDP, Random Projections, '
        'TFIDF, word2vec',

    platforms='any',

    zip_safe=False,

    classifiers=[  # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    test_suite="gensim.test",
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=linux_testenv,
    extras_require={
        'distributed': distributed_env,
        'test-win': win_testenv,
        'test': linux_testenv,
        'docs': docs_testenv,
    },

    include_package_data=True,
)
