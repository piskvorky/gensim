# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:01:49 2016

@author: robert
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize('/run/media/robert/1TB-1/linuxfolder/gitlair/word2vec_syntax/gensim/models/word2vec_inner_in_works.pyx'), include_dirs=[numpy.get_include()])