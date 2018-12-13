#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy 

ext_modules=[ Extension("gensim.models.word2vec_inner",
              ["./gensim/models/word2vec_inner.pyx"], include_dirs=[numpy.get_include()],
              libraries=["m"])]

setup(
  name = "fastloop",
  cmdclass = {"build_ext": build_ext},
  ext_modules = cythonize(ext_modules, gdb_debug=True))
