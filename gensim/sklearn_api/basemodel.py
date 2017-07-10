#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
follows on scikit learn API conventions
"""
from abc import ABCMeta, abstractmethod


class BaseTransformer(object):
    """
    Base sklearn wrapper module
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_params(self, deep=True):
        pass

    @abstractmethod
    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, docs, minimum_probability=None):
        pass

    @abstractmethod
    def partial_fit(self, X):
        pass
