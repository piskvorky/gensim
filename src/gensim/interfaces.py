#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains basic interfaces used throughout the whole gensim package.

The interfaces are realized as abstract base classes (ie., some optional functionality 
is provided in the interface itself, so that the interfaces can be subclassed).
"""


import utils


class CorpusABC(utils.SaveLoad):
    """
    Interface for corpora. A *corpus* is simply an iterable, where each 
    iteration step yields one document. A document is a list of (fieldId, fieldValue)
    2-tuples.
    
    See the corpora package for some example corpus implementations.
    
    Note that although a default len() method is provided, it is very inefficient
    (performs a linear scan through the corpus to determine its length). Wherever 
    the corpus size is needed and known in advance (or at least doesn't change so 
    that it can be cached), the len() method should be overridden.
    """
    def __iter__(self):
        """
        Iterate over the corpus, yielding one document at a time.
        """
        raise NotImplementedError('cannot instantiate abstract base class')

    
    def __len__(self):
        """
        Return the number of documents in the corpus. 
        
        This method is just the least common denominator and should really be 
        overridden when possible.
        """
        logging.warning("performing full corpus scan to determine its length; was this intended?")
        return sum(1 for doc in self) # sum(empty generator) == 0, so this works even for an empty corpus
#endclass CorpusABC


class TransformationABC(utils.SaveLoad):
    """
    Interface for transformations. A 'transformation' is any object which accepts
    a sparse document via the dictionary notation [] and returns another sparse
    document in its stead.
    
    See the :mod:`tfidfmodel` module for an example of a transformation.
    """
    class TransformedCorpus(CorpusABC):
        def __init__(self, fnc, corpus):
            self.fnc, self.corpus = fnc, corpus
        
        def __len__(self):
            return len(self.corpus)
        
        def __iter__(self):
            for doc in self.corpus:
                yield self.fnc(doc) 
    #endclass TransformedCorpus

    def __getitem__(self, vec):
        """
        Transform vector from one vector space into another.
        """
        raise NotImplementedError('cannot instantiate abstract base class')


    def apply(self, corpus):
        """
        Apply the transformation to a whole corpus (as opposed to a single document) 
        and return the result as another another corpus.
        """
        return TransformationABC.TransformedCorpus(self.__getitem__, corpus)
#endclass TransformationABC


