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


import utils, matutils


class CorpusABC(utils.SaveLoad):
    """
    Interface for corpora. A *corpus* is simply an iterable, where each 
    iteration step yields one document. A document is a list of (fieldId, fieldValue)
    2-tuples.
    
    See the `corpora` package for some example corpus implementations.
    
    Note that although a default :func:`len` method is provided, it is very inefficient
    (performs a linear scan through the corpus to determine its length). Wherever 
    the corpus size is needed and known in advance (or at least doesn't change so 
    that it can be cached), the :func:`len` method should be overridden.
    
    See the :mod:`gensim.corpora.mmcorpus` module for an example of a corpus.
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
    a sparse document via the dictionary notation `[]` and returns another sparse
    document in its stead.
    
    See the :mod:`gensim.models.tfidfmodel` module for an example of a transformation.
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
        Transform vector from one vector space into another
        
        **or**
        
        Transform a whole corpus into another.
        """
        raise NotImplementedError('cannot instantiate abstract base class')


    def _apply(self, corpus):
        """
        Apply the transformation to a whole corpus (as opposed to a single document) 
        and return the result as another corpus.
        """
        return TransformationABC.TransformedCorpus(self.__getitem__, corpus)
#endclass TransformationABC


class SimilarityABC(utils.SaveLoad):
    """
    Abstract interface for similarity searches over a corpus.
    
    In all instances, there is a corpus against which we want to perform the 
    similarity search.
    
    For similarity search, the input is a document and the output are its 
    similarities to individual corpus documents.
    
    Similarity queries are realized by calling ``self[query_document]``.
    
    There is also a convenience wrapper, where iterating over `self` yields 
    similarities of each document in the corpus against the whole corpus (ie.,
    the query is each corpus document in turn).
    """
    def __init__(self, corpus):
        raise NotImplementedError("cannot instantiate Abstract Base Class")


    def getSimilarities(self, doc):
        """
        Return similarity of a sparse vector `doc` to all documents in the corpus.
        
        The document is assumed to be either of unit length or empty.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


    def __getitem__(self, doc):
        # get similarities of doc to all documents in the corpus
        if self.normalize:
            doc = matutils.unitVec(doc)
        allSims = self.getSimilarities(doc)
        
        # return either all similarities as a list, or only self.numBest most similar, depending on settings from the constructor
        if self.numBest is None:
            return allSims
        else:
            tops = [(docNo, sim) for docNo, sim in enumerate(allSims) if sim > 0]
            tops = sorted(tops, key = lambda item: -item[1]) # sort by -sim => highest cossim first
            return tops[ : self.numBest] # return at most numBest top 2-tuples (docId, docSim)


    def __iter__(self):
        """
        For each corpus document, compute cosine similarity against all other 
        documents and yield the result.
        """
        for docNo, doc in enumerate(self.corpus):
            yield self[doc]
#endclass SimilarityABC

