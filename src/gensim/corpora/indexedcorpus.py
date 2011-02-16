#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Indexed corpus is a mechanism for random-accessing corpora.

While the standard corpus interface in gensim allows iterating over corpus with
`for doc in corpus: pass`, indexed corpus allows accessing the documents with 
`corpus[docno]`. 

**This is type of access is much slower than the iteration!** It does a disk seek
for every document access.
"""

import logging
import shelve

from gensim import interfaces, utils


class IndexedCorpus(interfaces.CorpusABC):
    def __init__(self, fname, index_fname=None):
        """
        Read previously saved index from `index_fname` and return corpus that behaves 
        just like the original corpus that was passed to `saveCorpus(fname)`, except
        that it also supports `corpus[docno]` (random access to document no. `docno`).
        
        Don't use this for corpus iteration ala `for i in xrange(len(corpus)): doc = corpus[i]`;
        standard `for doc in corpus:` is **much** more efficient.
        
        >>> corpus = [[(1, 0.5)], [(0,1.0), (1,2.0)]]
        
        >>> # save corpus in SvmLightCorpus format with an index
        >>> IndexedCorpus.saveCorpus('testfile.svmlight', corpus, gensim.corpora.SvmLightCorpus)
        
        >>> # load back
        >>> corpus_with_random_access = IndexedCorpus('tstfile.svmlight')
        >>> print corpus_with_random_access[1]
        [(0, 1.0), (1, 2.0)]
        
        """
        if index_fname is None:
            index_fname = fname + '.index'
        
        self.index = shelve.open(index_fname, flag='r')
        serializer = self.index['type']
        self.fname = fname
        self.corpus = serializer(fname)
    

    @staticmethod
    def saveCorpus(fname, corpus, serializer, index_fname=None):
        """
        Iterate through the document stream `corpus`, saving the documents to `fname`
        and recording byte offset of each document. Save the resulting index 
        structure to file `index_fname`.
        
        This relies on the underlying corpus class `serializer` supporting (in 
        addition to standard iteration):
          *  the `serializer.streamposition` attribute, which holds the byte offset
             of the last yielded document during iteration.
          * the `serializer.docbyoffset(offset)` method, which returns a document
            positioned at `offset` bytes within a file.
        
        """
        if index_fname is None:
            index_fname = fname + '.index'
        
        serializer.saveCorpus(fname, corpus)
        corpus = serializer(fname)
        
        logging.info("saving corpus index to %s" % index_fname)
        index = shelve.open(index_fname, flag='n', protocol=-1) # overwrite existing index file, if any
        index['type'] = serializer
        docno = -1
        for docno, doc in enumerate(corpus):
            index[str(docno)] = corpus.streamposition
        docno += 1
        # we've already iterated over the entire corpus, so we might as well remember its length
        index['len'] = docno
        index.close()

    
    def __iter__(self):
        for doc in self.corpus:
            yield doc

    
    def __len__(self):
        return self.index['len']
    
    
    def __getitem__(self, docno):
        return self.corpus.docbyoffset(self.index[str(docno)])
#endclass IndexedCorpus

