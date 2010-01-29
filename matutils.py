#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
This module contains math helper functions.
"""

import logging



class MmWriter(object):
    """
    Store corpus in Matrix Market format.
    
    Note the static determineNnz method, which lets you determine the MM format 
    headers from a corpus (matrix dimensions, number of non-zero entries). If
    these headers are known beforehand from elsewhere, you can skip calling 
    determineNnz() and use writeHeaders() directly.
    """
    
    def __init__(self, fname):
        self.fname = fname
        self.fout = open(self.fname, 'w')
        self.headersWritten = False
    
    @staticmethod
    def determineNnz(corpus):
        logging.info("calculating matrix shape and density")
        numDocs = numTerms = numNnz = 0
        for docNo, bow in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: at document %i/%i" % 
                             (docNo, len(corpus)))
            if len(bow) > 0:
                numDocs = max(numDocs, docNo + 1)
                numTerms = max(numTerms, max(wordId for wordId, _ in bow) + 1)
                numNnz += len(bow)
        
        logging.info("BOW of %ix%i matrix, density=%.3f%% (%i/%i)" % 
                     (numDocs, numTerms,
                      100.0 * numNnz / (numDocs * numTerms),
                      numNnz,
                      numDocs * numTerms))
        return numDocs, numTerms, numNnz
    

    def writeHeaders(self, numDocs, numTerms, numNnz):
        logging.info("saving sparse %sx%s matrix with %i non-zero entries to %s" %
                     (numDocs, numTerms, numNnz, self.fname))
        self.fout.write('%%matrixmarket matrix coordinate real general\n')
        self.fout.write('%i %i %i\n' % (numDocs, numTerms, numNnz))
        self.lastDocNo = -1
        self.headersWritten = True
    
    
    def __del__(self):
        """
        Automatic destructor which closes the underlying file. 
        
        There must be no circular references contained in the object for __del__
        to work! Closing the file explicitly via the close() method is preferred
        and safer.
        """
        self.close() # does nothing if called twice (on an already closed file), so no worries
    
    
    def close(self):
        logging.debug("closing %s" % self.fname)
        self.fout.close()
    
    
    def writeBowVector(self, docNo, pairs):
        """
        Write a single bag-of-words vector to the file.
        """
        assert self.headersWritten, "must write MM file headers before writing data!"
        assert self.lastDocNo < docNo, "documents %i and %i not in sequential order!" % (self.lastDocNo, docNo)
        for termId, weight in sorted(pairs): # write term ids in sorted order
            if weight != 0.0:
                self.fout.write("%i %i %f\n" % (docNo + 1, termId + 1, weight)) # +1 because MM format starts counting from 1
        self.lastDocNo = docNo

    
    def writeCorpus(self, corpus):
        """
        Save bag-of-words representation of an entire corpus to disk.
        
        Note that the documents are processed one at a time, so the whole corpus 
        is allowed to be larger than the available RAM.
        """
        logging.info("saving %i BOW vectors to %s" % (len(corpus), self.fname))
        for docNo, bow in enumerate(corpus):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: saving document %i/%i" % 
                             (docNo, len(corpus)))
            self.writeBowVector(docNo, bow)
#endclass MmWriter


class MmReader(object):
    """
    Wrap a corpus represented as term-document matrix on disk in matrix-market 
    format, and present it as an object which supports iteration over documents. 
    A document = list of (word, weight) 2-tuples. This iterable format is used 
    internally in LDA inference.
    
    Note that the file is read into memory one document at a time, not whole 
    corpus at once. This allows for representing corpora which do not wholly fit 
    in RAM.
    """
    def __init__(self, fname):
        """
        Initialize the corpus reader. The fname is a path to a file on local 
        filesystem, which is expected to be sparse (coordinate) matrix
        market format. Documents are assumed to be rows of the matrix -- if 
        documents are columns, save the matrix transposed.
        """
        logging.info("initializing corpus reader from %s" % fname)
        self.fname = fname
        fin = open(fname)
        header = fin.next()
        if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
            raise ValueError("File %s not in Matrix Market format with coordinate real general" % fname)
        self.noRows = self.noCols = self.noElements = 0
        for lineNo, line in enumerate(fin):
            if not line.startswith('%'):
                self.noRows, self.noCols, self.noElements = map(int, line.split())
                break
        logging.info("accepted corpus with %i documents, %i terms, %i non-zero entries" %
                     (self.noRows, self.noCols, self.noElements))
    
    def __len__(self):
        return self.noRows
        
    def __iter__(self):
        fin = open(self.fname)
        
        # skip headers
        for line in fin:
            if line.startswith('%'):
                continue
            break
        
        prevId = None
        for line in fin:
            docId, termId, val = line.split()
            if docId != prevId:
                if prevId is not None:
                    yield prevId, document
                prevId = docId
                document = []
            # add (termId, weight) pair to the document
            document.append((int(termId) - 1, float(val),)) # -1 because matrix market indexes are 1-based => convert to 0-based
        if prevId is not None: # handle the last document, as a special case
            yield prevId, document
#endclass MmReader

