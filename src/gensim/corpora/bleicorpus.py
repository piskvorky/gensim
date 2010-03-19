#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Blei's LDA-C format.
"""


import logging

from gensim import interfaces, utils


class BleiCorpus(interfaces.CorpusABC):
    """
    Corpus in Blei's LDA-C format.
    
    The corpus is represented as two files: one describing the documents, and another
    describing the mapping between words and their ids.
    
    Each document is one line::
    
      N fieldId1:fieldValue1 fieldId2:fieldValue2 ... fieldIdN:fieldValueN
    
    The vocabulary is a file with words, one word per line; word at line K has an
    implicit ``id=K``.
    """
    
    def __init__(self, fname, fnameVocab = None):
        """
        Initialize the corpus from a file.
        
        `fnameVocab` is the file with vocabulary; if not specified, it defaults to
        `fname.vocab`.
        """
        logging.info("loading corpus from %s" % fname)
        
        if fnameVocab is None:
            fnameVocab = fname + '.vocab'
        
        self.fname = fname # input file, see class doc for format
        words = open(fnameVocab).read().split('\n')
        self.id2word = dict(enumerate(words))
        self.length = None

    
    def __len__(self):
        if self.length is None:
            logging.info("caching corpus length")
            self.length = sum(1 for doc in self)
        return self.length


    def __iter__(self):
        """
        Iterate over the corpus, returning one sparse vector at a time.
        """
        for lineNo, line in enumerate(open(self.fname)):
            parts = line.split()
            if int(parts[0]) != len(parts) - 1:
                raise ValueError("invalid format at line %i in %s" %
                                 (lineNo, self.fname))
            doc = [part.rsplit(':', 1) for part in parts[1:]]
            doc = [(int(p1), float(p2)) for p1, p2 in doc]
            yield doc
    

    @staticmethod
    def saveCorpus(fname, corpus, id2word = None):
        """
        Save a corpus in the Matrix Market format.
        
        There are actually two files saved: `fname` and `fname.vocab`, where
        `fname.vocab` is the vocabulary file.
        """
        if id2word is None:
            logging.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dictFromCorpus(corpus)
            numTerms = len(id2word)
        else:
            numTerms = 1 + max([-1] + id2word.keys())
        
        logging.info("storing corpus in Blei's LDA-C format: %s" % fname)
        fout = open(fname, 'w')
        for doc in corpus:
            fout.write("%i %s\n" % (len(doc), ' '.join("%i:%f" % p for p in doc)))
        fout.close()
        
        # write out vocabulary, in a format compatible with Blei's topics.py script
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary of %i words to %s" % (numTerms, fnameVocab))
        fout = open(fnameVocab, 'w')
        for featureId in xrange(numTerms):
            fout.write("%s\n" % id2word.get(featureId, '---'))
        fout.close()
#endclass BleiCorpus

