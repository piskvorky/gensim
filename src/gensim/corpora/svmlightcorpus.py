#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in SVMlight format.
"""


import logging

from gensim import interfaces


class SvmLightCorpus(interfaces.CorpusABC):
    """
    Corpus in SVMlight format.
    
    Quoting http://svmlight.joachims.org/:
    The input file example_file contains the training examples. The first lines 
    may contain comments and are ignored if they start with #. Each of the following 
    lines represents one training example and is of the following format::
    
        <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
        <target> .=. +1 | -1 | 0 | <float> 
        <feature> .=. <integer> | "qid"
        <value> .=. <float>
        <info> .=. <string>
    
    The "qid" feature (used for SVMlight ranking), if present, is ignored.
    """
    
    def __init__(self, fname):
        """
        Initialize the corpus from a file.
        """
        logging.info("loading corpus from %s" % fname)
        
        self.fname = fname # input file, see class doc for format
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
            line = line[: line.find('#')].strip()
            if not line:
                continue # ignore comments and empty lines
            parts = line.split()
            if not parts:
                raise ValueError('invalid format at line no. %i in %s' %
                                 (lineNo, self.fname))
            target, fields = parts[0], [part.rsplit(':', 1) for part in parts[1:]]
            doc = [(int(p1), float(p2)) for p1, p2 in fields if p1 != 'qid'] # ignore 'qid' features
            yield doc
    

    @staticmethod
    def saveCorpus(fname, corpus, id2word = None):
        """
        Save a corpus in the SVMlight format.
        """
        logging.info("converting corpus to SVMlight format: %s" % fname)
        fout = open(fname, 'w')
        for doc in corpus:
            fout.write("%i %s\n" % (0, ' '.join("%i:%f" % p for p in doc)))
        fout.close()
#endclass SvmLightCorpus

