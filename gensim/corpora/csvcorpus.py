#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Corpus in CSV format. TODO: still serializes to SVMLight format, feel free to change this.
"""


from __future__ import with_statement

import logging
import csv

from gensim.corpora import IndexedCorpus
from gensim.corpora import SvmLightCorpus

logger = logging.getLogger('gensim.corpora.csvcorpus')


class CsvCorpus(IndexedCorpus):
    """
    Corpus in CSV format.

    """

    def __init__(self, fname, labels):
        """
        Initialize the corpus from a file. labels (bool) - whether labels are present in the input file.
        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s" % fname)

        self.fname = fname # input file, see class doc for format
        self.length = None
        self.labels = labels


    def __iter__(self):
        """
        Iterate over the corpus, returning one sparse vector at a time.
        """
        length = 0
        reader = csv.reader(open(self.fname))
        #line_no = 0
        
        for line in reader:
            doc = self.line2doc(line)
            if doc is not None:
                length += 1
                #line_no += 1
                #if line_no % 1000 == 0:
                    #print line_no
                    
                yield doc
        self.length = length
        
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, labels=False):
        """
        Save a corpus in the SVMlight format.

        The SVMlight `<target>` class tag is taken from the `labels` array, or set
        to 0 for all documents if `labels` is not supplied.

        This function is automatically called by `SvmLightCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        logger.info("converting corpus to SVMlight format: %s" % fname)

        offsets = []
        with open(fname, 'w') as fout:
            for docno, doc in enumerate(corpus):
                label = labels[docno] if labels else 0 # target class is 0 by default
                offsets.append(fout.tell())
                fout.write(SvmLightCorpus.doc2line(doc, label))
        return offsets


    def docbyoffset(self, offset):
        """
        Return the document stored at file position `offset`.
        """
        with open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())

    def line2doc(self, line):
        if self.labels:
            line.pop(0)
        line = map(float, line)
        indexes = range(len(line))
        doc = zip(indexes, line)
        return doc

    # used by save_corpus
    @staticmethod
    def doc2line(doc, label=0):
        """
        Output the document in SVMlight format, as a string. Inverse function to `line2doc`.
        """
        pairs = ' '.join("%i:%s" % (termid + 1, termval) for termid, termval in doc) # +1 to convert 0-base to 1-base
        return str(label) + " %s\n" % pairs
#endclass SvmLightCorpus
