#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Corpus in CSV format.

"""


from __future__ import with_statement

import logging
import csv
import itertools

from gensim import interfaces

logger = logging.getLogger('gensim.corpora.csvcorpus')


class CsvCorpus(interfaces.CorpusABC):
    """
    Corpus in CSV format. The CSV delimiter, headers etc. are guessed automatically
    based on the file content.

    All row values are expected to be ints/floats.

    """

    def __init__(self, fname, has_labels):
        """
        Initialize the corpus from a file.
        `has_labels` = are class labels present in the input file? => skip the first column

        """
        logger.info("loading corpus from %s" % fname)
        self.fname = fname
        self.length = None
        self.has_labels = has_labels
        self.labels = []

        # load the first few lines, to guess the CSV dialect
        head = ''.join(itertools.islice(open(self.fname), 5))
        self.headers = csv.Sniffer().has_header(head)
        self.dialect = csv.Sniffer().sniff(head)
        logger.info("sniffed CSV delimiter=%r, headers=%s" % (self.dialect.delimiter, self.headers))


    def __iter__(self):
        """
        Iterate over the corpus, returning one sparse vector at a time.

        """
        reader = csv.reader(open(self.fname), self.dialect)
        if self.headers:
            reader.next()  # skip the headers
        
        # we need to reset labels, don't we?
        self.labels = []    

        line_no = -1
        for line_no, line in enumerate(reader):
            if self.has_labels:
                label = line.pop(0)  # ignore the first column = class label
                self.labels.append(label)
            yield list(enumerate(map(float, line)))

        self.length = line_no + 1  # store the total number of CSV rows = documents

    def get_labels(self):
        return self.labels        
        