#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Corpus in CSV format."""


from __future__ import with_statement

import logging
import csv
import itertools

from gensim import interfaces, utils

logger = logging.getLogger(__name__)


class CsvCorpus(interfaces.CorpusABC):
    """Corpus in CSV format.

    Notes
    -----
    The CSV delimiter, headers etc. are guessed automatically based on the file content.
    All row values are expected to be ints/floats.

    """

    def __init__(self, fname, labels):
        """

        Parameters
        ----------
        fname : str
            Path to corpus.
        labels : bool
            If True - ignore first column (class labels).

        """
        logger.info("loading corpus from %s", fname)
        self.fname = fname
        self.length = None
        self.labels = labels

        # load the first few lines, to guess the CSV dialect
        with utils.open(self.fname, 'rb') as f:
            head = ''.join(itertools.islice(f, 5))

        self.headers = csv.Sniffer().has_header(head)
        self.dialect = csv.Sniffer().sniff(head)
        logger.info("sniffed CSV delimiter=%r, headers=%s", self.dialect.delimiter, self.headers)

    def __iter__(self):
        """Iterate over the corpus, returning one BoW vector at a time.

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        with utils.open(self.fname, 'rb') as f:
            reader = csv.reader(f, self.dialect)
            if self.headers:
                next(reader)    # skip the headers

            line_no = -1
            for line_no, line in enumerate(reader):
                if self.labels:
                    line.pop(0)  # ignore the first column = class label
                yield list(enumerate(float(x) for x in line))

            self.length = line_no + 1  # store the total number of CSV rows = documents
