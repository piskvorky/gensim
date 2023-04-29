#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html


"""Corpus in SVMlight format."""


from __future__ import with_statement

import logging

from gensim import utils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger(__name__)


class SvmLightCorpus(IndexedCorpus):
    """Corpus in SVMlight format.

    Quoting http://svmlight.joachims.org/:
    The input file contains the training examples. The first lines  may contain comments and are ignored
    if they start with #. Each of the following lines represents one training example
    and is of the following format::

        <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
        <target> .=. +1 | -1 | 0 | <float>
        <feature> .=. <integer> | "qid"
        <value> .=. <float>
        <info> .=. <string>

    The "qid" feature (used for SVMlight ranking), if present, is ignored.

    Notes
    -----
    Although not mentioned in the specification above, SVMlight also expect its feature ids to be 1-based
    (counting starts at 1). We convert features to 0-base internally by decrementing all ids when loading a SVMlight
    input file, and increment them again when saving as SVMlight.

    """

    def __init__(self, fname, store_labels=True):
        """

        Parameters
        ----------
        fname: str
            Path to corpus.
        store_labels : bool, optional
            Whether to store labels (~SVM target class). They currently have no application but stored
            in `self.labels` for convenience by default.

        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s", fname)

        self.fname = fname  # input file, see class doc for format
        self.length = None
        self.store_labels = store_labels
        self.labels = []

    def __iter__(self):
        """ Iterate over the corpus, returning one sparse (BoW) vector at a time.

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        lineno = -1
        self.labels = []
        with utils.open(self.fname, 'rb') as fin:
            for lineno, line in enumerate(fin):
                doc = self.line2doc(line)
                if doc is not None:
                    if self.store_labels:
                        self.labels.append(doc[1])
                    yield doc[0]
        self.length = lineno + 1

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, labels=False, metadata=False):
        """Save a corpus in the SVMlight format.

        The SVMlight `<target>` class tag is taken from the `labels` array, or set to 0 for all documents
        if `labels` is not supplied.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, float)
            Corpus in BoW format.
        id2word : dict of (str, str), optional
            Mapping id -> word.
        labels : list or False
            An SVMlight `<target>` class tags or False if not present.
        metadata : bool
            ARGUMENT WILL BE IGNORED.

        Returns
        -------
        list of int
            Offsets for each line in file (in bytes).

        """
        logger.info("converting corpus to SVMlight format: %s", fname)

        if labels is not False:
            # Cast any sequence (incl. a numpy array) to a list, to simplify the processing below.
            labels = list(labels)
        offsets = []
        with utils.open(fname, 'wb') as fout:
            for docno, doc in enumerate(corpus):
                label = labels[docno] if labels else 0  # target class is 0 by default
                offsets.append(fout.tell())
                fout.write(utils.to_utf8(SvmLightCorpus.doc2line(doc, label)))
        return offsets

    def docbyoffset(self, offset):
        """Get the document stored at file position `offset`.

        Parameters
        ----------
        offset : int
            Document's position.

        Returns
        -------
        tuple of (int, float)

        """
        with utils.open(self.fname, 'rb') as f:
            f.seek(offset)
            return self.line2doc(f.readline())[0]
            # TODO: it brakes if gets None from line2doc

    def line2doc(self, line):
        """Get a document from a single line in SVMlight format.
        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.doc2line`.

        Parameters
        ----------
        line : str
            Line in SVMLight format.

        Returns
        -------
        (list of (int, float), str)
            Document in BoW format and target class label.

        """
        line = utils.to_unicode(line)
        line = line[: line.find('#')].strip()
        if not line:
            return None  # ignore comments and empty lines
        parts = line.split()
        if not parts:
            raise ValueError('invalid line format in %s' % self.fname)
        target, fields = parts[0], [part.rsplit(':', 1) for part in parts[1:]]
        # ignore 'qid' features, convert 1-based feature ids to 0-based
        doc = [(int(p1) - 1, float(p2)) for p1, p2 in fields if p1 != 'qid']
        return doc, target

    @staticmethod
    def doc2line(doc, label=0):
        """Convert BoW representation of document in SVMlight format.
        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.line2doc`.

        Parameters
        ----------
        doc : list of (int, float)
            Document in BoW format.
        label : int, optional
            Document label (if provided).

        Returns
        -------
        str
            `doc` in SVMlight format.

        """
        pairs = ' '.join("%i:%s" % (termid + 1, termval) for termid, termval in doc)  # +1 to convert 0-base to 1-base
        return "%s %s\n" % (label, pairs)
