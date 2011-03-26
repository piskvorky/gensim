#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Blei's LDA-C format.
"""

from __future__ import with_statement

import logging

from gensim import interfaces, utils
from gensim.corpora import IndexedCorpus


class BleiCorpus(IndexedCorpus):
    """
    Corpus in Blei's LDA-C format.

    The corpus is represented as two files: one describing the documents, and another
    describing the mapping between words and their ids.

    Each document is one line::

      N fieldId1:fieldValue1 fieldId2:fieldValue2 ... fieldIdN:fieldValueN

    The vocabulary is a file with words, one word per line; word at line K has an
    implicit ``id=K``.
    """

    def __init__(self, fname, fnameVocab=None):
        """
        Initialize the corpus from a file.

        `fnameVocab` is the file with vocabulary; if not specified, it defaults to
        `fname.vocab`.
        """
        IndexedCorpus.__init__(self, fname)
        logging.info("loading corpus from %s" % fname)

        if fnameVocab is None:
            fnameVocab = fname + '.vocab'

        self.fname = fname
        words = [word.rstrip() for word in open(fnameVocab)]
        self.id2word = dict(enumerate(words))
        self.length = None


    def __iter__(self):
        """
        Iterate over the corpus, returning one sparse vector at a time.
        """
        length = 0
        for lineNo, line in enumerate(open(self.fname)):
            length += 1
            yield self.line2doc(line)
        self.length = length


    def line2doc(self, line):
        parts = line.split()
        if int(parts[0]) != len(parts) - 1:
            raise ValueError("invalid format in %s: %s" %
                             (self.fname, repr(line)))
        doc = [part.rsplit(':', 1) for part in parts[1:]]
        doc = [(int(p1), float(p2)) for p1, p2 in doc]
        return doc


    @staticmethod
    def saveCorpus(fname, corpus, id2word=None):
        """
        Save a corpus in the LDA-C format.

        There are actually two files saved: `fname` and `fname.vocab`, where
        `fname.vocab` is the vocabulary file.
        
        This function is automatically called by `BleiCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        if id2word is None:
            logging.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dictFromCorpus(corpus)
            numTerms = len(id2word)
        else:
            numTerms = 1 + max([-1] + id2word.keys())

        logging.info("storing corpus in Blei's LDA-C format: %s" % fname)
        with open(fname, 'w') as fout:
            offsets = []
            for doc in corpus:
                doc = list(doc)
                offsets.append(fout.tell())
                fout.write("%i %s\n" % (len(doc), ' '.join("%i:%s" % p for p in doc)))
            fout.close()

            # write out vocabulary, in a format compatible with Blei's topics.py script
            fnameVocab = fname + '.vocab'
            logging.info("saving vocabulary of %i words to %s" % (numTerms, fnameVocab))
            fout = open(fnameVocab, 'w')
            for featureId in xrange(numTerms):
                fout.write("%s\n" % utils.toUtf8(id2word.get(featureId, '---')))

        return offsets

    def docbyoffset(self, offset):
        """
        Return the document stored at file position `offset`.
        """
        with open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())
#endclass BleiCorpus

