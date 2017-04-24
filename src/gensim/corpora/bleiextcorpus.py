#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended version of BleiCorpus, has a list-based index
of all documents for fast retrieval of specific document and length calculation
"""

import logging, pickle

from gensim.corpora import BleiCorpus
from gensim import utils


class BleiExtCorpus(BleiCorpus):
    """
    Corpus in Blei's LDA-C format, extended with index
    """
    def __init__(self, fname, fnameVocab = None):
       super(BleiExtCorpus, self).__init__(fname, fnameVocab)
       self.index = pickle.load(open(self.fname + '.index', 'r'))

    def __len__(self):
        return len (self.index)

    def __getitem__(self, id):
        f = open (self.fname, 'r')
        f.seek(self.index[id])
        line = f.readline()
        parts = line.split()
        if int(parts[0]) != len(parts) - 1:
            raise ValueError("invalid format at line %i in %s" % (lineNo, self.fname))
        doc = [part.rsplit(':', 1) for part in parts[1:]]
        doc = [(int(p1), float(p2)) for p1, p2 in doc]
        return doc

    @staticmethod
    def saveCorpus(fname, corpus, id2word = None):
        """
        Save a corpus in the Matrix Market format.

        There are actually three files saved:
        * `fname`: the corpus itself.
        * `fname.vocab`: vocabulary file.
        * `fname.index`: index with pointers to documents.
        """
        if id2word is None:
            logging.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dictFromCorpus(corpus)
            numTerms = len(id2word)
        else:
            numTerms = 1 + max([-1] + id2word.keys())

        index = []
        offset = 0

        logging.info("storing corpus in Blei's LDA-C format: %s" % fname)
        fout = open(fname, 'w')
        for doc in corpus:
            doc = list(doc)
            line = "%i %s\n" % (len(doc), ' '.join("%i:%s" % p for p in doc))
            fout.write(line)
            index.append(offset)
            offset += len(line)
        fout.close()

        # write out vocabulary, in a format compatible with Blei's topics.py script
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary of %i words to %s" % (numTerms, fnameVocab))
        fout = open(fnameVocab, 'w')
        for featureId in xrange(numTerms):
            fout.write("%s\n" % utils.toUtf8(id2word.get(featureId, '---')))
        fout.close()

        # write out index
        pickle.dump(index, open(fname + '.index', 'w'))
