#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Сorpus in Blei's LDA-C format."""

from __future__ import with_statement

from os import path
import logging

from gensim import utils
from gensim.corpora import IndexedCorpus
from six.moves import xrange


logger = logging.getLogger(__name__)


class BleiCorpus(IndexedCorpus):
    """Corpus in Blei's LDA-C format.

    The corpus is represented as two files: one describing the documents, and another
    describing the mapping between words and their ids.

    Each document is one line::

        N fieldId1:fieldValue1 fieldId2:fieldValue2 ... fieldIdN:fieldValueN


    The vocabulary is a file with words, one word per line; word at line K has an implicit `id=K`.

    """

    def __init__(self, fname, fname_vocab=None):
        """

        Parameters
        ----------
        fname : str
            File path to Serialized corpus.
        fname_vocab : str, optional
            Vocabulary file. If `fname_vocab` is None, searching for the vocab.txt or `fname_vocab`.vocab file.

        Raises
        ------
        IOError
            If vocabulary file doesn't exist.

        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s", fname)

        if fname_vocab is None:
            fname_base, _ = path.splitext(fname)
            fname_dir = path.dirname(fname)
            for fname_vocab in [
                        utils.smart_extension(fname, '.vocab'),
                        utils.smart_extension(fname, '/vocab.txt'),
                        utils.smart_extension(fname_base, '.vocab'),
                        utils.smart_extension(fname_dir, '/vocab.txt'),
                        ]:
                if path.exists(fname_vocab):
                    break
            else:
                raise IOError('BleiCorpus: could not find vocabulary file')

        self.fname = fname
        with utils.smart_open(fname_vocab) as fin:
            words = [utils.to_unicode(word).rstrip() for word in fin]
        self.id2word = dict(enumerate(words))

    def __iter__(self):
        """Iterate over the corpus, returning one sparse (BoW) vector at a time.

        Yields
        ------
        list of (int, float)
            Document's BoW representation.

        """
        lineno = -1
        with utils.smart_open(self.fname) as fin:
            for lineno, line in enumerate(fin):
                yield self.line2doc(line)
        self.length = lineno + 1

    def line2doc(self, line):
        """Convert line in Blei LDA-C format to document (BoW representation).

        Parameters
        ----------
        line : str
            Line in Blei's LDA-C format.

        Returns
        -------
        list of (int, float)
            Document's BoW representation.

        """
        parts = utils.to_unicode(line).split()
        if int(parts[0]) != len(parts) - 1:
            raise ValueError("invalid format in %s: %s" % (self.fname, repr(line)))
        doc = [part.rsplit(':', 1) for part in parts[1:]]
        doc = [(int(p1), float(p2)) for p1, p2 in doc]
        return doc

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save a corpus in the LDA-C format.

        Notes
        -----
        There are actually two files saved: `fname` and `fname.vocab`, where `fname.vocab` is the vocabulary file.

        Parameters
        ----------
        fname : str
            Path to output filename.
        corpus : iterable of iterable of (int, float)
            Input corpus
        id2word : dict of (str, str), optional
            Mapping id -> word for `corpus`.
        metadata : bool, optional
            THIS PARAMETER WILL BE IGNORED.

        Returns
        -------
        list of int
            Offsets for each line in file (in bytes).

        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)
            num_terms = len(id2word)
        else:
            num_terms = 1 + max([-1] + id2word.keys())

        logger.info("storing corpus in Blei's LDA-C format into %s", fname)
        with utils.smart_open(fname, 'wb') as fout:
            offsets = []
            for doc in corpus:
                doc = list(doc)
                offsets.append(fout.tell())
                parts = ["%i:%g" % p for p in doc if abs(p[1]) > 1e-7]
                fout.write(utils.to_utf8("%i %s\n" % (len(doc), ' '.join(parts))))

        # write out vocabulary, in a format compatible with Blei's topics.py script
        fname_vocab = utils.smart_extension(fname, '.vocab')
        logger.info("saving vocabulary of %i words to %s", num_terms, fname_vocab)
        with utils.smart_open(fname_vocab, 'wb') as fout:
            for featureid in xrange(num_terms):
                fout.write(utils.to_utf8("%s\n" % id2word.get(featureid, '---')))

        return offsets

    def docbyoffset(self, offset):
        """Get document corresponding to `offset`,
        offset can be given from :meth:`~gensim.corpora.bleicorpus.BleiCorpus.save_corpus`.

        Parameters
        ----------
        offset : int
            Position of the document in the file (in bytes).

        Returns
        -------
        list of (int, float)
            Document in BoW format.

        """
        with utils.smart_open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())
