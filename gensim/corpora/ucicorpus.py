#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Jonathan Esterhazy <jonathan.esterhazy at gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
University of California, Irvine (UCI) Bag-of-Words format.

http://archive.ics.uci.edu/ml/datasets/Bag+of+Words
"""

from __future__ import with_statement

import logging
from collections import defaultdict

from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
from six import iteritems, string_types
from six.moves import xrange


logger = logging.getLogger('gensim.corpora.ucicorpus')


class UciReader(MmReader):
    def __init__(self, input):
        """
        Initialize the reader.

        The `input` parameter refers to a file on the local filesystem,
        which is expected to be in the UCI Bag-of-Words format.
        """

        logger.info('Initializing corpus reader from %s' % input)

        self.input = input

        with utils.smart_open(self.input) as fin:
            self.num_docs = self.num_terms = self.num_nnz = 0
            try:
                self.num_docs = int(next(fin).strip())
                self.num_terms = int(next(fin).strip())
                self.num_nnz = int(next(fin).strip())
            except StopIteration:
                pass

        logger.info('accepted corpus with %i documents, %i features, %i non-zero entries' %
            (self.num_docs, self.num_terms, self.num_nnz))

    def skip_headers(self, input_file):
        for lineno, _ in enumerate(input_file):
            if lineno == 2:
                break

# endclass UciReader


class UciWriter(MmWriter):
    """
    Store a corpus in UCI Bag-of-Words format.

    This corpus format is identical to MM format, except for
    different file headers. There is no format line, and the first
    three lines of the file contain number_docs, num_terms, and num_nnz,
    one value per line.

    This implementation is based on matutils.MmWriter, and works the same way.

    """
    MAX_HEADER_LENGTH = 20  # reserve 20 bytes per header value
    FAKE_HEADER = utils.to_utf8(' ' * MAX_HEADER_LENGTH + '\n')

    def write_headers(self):
        """
        Write blank header lines. Will be updated later, once corpus stats are known.
        """
        for _ in range(3):
            self.fout.write(self.FAKE_HEADER)

        self.last_docno = -1
        self.headers_written = True

    def update_headers(self, num_docs, num_terms, num_nnz):
        """
        Update headers with actual values.
        """
        offset = 0
        values = [utils.to_utf8(str(n)) for n in [num_docs, num_terms, num_nnz]]

        for value in values:
            if len(value) > len(self.FAKE_HEADER):
                raise ValueError('Invalid header: value too large!')
            self.fout.seek(offset)
            self.fout.write(value)
            offset += len(self.FAKE_HEADER)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False):
        writer = UciWriter(fname)
        writer.write_headers()

        num_terms, num_nnz = 0, 0
        docno, poslast = -1, -1
        offsets = []
        for docno, bow in enumerate(corpus):
            if docno % progress_cnt == 0:
                logger.info("PROGRESS: saving document #%i" % docno)
            if index:
                posnow = writer.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow

            vector = [(x, int(y)) for (x, y) in bow if int(y) != 0] # integer count, not floating weights
            max_id, veclen = writer.write_vector(docno, vector)
            num_terms = max(num_terms, 1 + max_id)
            num_nnz += veclen
        num_docs = docno + 1

        if num_docs * num_terms != 0:
            logger.info("saved %ix%i matrix, density=%.3f%% (%i/%i)" %
                         (num_docs, num_terms,
                          100.0 * num_nnz / (num_docs * num_terms),
                          num_nnz,
                          num_docs * num_terms))

        # now write proper headers, by seeking and overwriting the spaces written earlier
        writer.update_headers(num_docs, num_terms, num_nnz)

        writer.close()
        if index:
            return offsets

# endclass UciWriter


class UciCorpus(UciReader, IndexedCorpus):
    """
    Corpus in the UCI bag-of-words format.
    """
    def __init__(self, fname, fname_vocab=None):
        IndexedCorpus.__init__(self, fname)
        UciReader.__init__(self, fname)

        if fname_vocab is None:
            fname_vocab = fname + '.vocab'

        self.fname = fname
        with utils.smart_open(fname_vocab) as fin:
            words = [word.strip() for word in fin]
        self.id2word = dict(enumerate(words))

        self.transposed = True

    def __iter__(self):
        """
        Interpret a matrix in UCI bag-of-words format as a streamed gensim corpus
        (yielding one document at a time).
        """
        for docId, doc in super(UciCorpus, self).__iter__():
            yield doc # get rid of docId, return the sparse vector only

    def create_dictionary(self):
        """
        Utility method to generate gensim-style Dictionary directly from
        the corpus and vocabulary data.
        """
        dictionary = Dictionary()

        # replace dfs with defaultdict to avoid downstream KeyErrors
        # uci vocabularies may contain terms that are not used in the document data
        dictionary.dfs = defaultdict(int)

        dictionary.id2token = self.id2word
        dictionary.token2id = dict((v, k) for k, v in iteritems(self.id2word))

        dictionary.num_docs = self.num_docs
        dictionary.num_nnz = self.num_nnz

        for docno, doc in enumerate(self):
            if docno % 10000 == 0:
                logger.info('PROGRESS: processing document %i of %i' % (docno, self.num_docs))

            for word, count in doc:
                dictionary.dfs[word] += 1
                dictionary.num_pos += count

        return dictionary

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=10000, metadata=False):
        """
        Save a corpus in the UCI Bag-of-Words format.

        There are actually two files saved: `fname` and `fname.vocab`, where
        `fname.vocab` is the vocabulary file.

        This function is automatically called by `UciCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)
            num_terms = len(id2word)
        else:
            num_terms = 1 + max([-1] + id2word.keys())

        # write out vocabulary
        fname_vocab = fname + '.vocab'
        logger.info("saving vocabulary of %i words to %s" % (num_terms, fname_vocab))
        with utils.smart_open(fname_vocab, 'wb') as fout:
            for featureid in xrange(num_terms):
                fout.write(utils.to_utf8("%s\n" % id2word.get(featureid, '---')))

        logger.info("storing corpus in UCI Bag-of-Words format: %s" % fname)

        return UciWriter.write_corpus(fname, corpus, index=True, progress_cnt=progress_cnt)

# endclass UciCorpus
