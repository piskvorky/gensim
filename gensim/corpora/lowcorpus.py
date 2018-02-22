#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in `GibbsLda++ format <http://gibbslda.sourceforge.net/>`_."""

from __future__ import with_statement

import logging

from gensim import utils
from gensim.corpora import IndexedCorpus
from six import iterkeys
from six.moves import xrange, zip as izip


logger = logging.getLogger(__name__)


def split_on_space(s):
    """Split line by spaces, used in :class:`gensim.corpora.lowcorpus.LowCorpus`.

    Parameters
    ----------
    s : str
        Some line.

    Returns
    -------
    list of str
        List of tokens from `s`.

    """
    return [word for word in utils.to_unicode(s).strip().split(' ') if word]


class LowCorpus(IndexedCorpus):
    """Corpus handles input in `GibbsLda++ format <http://gibbslda.sourceforge.net/>`_.

    **Format description**

    Both data for training/estimating the model and new data (i.e., previously unseen data) have the same format
    as follows ::

        [M]
        [document1]
        [document2]
        ...
        [documentM]

    in which the first line is the total number for documents [M]. Each line after that is one document.
    [documenti] is the ith document of the dataset that consists of a list of Ni words/terms ::

        [documenti] = [wordi1] [wordi2] ... [wordiNi]

    in which all [wordij] (i=1..M, j=1..Ni) are text strings and they are separated by the blank character.

    Examples
    --------
    >>> from gensim.test.utils import datapath, get_tmpfile, common_texts
    >>> from gensim.corpora import LowCorpus
    >>> from gensim.corpora import Dictionary
    >>>
    >>> # Prepare needed data
    >>> dictionary = Dictionary(common_texts)
    >>> corpus = [dictionary.doc2bow(doc) for doc in common_texts]
    >>>
    >>> # Write corpus in GibbsLda++ format to disk
    >>> output_fname = get_tmpfile("corpus.low")
    >>> LowCorpus.serialize(output_fname, corpus, dictionary)
    >>>
    >>> # Read corpus
    >>> loaded_corpus = LowCorpus(output_fname)

    """
    def __init__(self, fname, id2word=None, line2words=split_on_space):
        """

        Parameters
        ----------
        fname : str
            Path to file in GibbsLda++ format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `fname`.
        line2words : callable, optional
            Function which converts lines(str) into tokens(list of str),
            using :func:`~gensim.corpora.lowcorpus.split_on_space` as default.

        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s", fname)

        self.fname = fname  # input file, see class doc for format
        self.line2words = line2words  # how to translate lines into words (simply split on space by default)
        self.num_docs = self._calculate_num_docs()

        if not id2word:
            # build a list of all word types in the corpus (distinct words)
            logger.info("extracting vocabulary from the corpus")
            all_terms = set()
            self.use_wordids = False  # return documents as (word, wordCount) 2-tuples
            for doc in self:
                all_terms.update(word for word, wordCnt in doc)
            all_terms = sorted(all_terms)  # sort the list of all words; rank in that list = word's integer id
            # build a mapping of word id(int) -> word (string)
            self.id2word = dict(izip(xrange(len(all_terms)), all_terms))
        else:
            logger.info("using provided word mapping (%i ids)", len(id2word))
            self.id2word = id2word
        self.num_terms = len(self.word2id)
        self.use_wordids = True  # return documents as (wordIndex, wordCount) 2-tuples

        logger.info(
            "loaded corpus with %i documents and %i terms from %s",
            self.num_docs, self.num_terms, fname
        )

    def _calculate_num_docs(self):
        """Get number of documents in file.

        Returns
        -------
        int
            Number of documents.

        """
        # the first line in input data is the number of documents (integer). throws exception on bad input.
        with utils.smart_open(self.fname) as fin:
            try:
                result = int(next(fin))
            except StopIteration:
                result = 0

        return result

    def __len__(self):
        return self.num_docs

    def line2doc(self, line):
        """Covert line into document in BoW format.

        Parameters
        ----------
        line : str
            Line from input file.

        Returns
        -------
        list of (int, int)
            Document in BoW format

        """
        words = self.line2words(line)

        if self.use_wordids:
            # get all distinct terms in this document, ignore unknown words
            uniq_words = set(words).intersection(iterkeys(self.word2id))

            # the following creates a unique list of words *in the same order*
            # as they were in the input. when iterating over the documents,
            # the (word, count) pairs will appear in the same order as they
            # were in the input (bar duplicates), which looks better.
            # if this was not needed, we might as well have used useWords = set(words)
            use_words, marker = [], set()
            for word in words:
                if (word in uniq_words) and (word not in marker):
                    use_words.append(word)
                    marker.add(word)
            # construct a list of (wordIndex, wordFrequency) 2-tuples
            doc = [(self.word2id.get(w), words.count(w)) for w in use_words]
        else:
            uniq_words = set(words)
            # construct a list of (word, wordFrequency) 2-tuples
            doc = [(w, words.count(w)) for w in uniq_words]

        # return the document, then forget it and move on to the next one
        # note that this way, only one doc is stored in memory at a time, not the whole corpus
        return doc

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format.

        """
        with utils.smart_open(self.fname) as fin:
            for lineno, line in enumerate(fin):
                if lineno > 0:  # ignore the first line = number of documents
                    yield self.line2doc(line)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save a corpus in the GibbsLda++ format.

        Warnings
        --------
        This function is automatically called by :meth:`gensim.corpora.lowcorpus.LowCorpus.serialize`,
        don't call it directly, call :meth:`gensim.corpora.lowcorpus.LowCorpus.serialize` instead.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `corpus`.
        metadata : bool, optional
            THIS PARAMETER WILL BE IGNORED.

        Return
        ------
        list of int
            List of offsets in resulting file for each document (in bytes),
            can be used for :meth:`~gensim.corpora.lowcorpus.LowCorpus.docbyoffset`

        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)

        logger.info("storing corpus in List-Of-Words format into %s" % fname)
        truncated = 0
        offsets = []
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8('%i\n' % len(corpus)))
            for doc in corpus:
                words = []
                for wordid, value in doc:
                    if abs(int(value) - value) > 1e-6:
                        truncated += 1
                    words.extend([utils.to_unicode(id2word[wordid])] * int(value))
                offsets.append(fout.tell())
                fout.write(utils.to_utf8('%s\n' % ' '.join(words)))

        if truncated:
            logger.warning(
                "List-of-words format can only save vectors with integer elements; "
                "%i float entries were truncated to integer value", truncated
            )
        return offsets

    def docbyoffset(self, offset):
        """Get the document stored in file by `offset` position.

        Parameters
        ----------
        offset : int
            Offset (in bytes) to begin of document.

        Returns
        -------
        list of (int, int)
            Document in BoW format.

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.corpora import LowCorpus
        >>>
        >>> data = LowCorpus(datapath("testcorpus.low"))
        >>> data.docbyoffset(1)  # end of first line
        []
        >>> data.docbyoffset(2)  # start of second line
        [(0, 1), (3, 1), (4, 1)]

        """
        with utils.smart_open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())

    @property
    def id2word(self):
        """Get mapping between words and their ids."""
        return self._id2word

    @id2word.setter
    def id2word(self, val):
        self._id2word = val
        self.word2id = utils.revdict(val)
