#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in GibbsLda++ format of List-Of-Words."""

from __future__ import with_statement

import logging

from gensim import utils
from gensim.corpora import IndexedCorpus
from six import iterkeys
from six.moves import xrange, zip as izip


logger = logging.getLogger('gensim.corpora.lowcorpus')


def split_on_space(s):
    return [word for word in utils.to_unicode(s).strip().split(' ') if word]


class LowCorpus(IndexedCorpus):
    """List_Of_Words corpus handles input in GibbsLda++ format.

    Notes
    -----
    Quoting http://gibbslda.sourceforge.net/#3.2_Input_Data_Format::

        Both data for training/estimating the model and new data (i.e., previously
        unseen data) have the same format as follows:

        [M]
        [document1]
        [document2]
        ...
        [documentM]

        in which the first line is the total number for documents [M]. Each line
        after that is one document. [documenti] is the ith document of the dataset
        that consists of a list of Ni words/terms.

        [documenti] = [wordi1] [wordi2] ... [wordiNi]

        in which all [wordij] (i=1..M, j=1..Ni) are text strings and they are separated
        by the blank character.

        For example create file "lowcorpus_document.txt" and fill it with:

        3
        this is sparta
        for real
        lowcorpus example

    Examples
    --------
    >>> from gensim.corpora import lowcorpus
    >>> data = lowcorpus.LowCorpus("lowcorpus_document")
    >>> print data.fname, data.id2word
    {0: u'example', 1: u'for', 2: u'is', 3: u'lowcorpus', 4: u'real', 5: u'sparta', 6: u'this'}

    """
    def __init__(self, fname, id2word=None, line2words=split_on_space):
        """Initialize the corpus from a file.

        Parameters
        ----------
        fname : str
            File name.
        id2word : str, optional
            If provided, it is a dictionary mapping between word_ids (integers) and words (strings).
            Otherwise, the mapping is constructed from the documents.
        line2words : str, optional
            Function which converts lines into tokens. Defaults to simple splitting on spaces.

        Attributes
        ----------
        fname : str
        line2words : str
        use_wordids : bool
            If True, then return documents as (wordIndex, wordCount) 2-tuples, otherwise - (word, wordCount) 2-tuples.
        id2word : dict
        num_terms : int
            Length of word2id.

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
        """Turn line into document.

        Parameters
        ----------
        line : str
            Line from input file.

        Return
        ------
        list of tuples
            Construct a list of (word, wordFrequency) 2-tuples.

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
        """Iterate over the corpus, returning one bag-of-words vector at a time."""
        with utils.smart_open(self.fname) as fin:
            for lineno, line in enumerate(fin):
                if lineno > 0:  # ignore the first line = number of documents
                    yield self.line2doc(line)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save a corpus in the List-of-words format.

        Notes
        -----
        This function is automatically called by `LowCorpus.serialize`; don't
        call it directly, call `serialize` instead.

        Parameters
        ----------
        fname : str
            File name.
        corpus : list of (list of str)
            Data for creating dictionary.
        id2word : str, optional
            If provided, it is a dictionary mapping between word_ids (integers) and words (strings).
            Otherwise, the mapping is constructed from the documents.
        metadata : noidea, optional
            Wait, it has no effect, wut???

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
        """Return the document stored at file position `offset`.

        Parameters
        ----------

        offset : int

        
        """
        with utils.smart_open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())

    @property
    def id2word(self):
        return self._id2word

    @id2word.setter
    def id2word(self, val):
        self._id2word = val
        self.word2id = utils.revdict(val)
