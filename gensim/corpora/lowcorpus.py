#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in GibbsLda++ format of List-Of-Words.
"""

from __future__ import with_statement

import logging
import itertools

from gensim import interfaces, utils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger('gensim.corpora.lowcorpus')


def split_on_space(s):
    return [word for word in s.strip().split(' ') if word]


class LowCorpus(IndexedCorpus):
    """
    List_Of_Words corpus handles input in GibbsLda++ format.

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
    """
    def __init__(self, fname, id2word=None, line2words=split_on_space):
        """
        Initialize the corpus from a file.

        `id2word` and `line2words` are optional parameters.
        If provided, `id2word` is a dictionary mapping between word_ids (integers)
        and words (strings). If not provided, the mapping is constructed from
        the documents.

        `line2words` is a function which converts lines into tokens. Defaults to
        simple splitting on spaces.
        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s" % fname)

        self.fname = fname # input file, see class doc for format
        self.line2words = line2words # how to translate lines into words (simply split on space by default)
        self.num_docs = int(open(fname).readline()) # the first line in input data is the number of documents (integer). throws exception on bad input.

        if not id2word:
            # build a list of all word types in the corpus (distinct words)
            logger.info("extracting vocabulary from the corpus")
            all_terms = set()
            self.use_wordids = False # return documents as (word, wordCount) 2-tuples
            for doc in self:
                all_terms.update(word for word, wordCnt in doc)
            all_terms = sorted(all_terms) # sort the list of all words; rank in that list = word's integer id
            self.id2word = dict(enumerate(all_terms)) # build a mapping of word id(int) -> word (string)
        else:
            logger.info("using provided word mapping (%i ids)" % len(id2word))
            self.id2word = id2word
        self.word2id = dict((v, k) for k, v in self.id2word.items())
        self.num_terms = len(self.word2id)
        self.use_wordids = True # return documents as (wordIndex, wordCount) 2-tuples

        logger.info("loaded corpus with %i documents and %i terms from %s" %
                     (self.num_docs, self.num_terms, fname))


    def __len__(self):
        return self.num_docs


    def line2doc(self, line):
        words = self.line2words(line)

        if self.use_wordids:
            # get all distinct terms in this document, ignore unknown words
            uniq_words = set(words).intersection(self.word2id.keys())

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
            doc = list(zip(map(self.word2id.get, use_words), map(words.count, use_words))) # using list.count is suboptimal but speed of this whole function is irrelevant
        else:
            uniq_words = set(words)
            # construct a list of (word, wordFrequency) 2-tuples
            doc = list(zip(uniq_words, map(words.count, uniq_words))) # using list.count is suboptimal but that's irrelevant at this point

        # return the document, then forget it and move on to the next one
        # note that this way, only one doc is stored in memory at a time, not the whole corpus
        return doc


    def __iter__(self):
        """
        Iterate over the corpus, returning one bag-of-words vector at a time.
        """
        for lineno, line in enumerate(open(self.fname)):
            if lineno > 0: # ignore the first line = number of documents
                yield self.line2doc(line)


    @staticmethod
    def save_corpus(fname, corpus, id2word=None):
        """
        Save a corpus in the List-of-words format.

        This function is automatically called by `LowCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)

        logger.info("storing corpus in List-Of-Words format: %s" % fname)
        truncated = 0
        offsets = []
        with open(fname, 'w') as fout:
            fout.write('%i\n' % len(corpus))
            for doc in corpus:
                words = []
                for wordid, value in doc:
                    if abs(int(value) - value) > 1e-6:
                        truncated += 1
                    words.extend([str(id2word[wordid])] * int(value))
                offsets.append(fout.tell())
                fout.write('%s\n' % ' '.join(words))

        if truncated:
            logger.warning("List-of-words format can only save vectors with "
                            "integer elements; %i float entries were truncated to integer value" %
                            truncated)
        return offsets


    def docbyoffset(self, offset):
        """
        Return the document stored at file position `offset`.
        """
        with open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())
#endclass LowCorpus

