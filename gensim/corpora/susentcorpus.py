#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Stanford Sentiment Treebank corpus
"""

from __future__ import with_statement

import logging
import os

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from six import iteritems, iterkeys
from six.moves import xrange, zip as izip
from collections import namedtuple, Counter

logger = logging.getLogger('gensim.corpora.susent')


class StanfordSentimentCorpus():
    """
    The Stanford Sentiment Treebank corpus, from
    http://nlp.Stanford.edu/sentiment/

    It's not too big, so read entirely into memory.
    """

    TRAIN = 1
    TEST = 2
    DEV = 3

    def __init__(self, dirname):
        """
        Initialize the corpus from a given directory, where
        http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
        has been expanded
        """
        logger.info("loading corpus from %s" % dirname)

        # many mangled chars in sentences (datasetSentences.txt)
        chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                             'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
        sentence_fixups = [(char.encode('utf-8').decode('latin1'), char) for char in chars_sst_mangled]
        # more junk, and the replace necessary for sentence-phrase consistency
        sentence_fixups.extend([
                ('Â', ''),
                ('\xa0', ' '),
                ('-LRB-', '('),
                ('-RRB-', ')'),
        ])
        # only this junk in phrases (dictionary.txt)
        phrase_fixups = [('\xa0', ' ')]

        # sentence_id and split are only positive for the full sentences
        Phrase = namedtuple('Phrase', 'id, text, sentiment, sentence_id, split')

        # read sentences to temp {sentence -> (id,split) dict, to correlate with dictionary.txt
        info_by_sentence = {}
        with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences, \
             open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)  # legend
            next(splits)     # legend
            for sentence_line, split_line in izip(sentences, splits):
                (id, text) = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                (id2, split_i) = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:    # discard duplicates
                    info_by_sentence[text] = (id, int(split_i))

        # read all phrase text
        self.phrases = [None] * 239232  # known size of phrases
        with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
            for line in phrase_lines:
                (text, id) = line.split('|')
                for junk, fix in phrase_fixups:
                    text = text.replace(junk, fix)
                self.phrases[int(id)] = text.rstrip()  # for 1st pass just string

        # add sentiment labels, correlate with sentences
        with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
            next(sentiments)  # legend
            for line in sentiments:
                (id, sentiment) = line.split('|')
                id = int(id)
                sentiment = float(sentiment)
                text = self.phrases[id]
                (sentence_id, split) = info_by_sentence.get(text, (-1, -1))
                self.phrases[id] = Phrase(id, text, sentiment, sentence_id, split)

        assert len([phrase for phrase in self.phrases if phrase.sentence_id > 0]) == len(info_by_sentence)  # all
        # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
        assert len([phrase for phrase in self.phrases if phrase.split is self.TRAIN]) == 8531  # 'train'
        assert len([phrase for phrase in self.phrases if phrase.split is self.TEST]) == 2210  # 'test'
        assert len([phrase for phrase in self.phrases if phrase.split is self.DEV]) == 1100  # 'dev'

        logger.info("loaded corpus with %i sentences and %i phrases from %s"
                    % (len(info_by_sentence), len(self.phrases), dirname))

    def split(self, n):
        """Return just the Phrase-tuples of the given split"""
        return [phrase for phrase in self.phrases if phrase.split == n]

    def __len__(self):
        return len(self.phrases)

    def __iter__(self):
        """
        Iterate over corpus, returning one phrase at a time, as LabeledSentence
        """
        for phrase in self.phrases:
            yield self[phrase.id]

    def __getitem__(self, phraseno):
        """Return phrase as LabeledSentence that Doc2Vec expects"""
        phrase = self.phrases[phraseno]
        words = phrase.text.split()
        return LabeledSentence(words, ['_*%s' % phrase.id])


# endclass StanfordSentimentCorpus
