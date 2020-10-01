#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains the classes of BrownCorpus and TaggedBrownCorpus."""

import os
from gensim import utils
from .taggeddocument import TaggedDocument


class BrownCorpus(object):
    def __init__(self, dirname):
        """Iterate over sentences from the `Brown corpus <https://en.wikipedia.org/wiki/Brown_Corpus>`_
        (part of `NLTK data <https://www.nltk.org/data.html>`_).

        """
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            with utils.open(fname, 'rb') as fin:
                for line in fin:
                    line = utils.to_unicode(line)
                    # each file line is a single sentence in the Brown corpus
                    # each token is WORD/POS_TAG
                    token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                    # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                    words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                    if not words:  # don't bother sending out empty sentences
                        continue
                    yield words


class TaggedBrownCorpus(object):
    def __init__(self, dirname):
        """Reader for the `Brown corpus (part of NLTK data) <http://www.nltk.org/book/ch02.html#tab-brown-sources>`_.

        Parameters
        ----------
        dirname : str
            Path to folder with Brown corpus.

        """
        self.dirname = dirname

    def __iter__(self):
        """Iterate through the corpus.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source`.

        """
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            with utils.open(fname, 'rb') as fin:
                for item_no, line in enumerate(fin):
                    line = utils.to_unicode(line)
                    # each file line is a single document in the Brown corpus
                    # each token is WORD/POS_TAG
                    token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                    # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                    words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                    if not words:  # don't bother sending out empty documents
                        continue
                    yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])
