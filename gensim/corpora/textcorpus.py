#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Text corpora usually reside on disk, as text files in one format or another
In a common scenario, we need to build a dictionary (a `word->integer id`
mapping), which is then used to construct sparse bag-of-word vectors
(= sequences of `(word_id, word_weight)` 2-tuples).

This module provides some code scaffolding to simplify this pipeline. For
example, given a corpus where each document is a separate line in file on disk,
you would override the `TextCorpus.get_texts` method to read one line=document
at a time, process it (lowercase, tokenize, whatever) and yield it as a sequence
of words.

Overriding `get_texts` is enough; you can then initialize the corpus with e.g.
`MyTextCorpus(bz2.BZ2File('mycorpus.txt.bz2'))` and it will behave correctly like a
corpus of sparse vectors. The `__iter__` methods is automatically set up, and
dictionary is automatically populated with all `word->id` mappings.

The resulting object can be used as input to all gensim models (TFIDF, LSI, ...),
serialized with any format (Matrix Market, SvmLight, Blei's LDA-C format etc).

See the `gensim.test.test_miislita.CorpusMiislita` class for a simple example.
"""


from __future__ import with_statement

import logging
import random

from gensim import interfaces, utils
from six import string_types
from gensim.corpora.dictionary import Dictionary

logger = logging.getLogger('gensim.corpora.textcorpus')


class TextCorpus(interfaces.CorpusABC):
    """
    Helper class to simplify the pipeline of getting bag-of-words vectors (= a
    gensim corpus) from plain text.

    This is an abstract base class: override the `get_texts()` and `__len__()`
    methods to match your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object
    will be automatically initialized with a dictionary in `self.dictionary` and
    will support the `iter` corpus method. You must only provide a correct `get_texts`
    implementation.

    """
    def __init__(self, input=None):
        super(TextCorpus, self).__init__()
        self.input = input
        self.dictionary = Dictionary()
        self.metadata = False
        if input is not None:
            self.dictionary.add_documents(self.get_texts())
        else:
            logger.warning("No input document stream provided; assuming "
                           "dictionary will be initialized some other way.")

    def __iter__(self):
        """
        The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        for text in self.get_texts():
            if self.metadata:
                yield self.dictionary.doc2bow(text[0], allow_update=False), text[1]
            else:
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        return utils.file_or_filename(self.input)

    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.

        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no further
        preprocessing of the words coming out of this function.
        """
        # Instead of raising NotImplementedError, let's provide a sample implementation:
        # assume documents are lines in a single file (one document per line).
        # Yield each document as a list of lowercase tokens, via `utils.tokenize`.
        with self.getstream() as lines:
            for lineno, line in enumerate(lines):
                if self.metadata:
                    yield utils.tokenize(line, lowercase=True), (lineno,)
                else:
                    yield utils.tokenize(line, lowercase=True)

    def sample_texts(self, n):
        """
        Yield n random texts from the corpus without replacement.

        Given the the number of remaingin elements in stream is remaining and we need
        to choose n elements, the probability for current element to be chosen is n/remaining.
        If we choose it, we just decreese the n and move to the next element.
        """
        length = len(self)
        if not n <= length:
            raise ValueError("sample larger than population")

        if not 0 <= n:
            raise ValueError("negative sample size")

        for i, sample in enumerate(self.get_texts()):
            remaining_in_stream = length - i
            chance = random.randint(1, remaining_in_stream)
            if chance <= n:
                n -= 1
                yield sample

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.get_texts())
        return self.length

# endclass TextCorpus
