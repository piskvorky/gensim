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

    def sample_texts(self, n, seed=None, length=None):
        """
        Yields n random documents from the corpus without replacement.

        Given the number of remaining documents in corpus, we need to choose n elements.
        The probability for current element to be chosen is n/remaining.
        If we choose it, we just decreese the n and move to the next element.
        Computing corpus length may be a costly operation so you can use optional paramter
        length instead.

        Args:
            n (int): number of documents we want to sample.
            seed (int|None): if specified, use it as a seed for local random generator.
            length (int|None): if specified, use it as guess of corpus length.

        Yeilds:
            list[str]: document represented as list of tokens. See get_texts method.

        Raises:
            ValueError: then n is invalid or length was set incorrectly.
        """
        random_generator = None
        if seed is None:
            random_generator = random
        else:
            random_generator = random.Random(seed)

        if length is None:
            length = len(self)

        if not n <= length:
            raise ValueError("n is larger than length of corpus.")
        if not 0 <= n:
            raise ValueError("Negative sample size.")

        for i, sample in enumerate(self.get_texts()):
            if i == length:
                break
            remaining_in_corpus = length - i
            chance = random_generator.randint(1, remaining_in_corpus)
            if chance <= n:
                n -= 1
                yield sample

        if n != 0:
            # This means that length was set to be smaller than nuber of items in stream.
            raise ValueError("length smaller than number of documents in stream")

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.get_texts())
        return self.length

# endclass TextCorpus
