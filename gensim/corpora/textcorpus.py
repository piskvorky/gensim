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

from gensim import interfaces, utils
from dictionary import Dictionary

logger = logging.getLogger('gensim.corpora.textcorpus')


def getstream(input):
    """
    If input is a filename (string), return `open(input)`.
    If input is a file-like object, reset it to the beginning with `input.seek(0)`.
    """
    assert input is not None
    if isinstance(input, basestring):
        # input was a filename: open as text file
        result = open(input)
    else:
        # input was a file-like object (BZ2, Gzip etc.); reset the stream to its beginning
        result = input
        result.seek(0)
    return result


class TextCorpus(interfaces.CorpusABC):
    """
    Helper class to simplify the pipeline of getting bag-of-words vectors (= a
    gensim corpus) from plain text.

    This is an abstract base class: override the `get_texts()` method to match
    your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object
    will be automatically initialized with a dictionary in `self.dictionary` and
    will support the `iter` corpus method. You must only provide a correct `get_texts`
    implementation.

    """
    def __init__(self, input=None):
        super(TextCorpus, self).__init__()
        self.input = input
        self.dictionary = Dictionary()
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
            yield self.dictionary.doc2bow(text, allow_update=False)


    def getstream(self):
        return getstream(self.input)


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
        length = 0
        for lineno, line in enumerate(getstream(self.input)):
            length += 1
            yield utils.tokenize(line, lowercase=True)
        self.length = length


    def __len__(self):
        return self.length # will throw if corpus not initialized
