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
import os
import re
import sys

from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS, strip_multiple_whitespaces
from gensim.utils import deaccent, simple_tokenize

logger = logging.getLogger('gensim.corpora.textcorpus')


def remove_stopwords(tokens, stopwords=STOPWORDS):
    return [token for token in tokens if token not in stopwords]


def remove_short(tokens, minsize=3):
    return [token for token in tokens if len(token) >= minsize]


def lower_to_unicode(text):
    return utils.to_unicode(text.lower(), 'utf8', 'strict')


class TextCorpus(interfaces.CorpusABC):
    """
    Helper class to simplify the pipeline of getting bag-of-words vectors (= a
    gensim corpus) from plain text.

    This is an abstract base class: override the `get_texts()` and `__len__()`
    methods to match your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object
    will be automatically initialized with a dictionary in `self.dictionary` and
    will support the `iter` corpus method. You have a few different ways of utilizing
    this class via subclassing or by construction with different preprocessing arguments.
    
    The `iter` method converts the lists of tokens produced by `get_texts` to BoW format
    using `Dictionary.doc2bow`. `get_texts` does the following:
    
    1.  Calls `getstream` to get a generator over the texts. It yields each document in
        turn from the underlying text file or files.
    2.  For each document from the stream, calls `preprocess_text` to produce a list of
        tokens; if metadata is enabled, it yields a 2-`tuple` with the document number as
        the second element.
        
    
    Preprocessing consists of 0+ `character_filters`, a `tokenizer`, and 0+ `token_filters`.
    
    The preprocessing consists of calling each filter in `character_filters` with the document 
    text; unicode is not guaranteed, and if desired, the first filter should convert to unicode. 
    The output of each character filter should be another string. The output from the final
    filter is fed to the `tokenizer`, which should split the string into a list of tokens (strings).
    Afterwards, the list of tokens is fed through each filter in `token_filters`. The final
    output returned from `preprocess_text` is the output from the final token filter.
    
    So to use this class, you can either pass in different preprocessing functions using the
    `character_filters`, `tokenizer`, and `token_filters` arguments, or you can subclass it.
    If subclassing: override `getstream` to take text from different input sources in different 
    formats. Overrride `preprocess_text` if you must provide different initial preprocessing,
    then call the `TextCorpus.preprocess_text` method to apply the normal preprocessing. You
    can also overrride `get_texts` in order to tag the documents (token lists) with different
    metadata.
    
    The default preprocessing consists of:
    
    1.  lowercase and convert to unicode; assumes utf8 encoding
    2.  deaccent (asciifolding)
    3.  collapse multiple whitespaces into a single one
    4.  tokenize by splitting on whitespace
    5.  remove words less than 3 characters long
    6.  remove stopwords; see `gensim.parsing.preprocessing` for the list of stopwords

    """
    def __init__(self, input=None, metadata=False, character_filters=None, tokenizer=None,
                 token_filters=None):
        super(TextCorpus, self).__init__()
        self.input = input
        self.dictionary = Dictionary()
        self.metadata = metadata

        self.character_filters = character_filters
        if self.character_filters is None:
            self.character_filters = [lower_to_unicode, deaccent, strip_multiple_whitespaces]

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = simple_tokenize

        self.token_filters = token_filters
        if self.token_filters is None:
            self.token_filters = [remove_short, remove_stopwords]

        if input is not None:
            self.dictionary.add_documents(self.get_texts())
        else:
            logger.warning("No input document stream provided; assuming "
                           "dictionary will be initialized some other way.")

    def __iter__(self):
        """The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        if self.metadata:
            for text, metadata in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False), metadata
        else:
            for text in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        """Yield documents from the underlying plain text collection (of one or more files).
        Each item yielded from this method will be considered a document by subsequent
        preprocessing methods.
        """
        with utils.file_or_filename(self.input) as f:
            for line in f:
                yield line

    def preprocess_text(self, text):
        """Apply preprocessing to a single text document. This should perform tokenization
        in addition to any other desired preprocessing steps.
        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

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
        lines = self.getstream()
        if self.metadata:
            for lineno, line in enumerate(lines):
                yield self.preprocess_text(line), (lineno,)
        else:
            for line in lines:
                yield self.preprocess_text(line)

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.get_texts())
        return self.length
# endclass TextCorpus


class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory,
    where each file is interpreted as a plain text document.
    """

    def __init__(self, input, metadata=False, min_depth=0, max_depth=None, pattern=None,
                 exclude_pattern=None, **kwargs):
        self._min_depth = min_depth
        self._max_depth = sys.maxsize if max_depth is None else max_depth
        self.pattern = pattern
        self.exclude_pattern = exclude_pattern
        super(TextDirectoryCorpus, self).__init__(input, metadata, **kwargs)

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        self._pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def exclude_pattern(self):
        return self._exclude_pattern

    @exclude_pattern.setter
    def exclude_pattern(self, pattern):
        self._exclude_pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def min_depth(self):
        return self._min_depth

    @min_depth.setter
    def min_depth(self, min_depth):
        self._min_depth = min_depth
        self.length = None

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self._max_depth = max_depth
        self.length = None

    def iter_filepaths(self):
        """Lazily yield paths to each file in the directory structure within the specified
        range of depths. If a filename pattern to match was given, further filter to only
        those filenames that match.
        """
        for depth, dirpath, dirnames, filenames in walk(self.input):
            if self.min_depth <= depth <= self.max_depth:
                if self.pattern is not None:
                    filenames = (n for n in filenames if self.pattern.match(n) is not None)
                if self.exclude_pattern is not None:
                    filenames = (n for n in filenames if self.exclude_pattern.match(n) is None)

                for name in filenames:
                    yield os.path.join(dirpath, name)

    def getstream(self):
        for path in self.iter_filepaths():
            with utils.smart_open(path) as f:
                doc_content = f.read()
            yield doc_content

    def __len__(self):
        if self.length is None:
            # cache the corpus length
            self.length = sum(1 for _ in self.iter_filepaths())
        return self.length
# endclass TextDirectoryCorpus


def walk(top, topdown=True, onerror=None, followlinks=False, depth=0):
    """This is a mostly copied version of `os.walk` from the Python 2 source code.
    The only difference is that it returns the depth in the directory tree structure
    at which each yield is taking place.
    """
    islink, join, isdir = os.path.islink, os.path.join, os.path.isdir

    try:
        # Should be O(1) since it's probably just reading your filesystem journal
        names = os.listdir(top)
    except OSError as err:
        if onerror is not None:
            onerror(err)
        return

    dirs, nondirs = [], []

    # O(n) where n = number of files in the directory
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield depth, top, dirs, nondirs

    # Again O(n), where n = number of directories in the directory
    for name in dirs:
        new_path = join(top, name)
        if followlinks or not islink(new_path):

            # Generator so besides the recursive `walk()` call, no additional cost here.
            for x in walk(new_path, topdown, onerror, followlinks, depth + 1):
                yield x
    if not topdown:
        yield depth, top, dirs, nondirs
