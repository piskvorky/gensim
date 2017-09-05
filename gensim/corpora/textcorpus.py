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
import random
import re
import sys

from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS, RE_WHITESPACE
from gensim.utils import deaccent, simple_tokenize

logger = logging.getLogger(__name__)


def remove_stopwords(tokens, stopwords=STOPWORDS):
    """Remove stopwords using list from `gensim.parsing.preprocessing.STOPWORDS."""
    return [token for token in tokens if token not in stopwords]


def remove_short(tokens, minsize=3):
    """Remove tokens smaller than `minsize` chars, which is 3 by default."""
    return [token for token in tokens if len(token) >= minsize]


def lower_to_unicode(text, encoding='utf8', errors='strict'):
    """Lowercase `text` and convert to unicode."""
    return utils.to_unicode(text.lower(), encoding, errors)


def strip_multiple_whitespaces(s):
    """Collapse multiple whitespace characters into a single space."""
    return RE_WHITESPACE.sub(" ", s)


class TextCorpus(interfaces.CorpusABC):
    """Helper class to simplify the pipeline of getting bag-of-words vectors (= a
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
    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None,
                 tokenizer=None, token_filters=None):
        """
        Args:
            input (str): path to top-level directory to traverse for corpus documents.
            dictionary (Dictionary): if a dictionary is provided, it will not be updated
                with the given corpus on initialization. If none is provided, a new dictionary
                will be built for the given corpus. If no corpus is given, the dictionary will
                remain uninitialized.
            metadata (bool): True to yield metadata with each document, else False (default).
            character_filters (iterable of callable): each will be applied to the text of each
                document in order, and should return a single string with the modified text.
                For Python 2, the original text will not be unicode, so it may be useful to
                convert to unicode as the first character filter. The default character filters
                lowercase, convert to unicode (strict utf8), perform ASCII-folding, then collapse
                multiple whitespaces.
            tokenizer (callable): takes as input the document text, preprocessed by all filters
                in `character_filters`; should return an iterable of tokens (strings).
            token_filters (iterable of callable): each will be applied to the iterable of tokens
                in order, and should return another iterable of tokens. These filters can add,
                remove, or replace tokens, or do nothing at all. The default token filters
                remove tokens less than 3 characters long and remove stopwords using the list
                in `gensim.parsing.preprocessing.STOPWORDS`.
        """
        self.input = input
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

        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)

    def init_dictionary(self, dictionary):
        """If `dictionary` is None, initialize to an empty Dictionary, and then if there
        is an `input` for the corpus, add all documents from that `input`. If the
        `dictionary` is already initialized, simply set it as the corpus's `dictionary`.
        """
        self.dictionary = dictionary if dictionary is not None else Dictionary()
        if self.input is not None:
            if dictionary is None:
                logger.info("Initializing dictionary")
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logger.info("Input stream provided but dictionary already initialized")
        else:
            logger.warning("No input document stream provided; assuming dictionary will be initialized some other way.")

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
        num_texts = 0
        with utils.file_or_filename(self.input) as f:
            for line in f:
                yield line
                num_texts += 1

        self.length = num_texts

    def preprocess_text(self, text):
        """Apply preprocessing to a single text document. This should perform tokenization
        in addition to any other desired preprocessing steps.

        Args:
            text (str): document text read from plain-text file.

        Returns:
            iterable of str: tokens produced from `text` as a result of preprocessing.
        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    def step_through_preprocess(self, text):
        """Yield tuples of functions and their output for each stage of preprocessing.
        This is useful for debugging issues with the corpus preprocessing pipeline.
        """
        for character_filter in self.character_filters:
            text = character_filter(text)
            yield (character_filter, text)

        tokens = self.tokenizer(text)
        yield (self.tokenizer, tokens)

        for token_filter in self.token_filters:
            tokens = token_filter(tokens)
            yield (token_filter, tokens)

    def get_texts(self):
        """Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.
        Each document will be fed through `preprocess_text`. That method should be
        overridden to provide different preprocessing steps. This method will need
        to be overridden if the metadata you'd like to yield differs from the line
        number.

        Returns:
            generator of lists of tokens (strings); each list corresponds to a preprocessed
            document from the corpus `input`.
        """
        lines = self.getstream()
        if self.metadata:
            for lineno, line in enumerate(lines):
                yield self.preprocess_text(line), (lineno,)
        else:
            for line in lines:
                yield self.preprocess_text(line)

    def sample_texts(self, n, seed=None, length=None):
        """Yield n random documents from the corpus without replacement.

        Given the number of remaining documents in a corpus, we need to choose n elements.
        The probability for the current element to be chosen is n/remaining.
        If we choose it, we just decrease the n and move to the next element.
        Computing the corpus length may be a costly operation so you can use the optional
        parameter `length` instead.

        Args:
            n (int): number of documents we want to sample.
            seed (int|None): if specified, use it as a seed for local random generator.
            length (int|None): if specified, use it as a guess of corpus length.
                It must be positive and not greater than actual corpus length.

        Yields:
            list[str]: document represented as a list of tokens. See get_texts method.

        Raises:
            ValueError: when n is invalid or length was set incorrectly.
        """
        random_generator = random if seed is None else random.Random(seed)
        if length is None:
            length = len(self)

        if not n <= length:
            raise ValueError("n {0:d} is larger/equal than length of corpus {1:d}.".format(n, length))
        if not 0 <= n:
            raise ValueError("Negative sample size n {0:d}.".format(n))

        i = 0
        for i, sample in enumerate(self.getstream()):
            if i == length:
                break

            remaining_in_corpus = length - i
            chance = random_generator.randint(1, remaining_in_corpus)
            if chance <= n:
                n -= 1
                if self.metadata:
                    yield self.preprocess_text(sample[0]), sample[1]
                else:
                    yield self.preprocess_text(sample)

        if n != 0:
            # This means that length was set to be greater than number of items in corpus
            # and we were not able to sample enough documents before the stream ended.
            raise ValueError("length {0:d} greater than number of documents in corpus {1:d}".format(length, i + 1))

    def __len__(self):
        if self.length is None:
            # cache the corpus length
            self.length = sum(1 for _ in self.getstream())
        return self.length
# endclass TextCorpus


class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory,
    where each file (or line of each file) is interpreted as a plain text document.
    """

    def __init__(self, input, dictionary=None, metadata=False, min_depth=0, max_depth=None,
                 pattern=None, exclude_pattern=None, lines_are_documents=False, **kwargs):
        """
        Args:
            min_depth (int): minimum depth in directory tree at which to begin searching for
                files. The default is 0, which means files starting in the top-level directory
                `input` will be considered.
            max_depth (int): max depth in directory tree at which files will no longer be
                considered. The default is None, which means recurse through all subdirectories.
            pattern (str or Pattern): regex to use for file name inclusion; all those files *not*
                matching this pattern will be ignored.
            exclude_pattern (str or Pattern): regex to use for file name exclusion; all files
                matching this pattern will be ignored.
            lines_are_documents (bool): if True, each line of each file is considered to be a
                document. If False (default), each file is considered to be a document.
            kwargs: keyword arguments passed through to the `TextCorpus` constructor. This is
                in addition to the non-kwargs `input`, `dictionary`, and `metadata`. See
                `TextCorpus.__init__` docstring for more details on these.
        """
        self._min_depth = min_depth
        self._max_depth = sys.maxsize if max_depth is None else max_depth
        self.pattern = pattern
        self.exclude_pattern = exclude_pattern
        self.lines_are_documents = lines_are_documents
        super(TextDirectoryCorpus, self).__init__(input, dictionary, metadata, **kwargs)

    @property
    def lines_are_documents(self):
        return self._lines_are_documents

    @lines_are_documents.setter
    def lines_are_documents(self, lines_are_documents):
        self._lines_are_documents = lines_are_documents
        self.length = None

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
        """Yield documents from the underlying plain text collection (of one or more files).
        Each item yielded from this method will be considered a document by subsequent
        preprocessing methods.

        If `lines_are_documents` was set to True, items will be lines from files. Otherwise
        there will be one item per file, containing the entire contents of the file.
        """
        num_texts = 0
        for path in self.iter_filepaths():
            with open(path, 'rt') as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip()
                        num_texts += 1
                else:
                    yield f.read().strip()
                    num_texts += 1

        self.length = num_texts

    def __len__(self):
        if self.length is None:
            self._cache_corpus_length()
        return self.length

    def _cache_corpus_length(self):
        if not self.lines_are_documents:
            self.length = sum(1 for _ in self.iter_filepaths())
        else:
            self.length = sum(1 for _ in self.getstream())
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
