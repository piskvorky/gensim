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

import functools
import itertools
import logging
import multiprocessing as mp
import os
import random
import re
import signal
import sys

from gensim.models.word2vec import MAX_WORDS_IN_BATCH

try:
    from itertools import imap
except ImportError:  # Python 3...
    imap = map

from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.text_processing_pool import TextProcessingPool, TextProcessor
from gensim.parsing.preprocessing import STOPWORDS, RE_WHITESPACE
from gensim.utils import deaccent, simple_tokenize, walk_with_depth

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


def init_to_ignore_interrupt():
    """Should only be used when master is prepared to handle termination of child processes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class TextPreprocessor(object):
    """Mixin for classes that perform text preprocessing."""

    def preprocess_text(self, text):
        """Apply preprocessing to a single text document. This should perform tokenization
        in addition to any other desired preprocessing steps.
        
        Note: The `TextCorpus` class transplants its own version of this method onto a
        dynamically created subclass that is used to spawn a multiprocessing worker.
        So if you want to subclass it in a `TextCorpus` subclass, and you want to call
        the super method using the `super` keyword, do it like this:
        
            # do some preprocessing of text
            tokens = super(self.__class__, self).preprocess_text(text)
            # do some post-processing of tokens

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


class _TextPreprocessorMP(TextProcessor, TextPreprocessor):
    """TextPreprocessor that can be used for multiprocessing."""
    pass


class TextCorpus(interfaces.CorpusABC, TextPreprocessor):
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
    def __init__(self, source=None, dictionary=None, metadata=False, character_filters=None,
                 tokenizer=None, token_filters=None, processes=-1):
        """
        Args:
            source (str): path to top-level directory to traverse for corpus documents.
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
        self.source = source
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

        max_cpus = mp.cpu_count() - 1
        if processes <= 0:
            self.processes = max(1, max_cpus)
        else:
            self.processes = min(processes, max_cpus)

        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)

    def init_dictionary(self, dictionary):
        """If `dictionary` is None, initialize to an empty Dictionary, and then if there
        is an `input` for the corpus, add all documents from that `input`. If the
        `dictionary` is already initialized, simply set it as the corpus's `dictionary`.
        """
        self.dictionary = dictionary if dictionary is not None else Dictionary()
        if self.source is not None:
            if dictionary is None:
                logger.info("Initializing dictionary")
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logger.info("Input stream provided but dictionary already initialized")
        else:
            logger.warning(
                "No input document stream provided; assuming "
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
        num_texts = 0
        with utils.file_or_filename(self.source) as f:
            for line in f:
                yield line
                num_texts += 1

        self.length = num_texts

    def _create_preprocessor_pool(self):
        state_kwargs = dict(
            character_filters=self.character_filters,
            tokenizer=self.tokenizer,
            token_filters=self.token_filters
        )
        _TextPreprocessor = type('_TextPreprocessor', (_TextPreprocessorMP,), {})
        func = getattr(self.__class__, 'preprocess_text')
        if hasattr(func, '__func__'):  # get unbound method in Python 2
            func = func.__func__
        _TextPreprocessor.process = func

        return TextProcessingPool(
            self.processes, init_to_ignore_interrupt,
            processor_class=_TextPreprocessor, state_kwargs=state_kwargs)

    def _get_mapper(self):
        if self.processes > 1:
            pool = self._create_preprocessor_pool()
            map_preprocess = pool.imap
        else:
            pool = None
            map_preprocess = functools.partial(imap, self.preprocess_text)

        return pool, map_preprocess

    def yield_tokens(self):
        texts = self.getstream()
        pool, map_preprocess = self._get_mapper()

        num_texts_total, num_texts = 0, 0
        num_positions_total, num_positions = 0, 0

        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
                for output in map_preprocess(group):
                    tokens = output[0] if isinstance(output, tuple) else output
                    num_texts_total += 1
                    num_positions_total += len(tokens)

                    if self.should_keep_tokens(output):
                        num_texts += 1
                        num_positions += len(tokens)
                        yield tokens
        except KeyboardInterrupt:
            logger.warning(
                "user terminated iteration over %s after %i docs with %i positions"
                " (total %i docs, %i positions before pruning)", self.__class__.__name__,
                num_texts, num_positions, num_texts_total, num_positions_total)
        else:
            logger.info(
                "finished iterating over %s of %i docs with %i positions"
                " (total %i docs, %i positions before pruning)", self.__class__.__name__,
                num_texts, num_positions, num_texts_total, num_positions_total)
            self.length = num_texts  # cache corpus length
        finally:
            if pool is not None:
                pool.terminate()

    def should_keep_tokens(self, output):
        """Output is either the list of tokens, or a tuple containing that list and some other
        elements from the preprocessing. The default implementation assumes it is the former
        and returns False if the list is empty.
        """
        return len(output) > 0

    def get_texts(self):
        """Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.

        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no further
        preprocessing of the words coming out of this function.
        """
        doc_token_stream = self.yield_tokens()
        if self.metadata:
            for lineno, tokens in enumerate(doc_token_stream):
                yield tokens, (lineno,)
        else:
            for tokens in doc_token_stream:
                yield tokens

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
            raise ValueError("n is larger than length of corpus.")
        if not 0 <= n:
            raise ValueError("Negative sample size.")

        # Use get_texts because some docs from getstream may be removed in preprocessing.
        for i, sample in enumerate(self.get_texts()):
            if i == length:
                break

            remaining_in_corpus = length - i
            chance = random_generator.randint(1, remaining_in_corpus)
            if chance <= n:
                n -= 1
                yield sample

        if n != 0:
            # This means that length was set to be greater than number of items in corpus
            # and we were not able to sample enough documents before the stream ended.
            raise ValueError("length greater than number of documents in corpus")

    def __len__(self):
        if self.length is None:
            self._cache_corpus_length()
        return self.length

    def _cache_corpus_length(self):
        # cache the corpus length
        # Use get_texts because some docs from getstream may be removed in preprocessing.
        self.length = sum(1 for _ in self.get_texts())
# endclass TextCorpus


class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory,
    where each file (or line of each file) is interpreted as a plain text document.
    """

    def __init__(self, source, dictionary=None, metadata=False, min_depth=0, max_depth=None,
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
        super(TextDirectoryCorpus, self).__init__(source, dictionary, metadata, **kwargs)

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
        for depth, dirpath, dirnames, filenames in walk_with_depth(self.source):
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
        for path in self.iter_filepaths():
            with open(path, 'rt') as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip()
                else:
                    yield f.read().strip()
# endclass TextDirectoryCorpus


def unicode_and_tokenize(text):
    return utils.to_unicode(text).split()


class TextTokensIterator(object):
    """Mixin for TextCorpus that changes its __iter__ to yield results of get_texts."""

    def __iter__(self):
        return self.get_texts()


class LineSentence(TextTokensIterator, TextCorpus):
    """Simple format: one sentence = one line.
    
    In general, words should already be preprocessed and separated by whitespace.
    If a line exceeds the `max_sentence_length`, it will be split into multiple sentences
    not exceeding this amount. Additional preprocessing can be applied using the `TextCorpus`
    preprocessing keyword arguments if needed.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None, **kwargs):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        kwargs['tokenizer'] = kwargs.get('tokenizer', unicode_and_tokenize)
        kwargs['character_filters'] = kwargs.get('character_filters', [])
        kwargs['token_filters'] = kwargs.get('token_filters', [])
        kwargs['processes'] = kwargs.get('processes', 1)
        TextCorpus.__init__(self, source, **kwargs)

    def getstream(self):
        with utils.smart_open(self.source) as fin:
            for line in itertools.islice(fin, self.limit):
                yield line

    def yield_tokens(self):
        doc_token_stream = super(self.__class__, self).yield_tokens()
        for tokens in doc_token_stream:
            i = 0
            while i < len(tokens):
                yield tokens[i: i + self.max_sentence_length]
                i += self.max_sentence_length


class PathLineSentences(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    Like LineSentence, but will process all files in a directory in alphabetical order by filename
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` should be a path to a directory (as a string) where all files can be opened by the
        LineSentence class. Each file will be read up to
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentencePath(os.getcwd() + '\\corpus\\')

        The files in the directory should be either text files, .bz2 files, or .gz files.

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

        if os.path.isfile(self.source):
            logging.warning('single file read, better to use models.word2vec.LineSentence')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logging.debug('reading directory ' + self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + file for file in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')

        logging.info('files read into PathLineSentences:' + '\n'.join(self.input_files))

    def __iter__(self):
        '''iterate through the files'''
        for file_name in self.input_files:
            logging.info('reading file ' + file_name)
            with utils.smart_open(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


class Text8Corpus(TextTokensIterator, TextCorpus):
    """Iterate over sentences from the "text8" corpus,
    unzipped from http://mattmahoney.net/dc/text8.zip.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, chunksize=65536, **kwargs):
        self.max_sentence_length = max_sentence_length
        self.chunksize = chunksize
        kwargs['tokenizer'] = kwargs.get('tokenizer', unicode_and_tokenize)
        kwargs['character_filters'] = kwargs.get('character_filters', [])
        TextCorpus.__init__(self, source, **kwargs)

    def _sentence_token_stream(self):
        # Entire corpus is one gigantic line -- there are no sentence marks at all.
        # So just split the token sequence arbitrarily into sentences of length
        # `max_sentence_length`.
        sentence, rest = [], b''
        with utils.smart_open(self.source) as fin:
            while True:
                text = rest + fin.read(self.chunksize)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = text.split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break

                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (text[:last_token].split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]

    def getstream(self):
        for sentence_tokens in self._sentence_token_stream():
            yield ' '.join(sentence_tokens)
