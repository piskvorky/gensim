#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Module provides some code scaffolding to simplify use of built dictionary for constructing BoW vectors.

Notes
-----
Text corpora usually reside on disk, as text files in one format or another In a common scenario,
we need to build a dictionary (a `word->integer id` mapping), which is then used to construct sparse bag-of-word vectors
(= iterable of `(word_id, word_weight)`).

This module provides some code scaffolding to simplify this pipeline. For example, given a corpus where each document
is a separate line in file on disk, you would override the :meth:`gensim.corpora.textcorpus.TextCorpus.get_texts`
to read one line=document at a time, process it (lowercase, tokenize, whatever) and yield it as a sequence of words.

Overriding :meth:`gensim.corpora.textcorpus.TextCorpus.get_texts` is enough, you can then initialize the corpus
with e.g. `MyTextCorpus("mycorpus.txt.bz2")` and it will behave correctly like a corpus of sparse vectors.
The :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__` method is automatically set up,
and dictionary is automatically populated with all `word->id` mappings.

The resulting object can be used as input to some of gensim models (:class:`~gensim.models.tfidfmodel.TfidfModel`,
:class:`~gensim.models.lsimodel.LsiModel`, :class:`~gensim.models.ldamodel.LdaModel`, ...), serialized with any format
(`Matrix Market <http://math.nist.gov/MatrixMarket/formats.html>`_,
`SvmLight <http://svmlight.joachims.org/>`_, `Blei's LDA-C format <https://github.com/blei-lab/lda-c>`_, etc).


See Also
--------
:class:`gensim.test.test_miislita.CorpusMiislita`
    Good simple example.

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
    """Remove stopwords using list from `gensim.parsing.preprocessing.STOPWORDS`.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    stopwords : iterable of str, optional
        Sequence of stopwords

    Returns
    -------
    list of str
        List of tokens without `stopwords`.

    """
    return [token for token in tokens if token not in stopwords]


def remove_short(tokens, minsize=3):
    """Remove tokens shorter than `minsize` chars.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    minsize : int, optimal
        Minimal length of token (include).

    Returns
    -------
    list of str
        List of tokens without short tokens.

    """
    return [token for token in tokens if len(token) >= minsize]


def lower_to_unicode(text, encoding='utf8', errors='strict'):
    """Lowercase `text` and convert to unicode, using :func:`gensim.utils.any2unicode`.

    Parameters
    ----------
    text : str
        Input text.
    encoding : str, optional
        Encoding that will be used for conversion.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).

    Returns
    -------
    str
        Unicode version of `text`.

    See Also
    --------
    :func:`gensim.utils.any2unicode`
        Convert any string to unicode-string.

    """
    return utils.to_unicode(text.lower(), encoding, errors)


def strip_multiple_whitespaces(s):
    """Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    s : str
        Input string

    Returns
    -------
    str
        String with collapsed whitespaces.

    """
    return RE_WHITESPACE.sub(" ", s)


class TextCorpus(interfaces.CorpusABC):
    """Helper class to simplify the pipeline of getting BoW vectors from plain text.

    Notes
    -----
    This is an abstract base class: override the :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` and
    :meth:`~gensim.corpora.textcorpus.TextCorpus.__len__` methods to match your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object will be automatically initialized
    with a dictionary in `self.dictionary` and will support the :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__`
    corpus method.  You have a few different ways of utilizing this class via subclassing or by construction with
    different preprocessing arguments.

    The :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__` method converts the lists of tokens produced by
    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` to BoW format using
    :meth:`gensim.corpora.dictionary.Dictionary.doc2bow`.

    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` does the following:

    #. Calls :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to get a generator over the texts.
       It yields each document in turn from the underlying text file or files.
    #. For each document from the stream, calls :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` to produce
       a list of tokens. If metadata=True, it yields a 2-`tuple` with the document number as the second element.

    Preprocessing consists of 0+ `character_filters`, a `tokenizer`, and 0+ `token_filters`.

    The preprocessing consists of calling each filter in `character_filters` with the document text.
    Unicode is not guaranteed, and if desired, the first filter should convert to unicode.
    The output of each character filter should be another string. The output from the final filter is fed
    to the `tokenizer`, which should split the string into a list of tokens (strings).
    Afterwards, the list of tokens is fed through each filter in `token_filters`. The final output returned from
    :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` is the output from the final token filter.

    So to use this class, you can either pass in different preprocessing functions using the
    `character_filters`, `tokenizer`, and `token_filters` arguments, or you can subclass it.

    If subclassing: override :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to take text from different input
    sources in different formats.
    Override :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` if you must provide different initial
    preprocessing, then call the :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` method to apply
    the normal preprocessing.
    You can also override :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` in order to tag the documents
    (token lists) with different metadata.

    The default preprocessing consists of:

    #. :func:`~gensim.corpora.textcorpus.lower_to_unicode` - lowercase and convert to unicode (assumes utf8 encoding)
    #. :func:`~gensim.utils.deaccent`- deaccent (asciifolding)
    #. :func:`~gensim.corpora.textcorpus.strip_multiple_whitespaces` - collapse multiple whitespaces into a single one
    #. :func:`~gensim.utils.simple_tokenize` - tokenize by splitting on whitespace
    #. :func:`~gensim.corpora.textcorpus.remove_short` - remove words less than 3 characters long
    #. :func:`~gensim.corpora.textcorpus.remove_stopwords` - remove stopwords

    """

    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None,
                 tokenizer=None, token_filters=None):
        """

        Parameters
        ----------
        input : str, optional
            Path to top-level directory (file) to traverse for corpus documents.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
            If `input` is None, the dictionary will remain uninitialized.
        metadata : bool, optional
            If True - yield metadata with each document.
        character_filters : iterable of callable, optional
            Each will be applied to the text of each document in order, and should return a single string with
            the modified text. For Python 2, the original text will not be unicode, so it may be useful to
            convert to unicode as the first character filter.
            If None - using :func:`~gensim.corpora.textcorpus.lower_to_unicode`,
            :func:`~gensim.utils.deaccent` and :func:`~gensim.corpora.textcorpus.strip_multiple_whitespaces`.
        tokenizer : callable, optional
            Tokenizer for document, if None - using :func:`~gensim.utils.simple_tokenize`.
        token_filters : iterable of callable, optional
            Each will be applied to the iterable of tokens in order, and should return another iterable of tokens.
            These filters can add, remove, or replace tokens, or do nothing at all.
            If None - using :func:`~gensim.corpora.textcorpus.remove_short` and
            :func:`~gensim.corpora.textcorpus.remove_stopwords`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim import utils
            >>>
            >>>
            >>> class CorpusMiislita(TextCorpus):
            ...     stopwords = set('for a of the and to in on'.split())
            ...
            ...     def get_texts(self):
            ...         for doc in self.getstream():
            ...             yield [word for word in utils.to_unicode(doc).lower().split() if word not in self.stopwords]
            ...
            ...     def __len__(self):
            ...         self.length = sum(1 for _ in self.get_texts())
            ...         return self.length
            >>>
            >>>
            >>> corpus = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))
            >>> len(corpus)
            250
            >>> document = next(iter(corpus.get_texts()))

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
        """Initialize/update dictionary.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.

        Notes
        -----
        If self.input is None - make nothing.

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
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format (+ metadata if self.metadata).

        """
        if self.metadata:
            for text, metadata in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False), metadata
        else:
            for text in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            Document read from plain-text file.

        Notes
        -----
        After generator end - initialize self.length attribute.

        """
        num_texts = 0
        with utils.file_or_filename(self.input) as f:
            for line in f:
                yield line
                num_texts += 1

        self.length = num_texts

    def preprocess_text(self, text):
        """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.

        Parameters
        ---------
        text : str
            Document read from plain-text file.

        Return
        ------
        list of str
            List of tokens extracted from `text`.

        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    def step_through_preprocess(self, text):
        """Apply preprocessor one by one and generate result.

        Warnings
        --------
        This is useful for debugging issues with the corpus preprocessing pipeline.

        Parameters
        ----------
        text : str
            Document text read from plain-text file.

        Yields
        ------
        (callable, object)
            Pre-processor, output from pre-processor (based on `text`)

        """
        for character_filter in self.character_filters:
            text = character_filter(text)
            yield (character_filter, text)

        tokens = self.tokenizer(text)
        yield (self.tokenizer, tokens)

        for token_filter in self.token_filters:
            yield (token_filter, token_filter(tokens))

    def get_texts(self):
        """Generate documents from corpus.

        Yields
        ------
        list of str
            Document as sequence of tokens (+ lineno if self.metadata)

        """
        lines = self.getstream()
        if self.metadata:
            for lineno, line in enumerate(lines):
                yield self.preprocess_text(line), (lineno,)
        else:
            for line in lines:
                yield self.preprocess_text(line)

    def sample_texts(self, n, seed=None, length=None):
        """Generate `n` random documents from the corpus without replacement.

        Parameters
        ----------
        n : int
            Number of documents we want to sample.
        seed : int, optional
            If specified, use it as a seed for local random generator.
        length : int, optional
            Value will used as corpus length (because calculate length of corpus can be costly operation).
            If not specified - will call `__length__`.

        Raises
        ------
        ValueError
            If `n` less than zero or greater than corpus size.

        Notes
        -----
        Given the number of remaining documents in a corpus, we need to choose n elements.
        The probability for the current element to be chosen is `n` / remaining. If we choose it,  we just decrease
        the `n` and move to the next element.

        Yields
        ------
        list of str
            Sampled document as sequence of tokens.

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
        """Get length of corpus

        Warnings
        --------
        If self.length is None - will read all corpus for calculate this attribute through
        :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream`.

        Returns
        -------
        int
            Length of corpus.

        """
        if self.length is None:
            # cache the corpus length
            self.length = sum(1 for _ in self.getstream())
        return self.length


class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory.
    Each file/line (depends on `lines_are_documents`) is interpreted as a plain text document.

    """

    def __init__(self, input, dictionary=None, metadata=False, min_depth=0, max_depth=None,
                 pattern=None, exclude_pattern=None, lines_are_documents=False, **kwargs):
        """

        Parameters
        ----------
        input : str
            Path to input file/folder.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
            If `input` is None, the dictionary will remain uninitialized.
        metadata : bool, optional
            If True - yield metadata with each document.
        min_depth : int, optional
            Minimum depth in directory tree at which to begin searching for files.
        max_depth : int, optional
            Max depth in directory tree at which files will no longer be considered.
            If None - not limited.
        pattern : str, optional
            Regex to use for file name inclusion, all those files *not* matching this pattern will be ignored.
        exclude_pattern : str, optional
            Regex to use for file name exclusion, all files matching this pattern will be ignored.
        lines_are_documents : bool, optional
            If True - each line is considered a document, otherwise - each file is one document.
        kwargs: keyword arguments passed through to the `TextCorpus` constructor.
            See :meth:`gemsim.corpora.textcorpus.TextCorpus.__init__` docstring for more details on these.

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
        """Generate (lazily)  paths to each file in the directory structure within the specified range of depths.
        If a filename pattern to match was given, further filter to only those filenames that match.

        Yields
        ------
        str
            Path to file

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
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            One document (if lines_are_documents - True), otherwise - each file is one document.

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
        """Get length of corpus.

        Returns
        -------
        int
            Length of corpus.

        """
        if self.length is None:
            self._cache_corpus_length()
        return self.length

    def _cache_corpus_length(self):
        """Calculate length of corpus and cache it to `self.length`."""
        if not self.lines_are_documents:
            self.length = sum(1 for _ in self.iter_filepaths())
        else:
            self.length = sum(1 for _ in self.getstream())


def walk(top, topdown=True, onerror=None, followlinks=False, depth=0):
    """Generate the file names in a directory tree by walking the tree either top-down or bottom-up.
    For each directory in the tree rooted at directory top (including top itself), it yields a 4-tuple
    (depth, dirpath, dirnames, filenames).

    Parameters
    ----------
    top : str
        Root directory.
    topdown : bool, optional
        If True - you can modify dirnames in-place.
    onerror : function, optional
        Some function, will be called with one argument, an OSError instance.
        It can report the error to continue with the walk, or raise the exception to abort the walk.
        Note that the filename is available as the filename attribute of the exception object.
    followlinks : bool, optional
        If True - visit directories pointed to by symlinks, on systems that support them.
    depth : int, optional
        Height of file-tree, don't pass it manually (this used as accumulator for recursion).

    Notes
    -----
    This is a mostly copied version of `os.walk` from the Python 2 source code.
    The only difference is that it returns the depth in the directory tree structure
    at which each yield is taking place.

    Yields
    ------
    (int, str, list of str, list of str)
        Depth, current path, visited directories, visited non-directories.

    See Also
    --------
    `os.walk documentation <https://docs.python.org/2/library/os.html#os.walk>`_

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
