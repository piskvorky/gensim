#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains the classes of LineSentence and PathLineSentences."""
import logging
from gensim import utils
import itertools
import os

try:
    from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
except ImportError:
    raise utils.NO_CYTHON

logger = logging.getLogger(__name__)


class LineSentence(object):
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """Iterate over a file that contains sentences: one line = one sentence.
        Words must be already preprocessed and separated by whitespace.

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        limit : int or None
            Clip the file to the first `limit` lines. Do no clipping if `limit is None` (the default).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> sentences = LineSentence(datapath('lee_background.cor'))
            >>> for sentence in sentences:
            ...     pass

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.open(self.source, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


class PathLineSentences(object):
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """Like :class:`~gensim.corpora.linesentence.LineSentence`, but process all files in a directory
        in alphabetical order by filename.

        The directory must only contain files that can be read by :class:`gensim.corpora.linesentence.LineSentence`:
        .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

        The format of files (either text, or compressed text files) in the path is one sentence = one line,
        with words already preprocessed and separated by whitespace.

        Warnings
        --------
        Does **not recurse** into subdirectories.

        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using gensim.corpora.linesentence.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with utils.open(file_name, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length
