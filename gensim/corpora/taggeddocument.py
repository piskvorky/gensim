#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Zygmunt Zając <zygmunt@fastml.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains the classes of TaggedDocument and TaggedLineDocument."""

from gensim import utils
from collections import namedtuple


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """Represents a document along with a tag, input document format for :class:`~gensim.models.doc2vec.Doc2Vec`.

    A single document, made up of `words` (a list of unicode string tokens) and `tags` (a list of tokens).
    Tags may be one or more unicode string tokens, but typical practice (which will also be the most memory-efficient)
    is for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from :class:`gensim.models.word2vec.Word2Vec`.

    """
    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).

        """
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.tags)


class TaggedLineDocument(object):
    def __init__(self, source):
        """Iterate over a file that contains documents: one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.

        Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
        automatically from the document line number (each document gets a unique integer tag).

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.doc2vec import TaggedLineDocument
            >>>
            >>> for document in TaggedLineDocument(datapath("head500.noblanks.cor")):
            ...     pass

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source` specified in the constructor.

        """
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.open(self.source, 'rb') as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
