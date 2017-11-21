#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Base Indexed Corpus class"""

import logging
import six

import numpy

from gensim import interfaces, utils

logger = logging.getLogger('gensim.corpora.indexedcorpus')


class IndexedCorpus(interfaces.CorpusABC):
    def __init__(self, fname, index_fname=None):
        """Indexed corpus is a mechanism for random-accessing corpora.

        While the standard corpus interface in gensim allows iterating over
        corpus with `for doc in corpus: pass`, indexed corpus allows accessing
        the documents with `corpus[docno]` (in O(1) look-up time).

        This functionality is achieved by storing an extra file (by default
        named the same as the '{corpus name}.index') that stores the byte
        offset of the beginning of each document.

        Parameters
        ----------
        fname : string
            Corpus filename
        index_fname : string or None
            Index filename, or None for loading `fname`.index

        Examples
        --------
        >>> # save corpus in SvmLightCorpus format with an index
            >>> corpus = [[(1, 0.5)], [(0, 1.0), (1, 2.0)]]
            >>> gensim.corpora.SvmLightCorpus.serialize('testfile.svmlight', corpus)
            >>> # load back as a document stream (*not* plain Python list)
            >>> corpus_with_random_access = gensim.corpora.SvmLightCorpus('tstfile.svmlight')
            >>> print(corpus_with_random_access[1])
            [(0, 1.0), (1, 2.0)]

        """
        try:
            if index_fname is None:
                index_fname = utils.smart_extension(fname, '.index')
            self.index = utils.unpickle(index_fname)
            # change self.index into a numpy.ndarray to support fancy indexing
            self.index = numpy.asarray(self.index)
            logger.info("loaded corpus index from %s", index_fname)
        except Exception:
            self.index = None
        self.length = None

    @classmethod
    def serialize(serializer, fname, corpus, id2word=None, index_fname=None,
                  progress_cnt=None, labels=None, metadata=False):
        """Iterate through the document stream `corpus`, saving the documents to
        `fname` and recording byte offset of each document.

        Save the resulting index structure to file `index_fname` (or
        `fname`.index is not set).

        This relies on the underlying corpus class `serializer` providing (in
        addition to standard iteration)::

            * `save_corpus` method that returns a sequence of byte offsets, one for
               each saved document
            * the `docbyoffset(offset)` method, which returns a document
              positioned at `offset` bytes within the persistent storage (file)
            * metadata if set to true will ensure that serialize will write out
            article titles to a pickle file.

        Parameters
        ----------
        fname : str
            Filename
        corpus : iterable
            Iterable of documents
        id2word : dict of (str, str), optional
            Transforms id to word (Default value = None)
        index_fname : str
             (Default value = None)
        progress_cnt : int
             (Default value = None)
        labels :
             (Default value = None)
        metadata : bool
            Any additional info (Default value = False)

        Examples
        --------
        >>> MmCorpus.serialize('test.mm', corpus)
        >>> mm = MmCorpus('test.mm') # `mm` document stream now has random access
        >>> print(mm[42]) # retrieve document no. 42, etc.
        """
        if getattr(corpus, 'fname', None) == fname:
            raise ValueError("identical input vs. output corpus filename, refusing to serialize: %s" % fname)

        if index_fname is None:
            index_fname = utils.smart_extension(fname, '.index')

        kwargs = {'metadata': metadata}
        if progress_cnt is not None:
            kwargs['progress_cnt'] = progress_cnt

        if labels is not None:
            kwargs['labels'] = labels

        offsets = serializer.save_corpus(fname, corpus, id2word, **kwargs)

        if offsets is None:
            raise NotImplementedError(
                "Called serialize on class %s which doesn't support indexing!" % serializer.__name__
            )

        # store offsets persistently, using pickle
        # we shouldn't have to worry about self.index being a numpy.ndarray as the serializer will return
        # the offsets that are actually stored on disk - we're not storing self.index in any case, the
        # load just needs to turn whatever is loaded from disk back into a ndarray - this should also ensure
        # backwards compatibility
        logger.info("saving %s index to %s", serializer.__name__, index_fname)
        utils.pickle(offsets, index_fname)

    def __len__(self):
        """
        Return the index length.

        If the corpus is not indexed, also count corpus length and cache this
        value.
        """
        if self.index is not None:
            return len(self.index)
        if self.length is None:
            logger.info("caching corpus length")
            self.length = sum(1 for _ in self)
        return self.length

    def __getitem__(self, docno):
        if self.index is None:
            raise RuntimeError("Cannot call corpus[docid] without an index")
        if isinstance(docno, (slice, list, numpy.ndarray)):
            return utils.SlicedCorpus(self, docno)
        elif isinstance(docno, six.integer_types + (numpy.integer,)):
            return self.docbyoffset(self.index[docno])
        else:
            raise ValueError('Unrecognised value for docno, use either a single integer, a slice or a numpy.ndarray')
