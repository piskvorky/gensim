#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.

"""General functions used for any2vec models."""

import logging
import numpy as np
from gensim import utils

from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring

from six.moves import xrange
from six import iteritems

logger = logging.getLogger(__name__)

try:
    from gensim.models._utils_any2vec import ft_hash as _ft_hash, compute_ngrams as _compute_ngrams
except ImportError:
    FAST_VERSION = -1

    # failed... fall back to plain python
    def _ft_hash(string):
        """Calculate hash based on `string`.
        Reproduce `hash method from Facebook fastText implementation
        <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`_.

        Parameters
        ----------
        string : str
            The string whose hash needs to be calculated.

        Returns
        -------
        int
            The hash of the string.

        """
        # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
        old_settings = np.seterr(all='ignore')
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

    def _compute_ngrams(word, min_n, max_n):
        """Get the list of all possible ngrams for a given word.

        Parameters
        ----------
        word : str
            The word whose ngrams need to be computed.
        min_n : int
            Minimum character length of the ngrams.
        max_n : int
            Maximum character length of the ngrams.

        Returns
        -------
        list of str
            Sequence of character ngrams.

        """
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return ngrams


def _save_word2vec_format(fname, vocab, vectors, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        vocab : dict
            The vocabulary of words.
        vectors : numpy.array
            The vectors to be stored.
        fvocab : str, optional
            File path used to save the vocabulary.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Explicitly specify total number of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        if not (vocab or vectors):
            raise RuntimeError("no input")
        if total_vec is None:
            total_vec = len(vocab)
        vector_size = vectors.shape[1]
        if fvocab is not None:
            logger.info("storing vocabulary in %s", fvocab)
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab_ in sorted(iteritems(vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab_.count)))
        logger.info("storing %sx%s projection weights into %s", total_vec, vector_size, fname)
        assert (len(vocab), vector_size) == vectors.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, vocab_ in sorted(iteritems(vocab), key=lambda item: -item[1].count):
                row = vectors[vocab_.index]
                if binary:
                    row = row.astype(REAL)
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


def _load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                          limit=None, datatype=REAL):
    """Load the input-hidden weight matrix from the original C word2vec-tool format.

    Note that the information stored in the file is incomplete (the binary tree is missing),
    so while you can query for word similarity etc., you cannot continue training
    with a model loaded this way.

    Parameters
    ----------
    fname : str
        The file path to the saved word2vec-format file.
    fvocab : str, optional
        File path to the vocabulary.Word counts are read from `fvocab` filename, if set
        (this is the file generated by `-save-vocab` flag of the original C tool).
    binary : bool, optional
        If True, indicates whether the data is in binary word2vec format.
    encoding : str, optional
        If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.
    unicode_errors : str, optional
        default 'strict', is a string suitable to be passed as the `errors`
        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
        file may include word tokens truncated in the middle of a multibyte unicode character
        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
    limit : int, optional
        Sets a maximum number of word-vectors to read from the file. The default,
        None, means read all.
    datatype : type, optional
        (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.
        Such types may result in much slower bulk operations or incompatibility with optimized routines.)

    Returns
    -------
    object
        Returns the loaded model as an instance of :class:`cls`.

    """
    from gensim.models.keyedvectors import Vocab
    counts = None
    if fvocab is not None:
        logger.info("loading word counts from %s", fvocab)
        counts = {}
        with utils.smart_open(fvocab) as fin:
            for line in fin:
                word, count = utils.to_unicode(line).strip().split()
                counts[word] = int(count)

    logger.info("loading projection weights from %s", fname)
    with utils.smart_open(fname) as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding)
        vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
        if limit:
            vocab_size = min(vocab_size, limit)
        result = cls(vector_size)
        result.vector_size = vector_size
        result.vectors = zeros((vocab_size, vector_size), dtype=datatype)

        def add_word(word, weights):
            word_id = len(result.vocab)
            if word in result.vocab:
                logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                return
            if counts is None:
                # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
            elif word in counts:
                # use count from the vocab file
                result.vocab[word] = Vocab(index=word_id, count=counts[word])
            else:
                # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                result.vocab[word] = Vocab(index=word_id, count=None)
            result.vectors[word_id] = weights
            result.index2word.append(word)

        if binary:
            binary_len = dtype(REAL).itemsize * vector_size
            for _ in xrange(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                weights = fromstring(fin.read(binary_len), dtype=REAL).astype(datatype)
                add_word(word, weights)
        else:
            for line_no in xrange(vocab_size):
                line = fin.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], [datatype(x) for x in parts[1:]]
                add_word(word, weights)
    if result.vectors.shape[0] != len(result.vocab):
        logger.info(
            "duplicate words detected, shrinking matrix size from %i to %i",
            result.vectors.shape[0], len(result.vocab)
        )
        result.vectors = ascontiguousarray(result.vectors[: len(result.vocab)])
    assert (len(result.vocab), vector_size) == result.vectors.shape

    logger.info("loaded %s matrix from %s", result.vectors.shape, fname)
    return result
