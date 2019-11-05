#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.

"""General functions used for any2vec models.

One of the goals of this module is to provide an abstraction over the Cython
extensions for FastText.  If they are not available, then the module substitutes
slower Python versions in their place.

Another related set of FastText functionality is computing ngrams for a word.
The :py:func:`compute_ngrams` and :py:func:`compute_ngrams_bytes` hashes achieve that.

Closely related is the functionality for hashing ngrams, implemented by the
:py:func:`ft_hash` and :py:func:`ft_hash_broken` functions.
The module exposes "working" and "broken" hash functions in order to maintain
backwards compatibility with older versions of Gensim.

For compatibility with older Gensim, use :py:func:`compute_ngrams` and
:py:func:`ft_hash_broken` to has each ngram.  For compatibility with the
current Facebook implementation, use :py:func:`compute_ngrams_bytes` and
:py:func:`ft_hash_bytes`.

"""

import logging
from gensim import utils

from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring

from six.moves import range
from six import iteritems, PY2

logger = logging.getLogger(__name__)


#
# UTF-8 bytes that begin with 10 are subsequent bytes of a multi-byte sequence,
# as opposed to a new character.
#
_MB_MASK = 0xC0
_MB_START = 0x80


def _byte_to_int_py3(b):
    return b


def _byte_to_int_py2(b):
    return ord(b)


_byte_to_int = _byte_to_int_py2 if PY2 else _byte_to_int_py3


def _is_utf8_continue(b):
    return _byte_to_int(b) & _MB_MASK == _MB_START


try:
    from gensim.models._utils_any2vec import (
        compute_ngrams,
        compute_ngrams_bytes,
        ft_hash_broken,
        ft_hash_bytes,
    )
except ImportError:
    raise utils.NO_CYTHON


def ft_ngram_hashes(word, minn, maxn, num_buckets, fb_compatible=True):
    """Calculate the ngrams of the word and hash them.

    Parameters
    ----------
    word : str
        The word to calculate ngram hashes for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length
    num_buckets : int
        The number of buckets
    fb_compatible : boolean, optional
        True for compatibility with the Facebook implementation.
        False for compatibility with the old Gensim implementation.

    Returns
    -------
        A list of hashes (integers), one per each detected ngram.

    """
    if fb_compatible:
        encoded_ngrams = compute_ngrams_bytes(word, minn, maxn)
        hashes = [ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
    else:
        text_ngrams = compute_ngrams(word, minn, maxn)
        hashes = [ft_hash_broken(n) % num_buckets for n in text_ngrams]
    return hashes


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
        with utils.open(fvocab, 'wb') as vout:
            for word, vocab_ in sorted(iteritems(vocab), key=lambda item: -item[1].count):
                vout.write(utils.to_utf8("%s %s\n" % (word, vocab_.count)))
    logger.info("storing %sx%s projection weights into %s", total_vec, vector_size, fname)
    assert (len(vocab), vector_size) == vectors.shape
    with utils.open(fname, 'wb') as fout:
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
        with utils.open(fvocab, 'rb') as fin:
            for line in fin:
                word, count = utils.to_unicode(line, errors=unicode_errors).strip().split()
                counts[word] = int(count)

    logger.info("loading projection weights from %s", fname)
    with utils.open(fname, 'rb') as fin:
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
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)  # Python uses I/O buffering internally
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                with utils.ignore_deprecation_warning():
                    # TODO use frombuffer or something similar
                    weights = fromstring(fin.read(binary_len), dtype=REAL).astype(datatype)
                add_word(word, weights)
        else:
            for line_no in range(vocab_size):
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
