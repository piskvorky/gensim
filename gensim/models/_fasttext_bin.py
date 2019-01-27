# -*- coding: utf-8 -*-
"""Load models from the native binary format released by Facebook.

The main entry point is the :func:`~gensim.models._fasttext_bin.load` function.
It returns a :class:`~gensim.models._fasttext_bin.Model` namedtuple containing everything loaded from the binary.

Examples
--------

Load a model from a binary file:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.models.fasttext_bin import load
    >>> with open(datapath('crime-and-punishment.bin'), 'rb') as fin:
    ...     model = load(fin)
    >>> model.nwords
    291
    >>> model.vectors_ngrams.shape
    (391, 5)
    >>> sorted(model.raw_vocab, key=lambda w: len(w), reverse=True)[:5]
    ['останавливаться', 'изворачиваться,', 'раздражительном', 'exceptionally', 'проскользнуть']

See Also
--------

`FB Implementation <https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc>`_.

"""

import collections
import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

_FASTTEXT_FILEFORMAT_MAGIC = 793712314

_NEW_HEADER_FORMAT = [
    ('dim', 'i'),
    ('ws', 'i'),
    ('epoch', 'i'),
    ('min_count', 'i'),
    ('neg', 'i'),
    ('_', 'i'),
    ('loss', 'i'),
    ('model', 'i'),
    ('bucket', 'i'),
    ('minn', 'i'),
    ('maxn', 'i'),
    ('_', 'i'),
    ('t', 'd'),
]

_OLD_HEADER_FORMAT = [
    ('epoch', 'i'),
    ('min_count', 'i'),
    ('neg', 'i'),
    ('_', 'i'),
    ('loss', 'i'),
    ('model', 'i'),
    ('bucket', 'i'),
    ('minn', 'i'),
    ('maxn', 'i'),
    ('_', 'i'),
    ('t', 'd'),
]


def _yield_field_names():
    for name, _ in _OLD_HEADER_FORMAT + _NEW_HEADER_FORMAT:
        if not name.startswith('_'):
            yield name
    yield 'raw_vocab'
    yield 'vocab_size'
    yield 'nwords'
    yield 'vectors_ngrams'
    yield 'hidden_output'


_FIELD_NAMES = sorted(set(_yield_field_names()))
Model = collections.namedtuple('Model', _FIELD_NAMES)
"""Holds data loaded from the Facebook binary.

Parameters
----------
dim : int
    The dimensionality of the vectors.
ws : int
    The window size.
epoch : int
    The number of training epochs.
neg : int
    If non-zero, indicates that the model uses negative sampling.
loss : int
    If equal to 1, indicates that the model uses hierarchical sampling.
model : int
    If equal to 2, indicates that the model uses skip-grams.
bucket : int
    The number of buckets.
min_count : int
    The threshold below which the model ignores terms.
t : float
    The sample threshold.
minn : int
    The minimum ngram length.
maxn : int
    The maximum ngram length.
raw_vocab : collections.OrderedDict
    A map from words (str) to their frequency (int).  The order in the dict
    corresponds to the order of the words in the Facebook binary.
nwords : int
    The number of words.
vocab_size : int
    The size of the vocabulary.
vectors_ngrams : numpy.array
    This is a matrix that contains vectors learned by the model.
    Each row corresponds to a vector.
    The number of vectors is equal to the number of words plus the number of buckets.
    The number of columns is equal to the vector dimensionality.
hidden_output : numpy.array
    This is a matrix that contains the shallow neural network output.
    This array has the same dimensions as vectors_ngrams.
    May be None - in that case, it is impossible to continue training the model.
"""


def _struct_unpack(fin, fmt):
    num_bytes = struct.calcsize(fmt)
    return struct.unpack(fmt, fin.read(num_bytes))


def _load_vocab(fin, new_format, encoding='utf-8'):
    """Load a vocabulary from a FB binary.

    Before the vocab is ready for use, call the prepare_vocab function and pass
    in the relevant parameters from the model.

    Parameters
    ----------
    fin : file
        An open file pointer to the binary.
    new_format: boolean
        True if the binary is of the newer format.
    encoding : str
        The encoding to use when decoding binary data into words.

    Returns
    -------
    tuple
        The loaded vocabulary.  Keys are words, values are counts.
        The vocabulary size.
        The number of words.
    """
    vocab_size, nwords, nlabels = _struct_unpack(fin, '@3i')

    # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
    if nlabels > 0:
        raise NotImplementedError("Supervised fastText models are not supported")
    logger.info("loading %s words for fastText model from %s", vocab_size, fin.name)

    _struct_unpack(fin, '@1q')  # number of tokens
    if new_format:
        pruneidx_size, = _struct_unpack(fin, '@q')

    raw_vocab = collections.OrderedDict()
    for i in range(vocab_size):
        word_bytes = b''
        char_byte = fin.read(1)
        # Read vocab word
        while char_byte != b'\x00':
            word_bytes += char_byte
            char_byte = fin.read(1)
        word = word_bytes.decode(encoding)
        count, _ = _struct_unpack(fin, '@qb')
        raw_vocab[word] = count

    if new_format:
        for j in range(pruneidx_size):
            _struct_unpack(fin, '@2i')

    return raw_vocab, vocab_size, nwords


def _load_matrix(fin, new_format=True):
    """Load a matrix from fastText native format.

    Interprets the matrix dimensions and type from the file stream.

    Parameters
    ----------
    fin : file
        A file handle opened for reading.
    new_format : bool, optional
        True if the quant_input variable precedes
        the matrix declaration.  Should be True for newer versions of fastText.

    Returns
    -------
    :class:`numpy.array`
        The vectors as an array.
        Each vector will be a row in the array.
        The number of columns of the array will correspond to the vector size.

    """
    if new_format:
        _struct_unpack(fin, '@?')  # bool quant_input in fasttext.cc

    num_vectors, dim = _struct_unpack(fin, '@2q')

    float_size = struct.calcsize('@f')
    if float_size == 4:
        dtype = np.dtype(np.float32)
    elif float_size == 8:
        dtype = np.dtype(np.float64)
    else:
        raise ValueError("Incompatible float size: %r" % float_size)

    matrix = np.fromfile(fin, dtype=dtype, count=num_vectors * dim)
    matrix = matrix.reshape((num_vectors, dim))
    return matrix


def load(fin, encoding='utf-8', full_model=True):
    """Load a model from a binary stream.

    Parameters
    ----------
    fin : file
        The readable binary stream.
    encoding : str, optional
        The encoding to use for decoding text
    full_model : boolean, optional
        If False, skips loading the hidden output matrix.  This saves a fair bit
        of CPU time and RAM, but prevents training continuation.

    Returns
    -------
    :class:`~gensim.models._fasttext_bin.Model`
        The loaded model.

    """
    if isinstance(fin, str):
        fin = open(fin, 'rb')

    magic, version = _struct_unpack(fin, '@2i')
    new_format = magic == _FASTTEXT_FILEFORMAT_MAGIC

    header_spec = _NEW_HEADER_FORMAT if new_format else _OLD_HEADER_FORMAT
    model = {name: _struct_unpack(fin, fmt)[0] for (name, fmt) in header_spec}
    if not new_format:
        model.update(dim=magic, ws=version)

    raw_vocab, vocab_size, nwords = _load_vocab(fin, new_format, encoding=encoding)
    model.update(raw_vocab=raw_vocab, vocab_size=vocab_size, nwords=nwords)

    vectors_ngrams = _load_matrix(fin, new_format=new_format)

    if not full_model:
        hidden_output = None
    else:
        hidden_output = _load_matrix(fin, new_format=new_format)
        assert fin.read() == b'', 'expected to reach EOF'

    model.update(vectors_ngrams=vectors_ngrams, hidden_output=hidden_output)
    model = {k: v for k, v in model.items() if k in _FIELD_NAMES}
    return Model(**model)
