#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Michael Penkov <m@penkov.dev>
# Copyright (C) 2019 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

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
import gzip
import io
import logging
import struct

import numpy as np

_END_OF_WORD_MARKER = b'\x00'

# FastText dictionary data structure holds elements of type `entry` which can have `entry_type`
# either `word` (0 :: int8) or `label` (1 :: int8). Here we deal with unsupervised case only
# so we want `word` type.
# See https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h

_DICT_WORD_ENTRY_TYPE_MARKER = b'\x00'


logger = logging.getLogger(__name__)

# Constants for FastText vesrion and FastText file format magic (both int32)
# https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc#L25

_FASTTEXT_VERSION = np.int32(12)
_FASTTEXT_FILEFORMAT_MAGIC = np.int32(793712314)


# _NEW_HEADER_FORMAT is constructed on the basis of args::save method, see
# https://github.com/facebookresearch/fastText/blob/master/src/args.cc

_NEW_HEADER_FORMAT = [
    ('dim', 'i'),
    ('ws', 'i'),
    ('epoch', 'i'),
    ('min_count', 'i'),
    ('neg', 'i'),
    ('word_ngrams', 'i'),   # Unused in loading
    ('loss', 'i'),
    ('model', 'i'),
    ('bucket', 'i'),
    ('minn', 'i'),
    ('maxn', 'i'),
    ('lr_update_rate', 'i'),   # Unused in loading
    ('t', 'd'),
]

_OLD_HEADER_FORMAT = [
    ('epoch', 'i'),
    ('min_count', 'i'),
    ('neg', 'i'),
    ('word_ngrams', 'i'),  # Unused in loading
    ('loss', 'i'),
    ('model', 'i'),
    ('bucket', 'i'),
    ('minn', 'i'),
    ('maxn', 'i'),
    ('lr_update_rate', 'i'),  # Unused in loading
    ('t', 'd'),
]

_FLOAT_SIZE = struct.calcsize('@f')
if _FLOAT_SIZE == 4:
    _FLOAT_DTYPE = np.dtype(np.float32)
elif _FLOAT_SIZE == 8:
    _FLOAT_DTYPE = np.dtype(np.float64)
else:
    _FLOAT_DTYPE = None


def _yield_field_names():
    for name, _ in _OLD_HEADER_FORMAT + _NEW_HEADER_FORMAT:
        if not name.startswith('_'):
            yield name
    yield 'raw_vocab'
    yield 'vocab_size'
    yield 'nwords'
    yield 'vectors_ngrams'
    yield 'hidden_output'
    yield 'ntokens'


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
        The number of tokens.
    """
    vocab_size, nwords, nlabels = _struct_unpack(fin, '@3i')

    # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
    if nlabels > 0:
        raise NotImplementedError("Supervised fastText models are not supported")
    logger.info("loading %s words for fastText model from %s", vocab_size, fin.name)

    ntokens = _struct_unpack(fin, '@q')[0]  # number of tokens

    if new_format:
        pruneidx_size, = _struct_unpack(fin, '@q')

    raw_vocab = collections.OrderedDict()
    for i in range(vocab_size):
        word_bytes = io.BytesIO()
        char_byte = fin.read(1)

        while char_byte != _END_OF_WORD_MARKER:
            word_bytes.write(char_byte)
            char_byte = fin.read(1)

        word_bytes = word_bytes.getvalue()
        try:
            word = word_bytes.decode(encoding)
        except UnicodeDecodeError:
            word = word_bytes.decode(encoding, errors='backslashreplace')
            logger.error(
                'failed to decode invalid unicode bytes %r; replacing invalid characters, using %r',
                word_bytes, word
            )
        count, _ = _struct_unpack(fin, '@qb')
        raw_vocab[word] = count

    if new_format:
        for j in range(pruneidx_size):
            _struct_unpack(fin, '@2i')

    return raw_vocab, vocab_size, nwords, ntokens


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
    if _FLOAT_DTYPE is None:
        raise ValueError('bad _FLOAT_SIZE: %r' % _FLOAT_SIZE)

    if new_format:
        _struct_unpack(fin, '@?')  # bool quant_input in fasttext.cc

    num_vectors, dim = _struct_unpack(fin, '@2q')
    count = num_vectors * dim

    #
    # numpy.fromfile doesn't play well with gzip.GzipFile as input:
    #
    # - https://github.com/RaRe-Technologies/gensim/pull/2476
    # - https://github.com/numpy/numpy/issues/13470
    #
    # Until they fix it, we have to apply a workaround.  We only apply the
    # workaround when it's necessary, because np.fromfile is heavily optimized
    # and very efficient (when it works).
    #
    if isinstance(fin, gzip.GzipFile):
        logger.warning(
            'Loading model from a compressed .gz file.  This can be slow. '
            'This is a work-around for a bug in NumPy: https://github.com/numpy/numpy/issues/13470. '
            'Consider decompressing your model file for a faster load. '
        )
        matrix = _fromfile(fin, _FLOAT_DTYPE, count)
    else:
        matrix = np.fromfile(fin, _FLOAT_DTYPE, count)

    assert matrix.shape == (count,), 'expected (%r,),  got %r' % (count, matrix.shape)
    matrix = matrix.reshape((num_vectors, dim))
    return matrix


def _batched_generator(fin, count, batch_size=1e6):
    """Read `count` floats from `fin`.

    Batches up read calls to avoid I/O overhead.  Keeps no more than batch_size
    floats in memory at once.

    Yields floats.

    """
    while count > batch_size:
        batch = _struct_unpack(fin, '@%df' % batch_size)
        for f in batch:
            yield f
        count -= batch_size

    batch = _struct_unpack(fin, '@%df' % count)
    for f in batch:
        yield f


def _fromfile(fin, dtype, count):
    """Reimplementation of numpy.fromfile."""
    return np.fromiter(_batched_generator(fin, count), dtype=dtype)


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

    raw_vocab, vocab_size, nwords, ntokens = _load_vocab(fin, new_format, encoding=encoding)
    model.update(raw_vocab=raw_vocab, vocab_size=vocab_size, nwords=nwords, ntokens=ntokens)

    vectors_ngrams = _load_matrix(fin, new_format=new_format)

    if not full_model:
        hidden_output = None
    else:
        hidden_output = _load_matrix(fin, new_format=new_format)
        assert fin.read() == b'', 'expected to reach EOF'

    model.update(vectors_ngrams=vectors_ngrams, hidden_output=hidden_output)
    model = {k: v for k, v in model.items() if k in _FIELD_NAMES}
    return Model(**model)


def _backslashreplace_backport(ex):
    """Replace byte sequences that failed to decode with character escapes.

    Does the same thing as errors="backslashreplace" from Python 3.  Python 2
    lacks this functionality out of the box, so we need to backport it.

    Parameters
    ----------
    ex: UnicodeDecodeError
        contains arguments of the string and start/end indexes of the bad portion.

    Returns
    -------
    text: unicode
        The Unicode string corresponding to the decoding of the bad section.
    end: int
        The index from which to continue decoding.

    Note
    ----
    Works on Py2 only.  Py3 already has backslashreplace built-in.

    """
    #
    # Based on:
    # https://stackoverflow.com/questions/42860186/exact-equivalent-of-b-decodeutf-8-backslashreplace-in-python-2
    #
    bstr, start, end = ex.object, ex.start, ex.end
    text = u''.join('\\x{:02x}'.format(ord(c)) for c in bstr[start:end])
    return text, end


def _sign_model(fout):
    """
    Write signature of the file in Facebook's native fastText `.bin` format
    to the binary output stream `fout`. Signature includes magic bytes and version.

    Name mimics original C++ implementation, see
    [FastText::signModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)

    Parameters
    ----------
    fout: writeable binary stream
    """
    fout.write(_FASTTEXT_FILEFORMAT_MAGIC.tobytes())
    fout.write(_FASTTEXT_VERSION.tobytes())


def _conv_field_to_bytes(field_value, field_type):
    """
    Auxiliary function that converts `field_value` to bytes based on request `field_type`,
    for saving to the binary file.

    Parameters
    ----------
    field_value: numerical
        contains arguments of the string and start/end indexes of the bad portion.

    field_type: str
        currently supported `field_types` are `i` for 32-bit integer and `d` for 64-bit float
    """
    if field_type == 'i':
        return (np.int32(field_value).tobytes())
    elif field_type == 'd':
        return (np.float64(field_value).tobytes())
    else:
        raise NotImplementedError('Currently conversion to "%s" type is not implemmented.' % field_type)


def _get_field_from_model(model, field):
    """
    Extract `field` from `model`.

    Parameters
    ----------
    model: gensim.models.fasttext.FastText
        model from which `field` is extracted
    field: str
        requested field name, fields are listed in the `_NEW_HEADER_FORMAT` list
    """
    if field == 'bucket':
        return model.wv.bucket
    elif field == 'dim':
        return model.vector_size
    elif field == 'epoch':
        return model.epochs
    elif field == 'loss':
        # `loss` => hs: 1, ns: 2, softmax: 3, ova-vs-all: 4
        # ns = negative sampling loss (default)
        # hs = hierarchical softmax loss
        # softmax =  softmax loss
        # one-vs-all = one vs all loss (supervised)
        if model.hs == 1:
            return 1
        elif model.hs == 0:
            return 2
        elif model.hs == 0 and model.negative == 0:
            return 1
    elif field == 'maxn':
        return model.wv.max_n
    elif field == 'minn':
        return model.wv.min_n
    elif field == 'min_count':
        return model.min_count
    elif field == 'model':
        # `model` => cbow:1, sg:2, sup:3
        # cbow = continous bag of words (default)
        # sg = skip-gram
        # sup = supervised
        return 2 if model.sg == 1 else 1
    elif field == 'neg':
        return model.negative
    elif field == 't':
        return model.sample
    elif field == 'word_ngrams':
        # This is skipped in gensim loading setting, using the default from FB C++ code
        return 1
    elif field == 'ws':
        return model.window
    elif field == 'lr_update_rate':
        # This is skipped in gensim loading setting, using the default from FB C++ code
        return 100
    else:
        msg = 'Extraction of header field "' + field + '" from Gensim FastText object not implemmented.'
        raise NotImplementedError(msg)


def _args_save(fout, model, fb_fasttext_parameters):
    """
    Saves header with `model` parameters to the binary stream `fout` containing a model in the Facebook's
    native fastText `.bin` format.

    Name mimics original C++ implementation, see
    [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which model is saved
    model: gensim.models.fasttext.FastText
        saved model
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    """
    for field, field_type in _NEW_HEADER_FORMAT:
        if field in fb_fasttext_parameters:
            field_value = fb_fasttext_parameters[field]
        else:
            field_value = _get_field_from_model(model, field)
        fout.write(_conv_field_to_bytes(field_value, field_type))


def _dict_save(fout, model, encoding):
    """
    Saves the dictionary from `model` to the to the binary stream `fout` containing a model in the Facebook's
    native fastText `.bin` format.

    Name mimics the original C++ implementation
    [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which the dictionary from the model is saved
    model: gensim.models.fasttext.FastText
        the model that contains the dictionary to save
    encoding: str
        string encoding used in the output
    """

    # In the FB format the dictionary can contain two types of entries, i.e.
    # words and labels. The first two fields of the dictionary contain
    # the dictionary size (size_) and the number of words (nwords_).
    # In the unsupervised case we have only words (no labels). Hence both fields
    # are equal.

    fout.write(np.int32(len(model.wv)).tobytes())

    fout.write(np.int32(len(model.wv)).tobytes())

    # nlabels=0 <- no labels  we are in unsupervised mode
    fout.write(np.int32(0).tobytes())

    fout.write(np.int64(model.corpus_total_words).tobytes())

    # prunedidx_size_=-1, -1 value denotes no prunning index (prunning is only supported in supervised mode)
    fout.write(np.int64(-1))

    for word in model.wv.index_to_key:
        word_count = model.wv.get_vecattr(word, 'count')
        fout.write(word.encode(encoding))
        fout.write(_END_OF_WORD_MARKER)
        fout.write(np.int64(word_count).tobytes())
        fout.write(_DICT_WORD_ENTRY_TYPE_MARKER)

    # We are in unsupervised case, therefore pruned_idx is empty, so we do not need to write anything else


def _input_save(fout, model):
    """
    Saves word and ngram vectors from `model` to the binary stream `fout` containing a model in
    the Facebook's native fastText `.bin` format.

    Corresponding C++ fastText code:
    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which the vectors are saved
    model: gensim.models.fasttext.FastText
        the model that contains the vectors to save
    """
    vocab_n, vocab_dim = model.wv.vectors_vocab.shape
    ngrams_n, ngrams_dim = model.wv.vectors_ngrams.shape

    assert vocab_dim == ngrams_dim
    assert vocab_n == len(model.wv)
    assert ngrams_n == model.wv.bucket

    fout.write(struct.pack('@2q', vocab_n + ngrams_n, vocab_dim))
    fout.write(model.wv.vectors_vocab.tobytes())
    fout.write(model.wv.vectors_ngrams.tobytes())


def _output_save(fout, model):
    """
    Saves output layer of `model` to the binary stream `fout` containing a model in
    the Facebook's native fastText `.bin` format.

    Corresponding C++ fastText code:
    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)

    Parameters
    ----------
    fout: writeable binary stream
        the model that contains the output layer to save
    model: gensim.models.fasttext.FastText
        saved model
    """
    if model.hs:
        hidden_output = model.syn1
    if model.negative:
        hidden_output = model.syn1neg

    hidden_n, hidden_dim = hidden_output.shape
    fout.write(struct.pack('@2q', hidden_n, hidden_dim))
    fout.write(hidden_output.tobytes())


def _save_to_stream(model, fout, fb_fasttext_parameters, encoding):
    """
    Saves word embeddings to binary stream `fout` using the Facebook's native fasttext `.bin` format.

    Parameters
    ----------
    fout: file name or writeable binary stream
        stream to which the word embeddings are saved
    model: gensim.models.fasttext.FastText
        the model that contains the word embeddings to save
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    encoding: str
        encoding used in the output file
    """

    _sign_model(fout)
    _args_save(fout, model, fb_fasttext_parameters)
    _dict_save(fout, model, encoding)
    fout.write(struct.pack('@?', False))  # Save 'quant_', which is False for unsupervised models

    # Save words and ngrams vectors
    _input_save(fout, model)
    fout.write(struct.pack('@?', False))  # Save 'quot_', which is False for unsupervised models

    # Save output layers of the model
    _output_save(fout, model)


def save(model, fout, fb_fasttext_parameters, encoding):
    """
    Saves word embeddings to the Facebook's native fasttext `.bin` format.

    Parameters
    ----------
    fout: file name or writeable binary stream
        stream to which model is saved
    model: gensim.models.fasttext.FastText
        saved model
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    encoding: str
        encoding used in the output file

    Notes
    -----
    Unfortunately, there is no documentation of the Facebook's native fasttext `.bin` format

    This is just reimplementation of
    [FastText::saveModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)

    Based on v0.9.1, more precisely commit da2745fcccb848c7a225a7d558218ee4c64d5333

    Code follows the original C++ code naming.
    """

    if isinstance(fout, str):
        with open(fout, "wb") as fout_stream:
            _save_to_stream(model, fout_stream, fb_fasttext_parameters, encoding)
    else:
        _save_to_stream(model, fout, fb_fasttext_parameters, encoding)
