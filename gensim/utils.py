#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains various general utility functions."""

from __future__ import with_statement
from contextlib import contextmanager
import collections
import logging
import warnings

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq

import numpy as np
import numbers
import scipy.sparse

from six import iterkeys, iteritems, itervalues, u, string_types, unichr
from six.moves import xrange

from smart_open import smart_open

if sys.version_info[0] >= 3:
    unicode = str

logger = logging.getLogger(__name__)


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


def get_random_state(seed):
    """Generate :class:`numpy.random.RandomState` based on input seed.

    Parameters
    ----------
    seed : {None, int, array_like}
        Seed for random state.

    Returns
    -------
    :class:`numpy.random.RandomState`
        Random state.

    Raises
    ------
    AttributeError
        If seed is not {None, int, array_like}.

    Notes
    -----
    Method originally from [1]_ and written by @joshloyal.

    References
    ----------
    .. [1] https://github.com/maciejkula/glove-python

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a np.random.RandomState instance' % seed)


def synchronous(tlockname):
    """A decorator to place an instance-based lock around a method.

    Notes
    -----
    Adapted from [2]_

    References
    ----------
    .. [2] http://code.activestate.com/recipes/577105-synchronization-decorator-for-class-methods/

    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = getattr(self, tlockname)
            logger.debug("acquiring lock %r for %s", tlockname, func.__name__)

            with tlock:  # use lock as a context manager to perform safe acquire/release pairs
                logger.debug("acquired lock %r for %s", tlockname, func.__name__)
                result = func(self, *args, **kwargs)
                logger.debug("releasing lock %r for %s", tlockname, func.__name__)
                return result
        return _synchronizer
    return _synched


def file_or_filename(input):
    """Open file with `smart_open`.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Returns
    -------
    input : file-like object
        Opened file OR seek out to 0 byte if `input` is already file-like object.

    """
    if isinstance(input, string_types):
        # input was a filename: open as file
        return smart_open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        return input


@contextmanager
def open_file(input):
    """Provide "with-like" behaviour except closing the file object.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Yields
    -------
    file
        File-like object based on input (or input if this already file-like).

    """
    mgr = file_or_filename(input)
    exc = False
    try:
        yield mgr
    except Exception:
        # Handling any unhandled exceptions from the code nested in 'with' statement.
        exc = True
        if not isinstance(input, string_types) or not mgr.__exit__(*sys.exc_info()):
            raise
        # Try to introspect and silence errors.
    finally:
        if not exc and isinstance(input, string_types):
            mgr.__exit__(None, None, None)


def deaccent(text):
    """Remove accentuation from the given string.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Unicode string without accentuation.

    Examples
    --------
    >>> from gensim.utils import deaccent
    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'

    """
    if not isinstance(text, unicode):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = u('').join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def copytree_hardlink(source, dest):
    """Recursively copy a directory ala shutils.copytree, but hardlink files instead of copying.

    Parameters
    ----------
    source : str
        Path to source directory
    dest : str
        Path to destination directory

    Warnings
    --------
    Available on UNIX systems only.

    """
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2


def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    """Iteratively yield tokens as unicode strings, removing accent marks and optionally lowercasing string
    if any from `lowercase`, `to_lower`, `lower` set to True.

    Parameters
    ----------
    text : str
        Input string.
    lowercase : bool, optional
        If True - lowercase input string.
    deacc : bool, optional
        If True - remove accentuation from string by :func:`~gensim.utils.deaccent`.
    encoding : str, optional
        Encoding of input string, used as parameter for :func:`~gensim.utils.to_unicode`.
    errors : str, optional
        Error handling behaviour, used as parameter for :func:`~gensim.utils.to_unicode`.
    to_lower : bool, optional
        Same as `lowercase`.
    lower : bool, optional
        Same as `lowercase`.

    Yields
    ------
    str
        Contiguous sequences of alphabetic characters (no digits!), using :func:`~gensim.utils.simple_tokenize`

    Examples
    --------
    >>> from gensim.utils import tokenize
    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc=True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']

    """
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    """Tokenize input test using :const:`gensim.utils.PAT_ALPHABETIC`.

    Parameters
    ----------
    text : str
        Input text.

    Yields
    ------
    str
        Tokens from `text`.

    """
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    """Convert a document into a list of tokens (also with lowercase and optional de-accents),
    used :func:`~gensim.utils.tokenize`.

    Parameters
    ----------
    doc : str
        Input document.
    deacc : bool, optional
        If True - remove accentuation from string by :func:`~gensim.utils.deaccent`.
    min_len : int, optional
        Minimal length of token in result (inclusive).
    max_len : int, optional
        Maximal length of token in result (inclusive).

    Returns
    -------
    list of str
        Tokens extracted from `doc`.

    """
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert `text` to bytestring in utf8.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).
    encoding : str, optional
        Encoding of `text` for `unicode` function (python2 only).

    Returns
    -------
    str
        Bytestring in utf8.

    """

    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).
    encoding : str, optional
        Encoding of `text` for `unicode` function (python2 only).

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode


def call_on_class_only(*args, **kwargs):
    """Helper for raise `AttributeError` if method should be called from instance.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Raises
    ------
    AttributeError
        If `load` method are called on instance.

    """
    raise AttributeError('This method should be called on a class object.')


class SaveLoad(object):
    """Class which inherit from this class have save/load functions, which un/pickle them to disk.

    Warnings
    --------
    This uses pickle for de/serializing, so objects must not contain unpicklable attributes,
    such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """Load a previously saved object (using :meth:`~gensim.utils.SaveLoad.save`) from file.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str, optional
            Memory-map option.  If the object was saved with large arrays stored separately, you can load these arrays
            via mmap (shared memory) using `mmap='r'.
            If the file being loaded is compressed (either '.gz' or '.bz2'), then `mmap=None` **must be** set.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.save`

        Returns
        -------
        object
            Object loaded from `fname`.

        Raises
        ------
        IOError
            When methods are called on instance (should be called from class).

        """
        logger.info("loading %s object from %s", cls.__name__, fname)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        logger.info("loaded %s", fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """Loads any attributes that were stored specially, and gives the same opportunity
        to recursively included :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str
            Memory-map option.
        compress : bool
            Set to True if file is compressed.
        subname : str
            ...


        """
        def mmap_error(obj, filename):
            return IOError(
                'Cannot mmap compressed object %s in file %s. ' % (obj, filename) +
                'Use `load(fname, mmap=None)` or uncompress files manually.'
            )

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s", attrib, cfname, mmap)
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None", attrib)
            setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        (bool, function)
            First argument will be True if `fname` compressed.

        """
        compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
        return compress, lambda *args: '.'.join(args + (suffix,))

    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to file.

        Parameters
        ----------
        fname : str
            Path to file.
        separately : list, optional
            Iterable of attributes than need to store distinctly.
        sep_limit : int, optional
            Limit for separation.
        ignore : frozenset, optional
            Attributes that shouldn't be store.
        pickle_protocol : int, optional
            Protocol number for pickle.

        Notes
        -----
        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`

        """
        logger.info("saving %s object under %s, separately %s", self.__class__.__name__, fname, separately)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol,
                                       compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            # restore attribs handled specially
            for obj, asides in restores:
                for attrib, val in iteritems(asides):
                    setattr(obj, attrib, val)
        logger.info("saved %s", fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Output filename.
        separately : list or None
            Iterable of attributes than need to store distinctly
        sep_limit : int
            Limit for separation.
        ignore : iterable of str
            Attributes that shouldn't be store.
        pickle_protocol : int
            Protocol number for pickle.
        compress : bool
            If True - compress output with :func:`numpy.savez_compressed`.
        subname : function
            Produced by :meth:`~gensim.utils.SaveLoad._adapt_by_suffix`

        Returns
        -------
        list of (obj, {attrib: value, ...})
            Settings that the caller should use to restore each object's attributes that were set aside
            during the default :func:`~gensim.utils.pickle`.

        """
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in iteritems(self.__dict__):
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)

        # whatever's in `separately` or `ignore` at this point won't get pickled
        for attrib in separately + list(ignore):
            if hasattr(self, attrib):
                asides[attrib] = getattr(self, attrib)
                delattr(self, attrib)

        recursive_saveloads = []
        restores = []
        for attrib, val in iteritems(self.__dict__):
            if hasattr(val, '_save_specials'):  # better than 'isinstance(val, SaveLoad)' if IPython reloading
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore, pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))

                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(
                            subname(fname, attrib, 'sparse'),
                            data=val.data,
                            indptr=val.indptr,
                            indices=val.indices
                        )
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        # store array-less object
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s", attrib)
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except Exception:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to file.

        Parameters
        ----------
        fname_or_handle : str or file-like
            Path to output file or already opened file-like object. If the object is a file handle,
            no special array handling will be performed, all attributes will be saved to the same file.
        separately : list of str or None, optional
            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This avoids pickle memory errors and allows mmap'ing large arrays
            back on load efficiently.
            If list of str - this attributes will be stored in separate files, the automatic check
            is not performed in this case.
        sep_limit : int
            Limit for automatic separation.
        ignore : frozenset of str
            Attributes that shouldn't be serialize/store.
        pickle_protocol : int
            Protocol number for pickle.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object", self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore, pickle_protocol=pickle_protocol)


def identity(p):
    """Identity fnc, for flows that don't accept lambda (pickling etc).

    Parameters
    ----------
    p : object
        Input parameter.

    Returns
    -------
    object
        Same as `p`.

    """
    return p


def get_max_id(corpus):
    """Get the highest feature id that appears in the corpus.

    Parameters
    ----------
    corpus : iterable of iterable of (int, int)
        Collection of texts in BoW format.

    Returns
    ------
    int
        Highest feature id.

    Notes
    -----
    For empty `corpus` return -1.

    """
    maxid = -1
    for document in corpus:
        maxid = max(maxid, max([-1] + [fieldid for fieldid, _ in document]))  # [-1] to avoid exceptions from max(empty)
    return maxid


class FakeDict(object):
    """Objects of this class act as dictionaries that map integer->str(integer), for a specified
    range of integers <0, num_terms).

    This is meant to avoid allocating real dictionaries when `num_terms` is huge, which is a waste of memory.

    """

    def __init__(self, num_terms):
        """

        Parameters
        ----------
        num_terms : int
            Number of terms.

        """
        self.num_terms = num_terms

    def __str__(self):
        return "FakeDict(num_terms=%s)" % self.num_terms

    def __getitem__(self, val):
        if 0 <= val < self.num_terms:
            return str(val)
        raise ValueError("internal id out of bounds (%s, expected <0..%s))" % (val, self.num_terms))

    def iteritems(self):
        """Iterate over all keys and values.


        Yields
        ------
        (int, str)
            Pair of (id, token).

        """
        for i in xrange(self.num_terms):
            yield i, str(i)

    def keys(self):
        """Override the `dict.keys()`, which is used to determine the maximum internal id of a corpus,
        i.e. the vocabulary dimensionality.

        Returns
        -------
        list of int
            Highest id, packed in list.

        Warnings
        --------
        To avoid materializing the whole `range(0, self.num_terms)`,
        this returns the highest id = `[self.num_terms - 1]` only.

        """
        return [self.num_terms - 1]

    def __len__(self):
        return self.num_terms

    def get(self, val, default=None):
        if 0 <= val < self.num_terms:
            return str(val)
        return default


def dict_from_corpus(corpus):
    """Scan corpus for all word ids that appear in it, then construct a mapping
    which maps each `word_id` -> `str(word_id)`.

    Parameters
    ----------
    corpus : iterable of iterable of (int, int)
        Collection of texts in BoW format.

    Returns
    ------
    id2word : :class:`~gensim.utils.FakeDict`
        "Fake" mapping which maps each `word_id` -> `str(word_id)`.

    Warnings
    --------
    This function is used whenever *words* need to be displayed (as opposed to just their ids)
    but no `word_id` -> `word` mapping was provided. The resulting mapping only covers words actually
    used in the corpus, up to the highest `word_id` found.

    """
    num_terms = 1 + get_max_id(corpus)
    id2word = FakeDict(num_terms)
    return id2word


def is_corpus(obj):
    """Check whether `obj` is a corpus.

    Parameters
    ----------
    obj : object
        Something `iterable of iterable` that contains (int, int).

    Return
    ------
    (bool, object)
        Pair of (is_corpus, `obj`), is_corpus True if `obj` is corpus.

    Warnings
    --------
    An "empty" corpus (empty input sequence) is ambiguous, so in this case
    the result is forcefully defined as (False, `obj`).

    """
    try:
        if 'Corpus' in obj.__class__.__name__:  # the most common case, quick hack
            return True, obj
    except Exception:
        pass
    try:
        if hasattr(obj, 'next') or hasattr(obj, '__next__'):
            # the input is an iterator object, meaning once we call next()
            # that element could be gone forever. we must be careful to put
            # whatever we retrieve back again
            doc1 = next(obj)
            obj = itertools.chain([doc1], obj)
        else:
            doc1 = next(iter(obj))  # empty corpus is resolved to False here
        if len(doc1) == 0:  # sparse documents must have a __len__ function (list, tuple...)
            return True, obj  # the first document is empty=>assume this is a corpus

        # if obj is a 1D numpy array(scalars) instead of 2-tuples, it resolves to False here
        id1, val1 = next(iter(doc1))
        id1, val1 = int(id1), float(val1)  # must be a 2-tuple (integer, float)
    except Exception:
        return False, obj
    return True, obj


def get_my_ip():
    """Try to obtain our external ip (from the Pyro4 nameserver's point of view)

    Returns
    -------
    str
        IP address.

    Warnings
    --------
    This tries to sidestep the issue of bogus `/etc/hosts` entries and other local misconfiguration,
    which often mess up hostname resolution.
    If all else fails, fall back to simple `socket.gethostbyname()` lookup.

    """
    import socket
    try:
        from Pyro4.naming import locateNS
        # we know the nameserver must exist, so use it as our anchor point
        ns = locateNS()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ns._pyroUri.host, ns._pyroUri.port))
        result, port = s.getsockname()
    except Exception:
        try:
            # see what ifconfig says about our default interface
            import commands
            result = commands.getoutput("ifconfig").split("\n")[1].split()[1][5:]
            if len(result.split('.')) != 4:
                raise Exception()
        except Exception:
            # give up, leave the resolution to gethostbyname
            result = socket.gethostbyname(socket.gethostname())
    return result


class RepeatCorpus(SaveLoad):
    """Wrap a `corpus` as another corpus of length `reps`. This is achieved by repeating documents from `corpus`
    over and over again, until the requested length `len(result) == reps` is reached.
    Repetition is done on-the-fly=efficiently, via `itertools`.

    Examples
    --------
    >>> from gensim.utils import RepeatCorpus
    >>>
    >>> corpus = [[(1, 2)], []] # 2 documents
    >>> list(RepeatCorpus(corpus, 5)) # repeat 2.5 times to get 5 documents
    [[(1, 2)], [], [(1, 2)], [], [(1, 2)]]

    """

    def __init__(self, corpus, reps):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.
        reps : int
            Number of repeats for documents from corpus.

        """
        self.corpus = corpus
        self.reps = reps

    def __iter__(self):
        return itertools.islice(itertools.cycle(self.corpus), self.reps)


class RepeatCorpusNTimes(SaveLoad):
    """Wrap a `corpus` and repeat it `n` times.

    Examples
    --------
    >>> from gensim.utils import RepeatCorpusNTimes
    >>>
    >>> corpus = [[(1, 0.5)], []]
    >>> list(RepeatCorpusNTimes(corpus, 3)) # repeat 3 times
    [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]

    """

    def __init__(self, corpus, n):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.
        n : int
            Number of repeats for corpus.

        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in xrange(self.n):
            for document in self.corpus:
                yield document


class ClippedCorpus(SaveLoad):
    """Wrap a `corpus` and return `max_doc` element from it"""

    def __init__(self, corpus, max_docs=None):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.
        max_docs : int
            Maximal number of documents in result corpus.

        Warnings
        --------
        Any documents after `max_docs` are ignored. This effectively limits the length of the returned corpus
        to <= `max_docs`. Set `max_docs=None` for "no limit", effectively wrapping the entire input corpus.

        """
        self.corpus = corpus
        self.max_docs = max_docs

    def __iter__(self):
        return itertools.islice(self.corpus, self.max_docs)

    def __len__(self):
        return min(self.max_docs, len(self.corpus))


class SlicedCorpus(SaveLoad):
    """Wrap `corpus` and return the slice of it"""

    def __init__(self, corpus, slice_):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.
        slice_ : slice or iterable
            Slice for `corpus`

        Notes
        -----
        Negative slicing can only be used if the corpus is indexable, otherwise, the corpus will be iterated over.
        Slice can also be a np.ndarray to support fancy indexing.

        Calculating the size of a SlicedCorpus is expensive when using a slice as the corpus has
        to be iterated over once. Using a list or np.ndarray does not have this drawback, but consumes more memory.

        """
        self.corpus = corpus
        self.slice_ = slice_
        self.length = None

    def __iter__(self):
        if hasattr(self.corpus, 'index') and len(self.corpus.index) > 0:
            return (self.corpus.docbyoffset(i) for i in self.corpus.index[self.slice_])
        return itertools.islice(self.corpus, self.slice_.start, self.slice_.stop, self.slice_.step)

    def __len__(self):
        # check cached length, calculate if needed
        if self.length is None:
            if isinstance(self.slice_, (list, np.ndarray)):
                self.length = len(self.slice_)
            elif isinstance(self.slice_, slice):
                (start, end, step) = self.slice_.indices(len(self.corpus.index))
                diff = end - start
                self.length = diff // step + (diff % step > 0)
            else:
                self.length = sum(1 for x in self)

        return self.length


def safe_unichr(intval):
    """

    Parameters
    ----------
    intval : int
        Integer code of character

    Returns
    -------
    string
        Unicode string of character

    """
    try:
        return unichr(intval)
    except ValueError:
        # ValueError: unichr() arg not in range(0x10000) (narrow Python build)
        s = "\\U%08x" % intval
        # return UTF16 surrogate pair
        return s.decode('unicode-escape')


def decode_htmlentities(text):
    """Decode HTML entities in text, coded as hex, decimal or named.
    This function from [3]_.

    Parameters
    ----------
    text : str
        Input html text.

    Examples
    --------
    >>> from gensim.utils import decode_htmlentities
    >>>
    >>> u = u'E tu vivrai nel terrore - L&#x27;aldil&#xE0; (1981)'
    >>> print(decode_htmlentities(u).encode('UTF-8'))
    E tu vivrai nel terrore - L'aldilà (1981)
    >>> print(decode_htmlentities("l&#39;eau"))
    l'eau
    >>> print(decode_htmlentities("foo &lt; bar"))
    foo < bar

    References
    ----------
    .. [3] http://github.com/sku/python-twitter-ircbot/blob/321d94e0e40d0acc92f5bf57d126b57369da70de/html_decode.py

    """
    def substitute_entity(match):
        try:
            ent = match.group(3)
            if match.group(1) == "#":
                # decoding by number
                if match.group(2) == '':
                    # number is in decimal
                    return safe_unichr(int(ent))
                elif match.group(2) in ['x', 'X']:
                    # number is in hex
                    return safe_unichr(int(ent, 16))
            else:
                # they were using a name
                cp = n2cp.get(ent)
                if cp:
                    return safe_unichr(cp)
                else:
                    return match.group()
        except Exception:
            # in case of errors, return original input
            return match.group()

    return RE_HTML_ENTITY.sub(substitute_entity, text)


def chunkize_serial(iterable, chunksize, as_numpy=False, dtype=np.float32):
    """Give elements from the iterable in `chunksize`-ed lists.
    The last returned element may be smaller (if length of collection is not divisible by `chunksize`).

    Parameters
    ----------
    iterable : iterable of object
        Any iterable.
    chunksize : int
        Size of chunk from result.
    as_numpy : bool, optional
        If True - yield `np.ndarray`, otherwise - list

    Yields
    ------
    list of object OR np.ndarray
        Groups based on `iterable`

    Examples
    --------
    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc, dtype=dtype) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()


grouper = chunkize_serial


class InputQueue(multiprocessing.Process):
    def __init__(self, q, corpus, chunksize, maxsize, as_numpy):
        super(InputQueue, self).__init__()
        self.q = q
        self.maxsize = maxsize
        self.corpus = corpus
        self.chunksize = chunksize
        self.as_numpy = as_numpy

    def run(self):
        it = iter(self.corpus)
        while True:
            chunk = itertools.islice(it, self.chunksize)
            if self.as_numpy:
                # HACK XXX convert documents to numpy arrays, to save memory.
                # This also gives a scipy warning at runtime:
                # "UserWarning: indices array has non-integer dtype (float64)"
                wrapped_chunk = [[np.asarray(doc) for doc in chunk]]
            else:
                wrapped_chunk = [list(chunk)]

            if not wrapped_chunk[0]:
                self.q.put(None, block=True)
                break

            try:
                qsize = self.q.qsize()
            except NotImplementedError:
                qsize = '?'
            logger.debug("prepared another chunk of %i documents (qsize=%s)", len(wrapped_chunk[0]), qsize)
            self.q.put(wrapped_chunk.pop(), block=True)


if os.name == 'nt':
    warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")

    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        """Split `corpus` into smaller chunks, used :func:`~gensim.utils.chunkize_serial`.

        Parameters
        ----------
        corpus : iterable of object
            Any iterable object.
        chunksize : int
            Size of chunk from result.
        maxsize : int, optional
            THIS PARAMETER IGNORED.
        as_numpy : bool, optional
            If True - yield `np.ndarray`, otherwise - list

        Yields
        ------
        list of object OR np.ndarray
            Groups based on `iterable`

        """
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk
else:
    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        """Split `corpus` into smaller chunks, used :func:`~gensim.utils.chunkize_serial`.

        Parameters
        ----------
        corpus : iterable of object
            Any iterable object.
        chunksize : int
            Size of chunk from result.
        maxsize : int, optional
            THIS PARAMETER IGNORED.
        as_numpy : bool, optional
            If True - yield `np.ndarray`, otherwise - list

        Notes
        -----
        Each chunk is of length `chunksize`, except the last one which may be smaller.
        A once-only input stream (`corpus` from a generator) is ok, chunking is done efficiently via itertools.

        If `maxsize > 1`, don't wait idly in between successive chunk `yields`, but rather keep filling a short queue
        (of size at most `maxsize`) with forthcoming chunks in advance. This is realized by starting a separate process,
        and is meant to reduce I/O delays, which can be significant when `corpus` comes from a slow medium (like HDD).

        If `maxsize == 0`, don't fool around with parallelism and simply yield the chunksize
        via :func:`~gensim.utils.chunkize_serial` (no I/O optimizations).

        Yields
        ------
        list of object OR np.ndarray
            Groups based on `iterable`

        """
        assert chunksize > 0

        if maxsize > 0:
            q = multiprocessing.Queue(maxsize=maxsize)
            worker = InputQueue(q, corpus, chunksize, maxsize=maxsize, as_numpy=as_numpy)
            worker.daemon = True
            worker.start()
            while True:
                chunk = [q.get(block=True)]
                if chunk[0] is None:
                    break
                yield chunk.pop()
        else:
            for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
                yield chunk


def smart_extension(fname, ext):
    """Generate filename with `ext`.

    Parameters
    ----------
    fname : str
        Path to file.
    ext : str
        File extension.

    Returns
    -------
    str
        New path to file with `ext`.

    """
    fname, oext = os.path.splitext(fname)
    if oext.endswith('.bz2'):
        fname = fname + oext[:-4] + ext + '.bz2'
    elif oext.endswith('.gz'):
        fname = fname + oext[:-3] + ext + '.gz'
    else:
        fname = fname + oext + ext

    return fname


def pickle(obj, fname, protocol=2):
    """Pickle object `obj` to file `fname`.

    Parameters
    ----------
    obj : object
        Any python object.
    fname : str
        Path to pickle file.
    protocol : int, optional
        Pickle protocol number, default is 2 to support compatible across python 2.x and 3.x.

    """
    with smart_open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load object from `fname`.

    Parameters
    ----------
    fname : str
        Path to pickle file.

    Returns
    -------
    object
        Python object loaded from `fname`.

    """
    with smart_open(fname, 'rb') as f:
        # Because of loading from S3 load can't be used (missing readline in smart_open)
        if sys.version_info > (3, 0):
            return _pickle.load(f, encoding='latin1')
        else:
            return _pickle.loads(f.read())


def revdict(d):
    """Reverse a dictionary mapping, i.e. `{1: 2, 3: 4}` -> `{2: 1, 4: 3}`.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    dict
        Reversed dictionary mapping.

    Notes
    -----
    When two keys map to the same value, only one of them will be kept in the result (which one is kept is arbitrary).

    Examples
    --------
    >>> from gensim.utils import revdict
    >>> d = {1: 2, 3: 4}
    >>> revdict(d)
    {2: 1, 4: 3}

    """
    return {v: k for (k, v) in iteritems(dict(d))}


def deprecated(reason):
    """Decorator which can be used to mark functions as deprecated.

    Parameters
    ----------
    reason : str
        Reason of deprecation.

    Returns
    -------
    function
        Decorated function

    Notes
    -----
    It will result in a warning being emitted when the function is used, base code from [4]_.

    References
    ----------
    .. [4] https://stackoverflow.com/a/40301488/8001386

    """
    if isinstance(reason, string_types):
        def decorator(func):
            fmt = "Call to deprecated `{name}` ({reason})."

            @wraps(func)
            def new_func1(*args, **kwargs):
                warnings.warn(
                    fmt.format(name=func.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)

            return new_func1
        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason
        fmt = "Call to deprecated `{name}`."

        @wraps(func)
        def new_func2(*args, **kwargs):
            warnings.warn(
                fmt.format(name=func.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func2

    else:
        raise TypeError(repr(type(reason)))


@deprecated("Function will be removed in 4.0.0")
def toptexts(query, texts, index, n=10):
    """
    Debug fnc to help inspect the top `n` most similar documents (according to a
    similarity index `index`), to see if they are actually related to the query.

    Parameters
    ----------
    query : list
        vector OR BoW (list of tuples)
    texts : str
        object that can return something insightful for each document via `texts[docid]`,
        such as its fulltext or snippet.
    index : any
        a class from gensim.similarity.docsim

    Return
    ------
    list
        a list of 3-tuples (docid, doc's similarity to the query, texts[docid])

    """
    sims = index[query]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    return [(topid, topcosine, texts[topid]) for topid, topcosine in sims[:n]]  # only consider top-n most similar docs


def randfname(prefix='gensim'):
    """Generate path with random filename/

    Parameters
    ----------
    prefix : str
        Prefix of filename.

    Returns
    -------
    str
        Full path with random filename (in temporary folder).

    """
    randpart = hex(random.randint(0, 0xffffff))[2:]
    return os.path.join(tempfile.gettempdir(), prefix + randpart)


@deprecated("Function will be removed in 4.0.0")
def upload_chunked(server, docs, chunksize=1000, preprocess=None):
    """Memory-friendly upload of documents to a SimServer (or Pyro SimServer proxy).
    Notes
    -----
    Use this function to train or index large collections -- avoid sending the
    entire corpus over the wire as a single Pyro in-memory object. The documents
    will be sent in smaller chunks, of `chunksize` documents each.
    """
    start = 0
    for chunk in grouper(docs, chunksize):
        end = start + len(chunk)
        logger.info("uploading documents %i-%i", start, end - 1)
        if preprocess is not None:
            pchunk = []
            for doc in chunk:
                doc['tokens'] = preprocess(doc['text'])
                del doc['text']
                pchunk.append(doc)
            chunk = pchunk
        server.buffer(chunk)
        start = end


def getNS(host=None, port=None, broadcast=True, hmac_key=None):
    """Get a Pyro4 name server proxy.

    Parameters
    ----------
    host : str, optional
        Hostname of ns.
    port : int, optional
        Port of ns.
    broadcast : bool, optional
        If True - use broadcast mechanism (i.e. all Pyro nodes in local network), not otherwise.
    hmac_key : str, optional
        Private key.

    Raises
    ------
    RuntimeError
        when Pyro name server is not found

    Returns
    -------
    :class:`Pyro4.core.Proxy`
        Proxy from Pyro4.

    """
    import Pyro4
    try:
        return Pyro4.locateNS(host, port, broadcast, hmac_key)
    except Pyro4.errors.NamingError:
        raise RuntimeError("Pyro name server not found")


def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None, ns_conf=None):
    """Register object with name server (starting the name server if not running
    yet) and block until the daemon is terminated. The object is registered under
    `name`, or `name`+ some random suffix if `random_suffix` is set.

    """
    if ns_conf is None:
        ns_conf = {}
    if random_suffix:
        name += '.' + hex(random.randint(0, 0xffffff))[2:]

    import Pyro4
    with getNS(**ns_conf) as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            # register server for remote access
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')", name, uri)
            daemon.requestLoop()


def has_pattern():
    """Check that `pattern` [5]_ package already installed.

    Returns
    -------
    bool
        True if `pattern` installed, False otherwise.

    References
    ----------
    .. [5] https://github.com/clips/pattern

    """
    try:
        from pattern.en import parse  # noqa:F401
        return True
    except ImportError:
        return False


def lemmatize(content, allowed_tags=re.compile(r'(NN|VB|JJ|RB)'), light=False,
              stopwords=frozenset(), min_length=2, max_length=15):
    """Use the English lemmatizer from `pattern` [5]_ to extract UTF8-encoded tokens in
    their base form=lemma, e.g. "are, is, being" -> "be" etc.
    This is a smarter version of stemming, taking word context into account.

    Parameters
    ----------
    content : str
        Input string
    allowed_tags : :class:`_sre.SRE_Pattern`, optional
        Compiled regexp to select POS that will be used.
        Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).
    light : bool, optional
        DEPRECATED FLAG, DOESN'T SUPPORT BY `pattern`.
    stopwords : frozenset
        Set of words that will be removed from output.
    min_length : int
        Minimal token length in output (inclusive).
    max_length : int
        Maximal token length in output (inclusive).

    Returns
    -------
    list of str
        List with tokens with POS tag.

    Warnings
    --------
    This function is only available when the optional 'pattern' package is installed.

    Examples
    --------
    >>> from gensim.utils import lemmatize
    >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
    ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']
    >>> lemmatize('The study ranks high.')
    ['study/NN', 'rank/VB', 'high/JJ']
    >>> lemmatize('The ranks study hard.')
    ['rank/NN', 'study/VB', 'hard/RB']

    """
    if not has_pattern():
        raise ImportError(
            "Pattern library is not installed. Pattern library is needed in order to use lemmatize function"
        )
    from pattern.en import parse

    if light:
        import warnings
        warnings.warn("The light flag is no longer supported by pattern.")

    # tokenization in `pattern` is weird; it gets thrown off by non-letters,
    # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
    # FIXME this throws away all fancy parsing cues, including sentence structure,
    # abbreviations etc.
    content = u(' ').join(tokenize(content, lower=True, errors='ignore'))

    parsed = parse(content, lemmata=True, collapse=False)
    result = []
    for sentence in parsed:
        for token, tag, _, _, lemma in sentence:
            if min_length <= len(lemma) <= max_length and not lemma.startswith('_') and lemma not in stopwords:
                if allowed_tags.match(tag):
                    lemma += "/" + tag[:2]
                    result.append(lemma.encode('utf8'))
    return result


def mock_data_row(dim=1000, prob_nnz=0.5, lam=1.0):
    """Create a random gensim BoW vector.

    Parameters
    ----------
    dim : int, optional
        Dimension of vector.
    prob_nnz : float, optional
        Probability of each coordinate will be nonzero, will be drawn from Poisson distribution.
    lam : float, optional
        Parameter for Poisson distribution.

    Returns
    -------
    list of (int, float)
        Vector in BoW format.

    """
    nnz = np.random.uniform(size=(dim,))
    return [(i, float(np.random.poisson(lam=lam) + 1.0)) for i in xrange(dim) if nnz[i] < prob_nnz]


def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    """Create a random gensim-style corpus (BoW), used :func:`~gensim.utils.mock_data_row`.

    Parameters
    ----------
    n_items : int
        Size of corpus
    dim : int
        Dimension of vector, used for :func:`~gensim.utils.mock_data_row`.
    prob_nnz : float, optional
        Probability of each coordinate will be nonzero, will be drawn from Poisson distribution,
        used for :func:`~gensim.utils.mock_data_row`.
    lam : float, optional
        Parameter for Poisson distribution, used for :func:`~gensim.utils.mock_data_row`.

    Returns
    -------
    list of list of (int, float)
        Gensim-style corpus.

    """
    return [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam) for _ in xrange(n_items)]


def prune_vocab(vocab, min_reduce, trim_rule=None):
    """Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.

    Modifies `vocab` in place, returns the sum of all counts that were pruned.
    Parameters
    ----------
    vocab : dict
        Input dictionary.
    min_reduce : int
        Frequency threshold for tokens in `vocab`.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_reduce`.

    Returns
    -------
    result : int
        Sum of all counts that were pruned.

    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    logger.info(
        "pruned out %i tokens with count <=%i (before %i, after %i)",
        old_len - len(vocab), min_reduce, old_len, len(vocab)
    )
    return result


def trim_vocab_by_freq(vocab, topk, trim_rule=None):
    """Retain `topk` most frequent words in `vocab`.
    If there are more words with the same frequency as `topk`-th one, they will be kept.
    Modifies `vocab` in place, returns nothing.

    Parameters
    ----------
    vocab : dict
        Input dictionary.
    topk : int
        Number of words with highest frequencies to keep.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_count`.

    """
    if topk >= len(vocab):
        return

    min_count = heapq.nlargest(topk, itervalues(vocab))[-1]
    prune_vocab(vocab, min_count, trim_rule=trim_rule)


def merge_counts(dict1, dict2):
    """Merge `dict1` of (word, freq1) and `dict2` of (word, freq2) into `dict1` of (word, freq1+freq2).
    Parameters
    ----------
    dict1 : dict of (str, int)
        First dictionary.
    dict2 : dict of (str, int)
        Second dictionary.
    Returns
    -------
    result : dict
        Merged dictionary with sum of frequencies as values.
    """
    for word, freq in iteritems(dict2):
        if word in dict1:
            dict1[word] += freq
        else:
            dict1[word] = freq

    return dict1


def qsize(queue):
    """Get the (approximate) queue size where available.

    Parameters
    ----------
    queue : :class:`queue.Queue`
        Input queue.

    Returns
    -------
    int
        Queue size, -1 if `qsize` method isn't implemented (OS X).

    """
    try:
        return queue.qsize()
    except NotImplementedError:
        # OS X doesn't support qsize
        return -1


RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2


def keep_vocab_item(word, count, min_count, trim_rule=None):
    """Check that should we keep `word` in vocab or remove.

    Parameters
    ----------
    word : str
        Input word.
    count : int
        Number of times that word contains in corpus.
    min_count : int
        Frequency threshold for `word`.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_reduce`.

    Returns
    -------
    bool
        True if `word` should stay, False otherwise.

    """
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res


def check_output(stdout=subprocess.PIPE, *popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.
    Backported from Python 2.7 as it's implemented as pure python on stdlib + small modification.
    Widely used for :mod:`gensim.models.wrappers`.

    Very similar with [6]_

    Examples
    --------
    >>> from gensim.utils import check_output
    >>> check_output(args=['echo', '1'])
    '1\n'

    Raises
    ------
    KeyboardInterrupt
        If Ctrl+C pressed.

    References
    ----------
    .. [6] https://docs.python.org/2/library/subprocess.html#subprocess.check_output

    """
    try:
        logger.debug("COMMAND: %s %s", popenargs, kwargs)
        process = subprocess.Popen(stdout=stdout, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output
    except KeyboardInterrupt:
        process.terminate()
        raise


def sample_dict(d, n=10, use_random=True):
    """Pick `n` items from dictionary `d`.

    Parameters
    ----------
    d : dict
        Input dictionary.
    n : int, optional
        Number of items that will be picked.
    use_random : bool, optional
        If True - pick items randomly, otherwise - according to natural dict iteration.

    Returns
    -------
    list of (object, object)
        Picked items from dictionary, represented as list.

    """
    selected_keys = random.sample(list(d), min(len(d), n)) if use_random else itertools.islice(iterkeys(d), n)
    return [(key, d[key]) for key in selected_keys]


def strided_windows(ndarray, window_size):
    """Produce a numpy.ndarray of windows, as from a sliding window.

    Parameters
    ----------
    ndarray : numpy.ndarray
        Input array
    window_size : int
        Sliding window size.

    Returns
    -------
    numpy.ndarray
        Subsequences produced by sliding a window of the given size over the `ndarray`.
        Since this uses striding, the individual arrays are views rather than copies of `ndarray`.
        Changes to one view modifies the others and the original.

    Examples
    --------
    >>> from gensim.utils import strided_windows
    >>> strided_windows(np.arange(5), 2)
    array([[0, 1],
           [1, 2],
           [2, 3],
           [3, 4]])
    >>> strided_windows(np.arange(10), 5)
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6],
           [3, 4, 5, 6, 7],
           [4, 5, 6, 7, 8],
           [5, 6, 7, 8, 9]])

    """
    ndarray = np.asarray(ndarray)
    if window_size == ndarray.shape[0]:
        return np.array([ndarray])
    elif window_size > ndarray.shape[0]:
        return np.ndarray((0, 0))

    stride = ndarray.strides[0]
    return np.lib.stride_tricks.as_strided(
        ndarray, shape=(ndarray.shape[0] - window_size + 1, window_size),
        strides=(stride, stride))


def iter_windows(texts, window_size, copy=False, ignore_below_size=True, include_doc_num=False):
    """Produce a generator over the given texts using a sliding window of `window_size`.
    The windows produced are views of some subsequence of a text.
    To use deep copies instead, pass `copy=True`.


    Parameters
    ----------
    texts : list of str
        List of string sentences.
    window_size : int
        Size of sliding window.
    copy : bool, optional
        If True - produce deep copies.
    ignore_below_size : bool, optional
        If True - ignore documents that are not at least `window_size` in length.
    include_doc_num : bool, optional
        If True - will be yield doc_num too.

    """
    for doc_num, document in enumerate(texts):
        for window in _iter_windows(document, window_size, copy, ignore_below_size):
            if include_doc_num:
                yield (doc_num, window)
            else:
                yield window


def _iter_windows(document, window_size, copy=False, ignore_below_size=True):
    doc_windows = strided_windows(document, window_size)
    if doc_windows.shape[0] == 0:
        if not ignore_below_size:
            yield document.copy() if copy else document
    else:
        for doc_window in doc_windows:
            yield doc_window.copy() if copy else doc_window


def flatten(nested_list):
    """Recursively flatten out a nested list.

    Parameters
    ----------
    nested_list : list
        Possibly nested list.

    Returns
    -------
    list
        Flattened version of input, where any list elements have been unpacked into the top-level list
        in a recursive fashion.

    """
    return list(lazy_flatten(nested_list))


def lazy_flatten(nested_list):
    """Lazy version of :func:`~gensim.utils.flatten`.

    Parameters
    ----------
    nested_list : list
        Possibly nested list.

    Yields
    ------
    object
        Element of list

    """
    for el in nested_list:
        if isinstance(el, collections.Iterable) and not isinstance(el, string_types):
            for sub in flatten(el):
                yield sub
        else:
            yield el
