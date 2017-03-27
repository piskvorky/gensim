#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains various general utility functions.
"""

from __future__ import with_statement

import logging, warnings

logger = logging.getLogger(__name__)

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
from functools import wraps  # for `synchronous` function lock
import multiprocessing
import shutil
import sys
from contextlib import contextmanager
import subprocess

import numpy as np
import numbers
import scipy.sparse

if sys.version_info[0] >= 3:
    unicode = str

from six import iterkeys, iteritems, u, string_types, unichr
from six.moves import xrange

try:
    from smart_open import smart_open
except ImportError:
    logger.info("smart_open library not found; falling back to local-filesystem-only")

    def make_closing(base, **attrs):
        """
        Add support for `with Base(attrs) as fout:` to the base class if it's missing.
        The base class' `close()` method will be called on context exit, to always close the file properly.

        This is needed for gzip.GzipFile, bz2.BZ2File etc in older Pythons (<=2.6), which otherwise
        raise "AttributeError: GzipFile instance has no attribute '__exit__'".

        """
        if not hasattr(base, '__enter__'):
            attrs['__enter__'] = lambda self: self
        if not hasattr(base, '__exit__'):
            attrs['__exit__'] = lambda self, type, value, traceback: self.close()
        return type('Closing' + base.__name__, (base, object), attrs)

    def smart_open(fname, mode='rb'):
        _, ext = os.path.splitext(fname)
        if ext == '.bz2':
            from bz2 import BZ2File
            return make_closing(BZ2File)(fname, mode)
        if ext == '.gz':
            from gzip import GzipFile
            return make_closing(GzipFile)(fname, mode)
        return open(fname, mode)


PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


def get_random_state(seed):
     """ Turn seed into a np.random.RandomState instance.

         Method originally from maciejkula/glove-python, and written by @joshloyal
     """
     if seed is None or seed is np.random:
         return np.random.mtrand._rand
     if isinstance(seed, (numbers.Integral, np.integer)):
         return np.random.RandomState(seed)
     if isinstance(seed, np.random.RandomState):
        return seed
     raise ValueError('%r cannot be used to seed a np.random.RandomState instance' % seed)


def synchronous(tlockname):
    """
    A decorator to place an instance-based lock around a method.

    Adapted from http://code.activestate.com/recipes/577105-synchronization-decorator-for-class-methods/
    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = getattr(self, tlockname)
            logger.debug("acquiring lock %r for %s" % (tlockname, func.__name__))

            with tlock: # use lock as a context manager to perform safe acquire/release pairs
                logger.debug("acquired lock %r for %s" % (tlockname, func.__name__))
                result = func(self, *args, **kwargs)
                logger.debug("releasing lock %r for %s" % (tlockname, func.__name__))
                return result
        return _synchronizer
    return _synched


class NoCM(object):
    def acquire(self):
        pass
    def release(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
nocm = NoCM()


@contextmanager
def file_or_filename(input):
    """
    Return a file-like object ready to be read from the beginning. `input` is either
    a filename (gz/bz2 also supported) or a file-like object supporting seek.

    """
    if isinstance(input, string_types):
        # input was a filename: open as file
        yield smart_open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        yield input


def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.

    Return input string with accents removed, as unicode.

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
    """
    Recursively copy a directory ala shutils.copytree, but hardlink files
    instead of copying. Available on UNIX systems only.
    """
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2


def tokenize(text, lowercase=False, deacc=False, errors="strict", to_lower=False, lower=False):
    """
    Iteratively yield tokens as unicode strings, removing accent marks
    and optionally lowercasing the unidoce string by assigning True
    to one of the parameters, lowercase, to_lower, or lower.

    Input text may be either unicode or utf8-encoded byte string.

    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).

    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']

    """
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    """
    Convert a document into a list of tokens.

    This lowercases, tokenizes, de-accents (optional). -- the output are final
    tokens = unicode strings, that won't be processed any further.

    """
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)
to_unicode = any2unicode

def call_on_class_only(*args, **kwargs):
    """Raise exception when load methods are called on instance"""
    raise AttributeError('This method should be called on a class object.')


class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions, which un/pickle
    them to disk.

    This uses pickle for de/serializing, so objects must not contain
    unpicklable attributes, such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """
        Load a previously saved object from file (also see `save`).

        If the object was saved with large arrays stored separately, you can load
        these arrays via mmap (shared memory) using `mmap='r'`. Default: don't use
        mmap, load large arrays as normal objects.

        If the file being loaded is compressed (either '.gz' or '.bz2'), then
        `mmap=None` must be set.  Load will raise an `IOError` if this condition
        is encountered.

        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        logger.info("loaded %s", fname)
        if cls.__name__ == 'Similarity':
            if obj.output_prefix is not fname:
                obj.output_prefix = os.path.join(fname[: fname.rfind("shard")] , 'shard' ,'') # '' to get leading slash
                obj.check_moved()
        return obj


    def _load_specials(self, fname, mmap, compress, subname):
        """
        Loads any attributes that were stored specially, and gives the same
        opportunity to recursively included SaveLoad instances.

        """
        mmap_error = lambda x, y: IOError(
            'Cannot mmap compressed object %s in file %s. ' % (x, y) +
            'Use `load(fname, mmap=None)` or uncompress files manually.')

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s" % (
                attrib, cfname, mmap))
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))
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
            logger.info("setting ignored attribute %s to None" % (attrib))
            setattr(self, attrib, None)


    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula"""
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            compress = True
            subname = lambda *args: '.'.join(list(args) + ['npz'])
        else:
            compress = False
            subname = lambda *args: '.'.join(list(args) + ['npy'])
        return (compress, subname)


    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2,
                    ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        logger.info(
            "saving %s object under %s, separately %s" % (
                self.__class__.__name__, fname, separately))

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
        """
        Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves SaveLoad instances.

        Returns a list of (obj, {attrib: value, ...}) settings that the caller
        should use to restore each object's attributes that were set aside
        during the default pickle().

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
                cfname = '.'.join((fname,attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore,
                                                   pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))

                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        np.savez_compressed(subname(fname, attrib, 'sparse'),
                                               data=val.data,
                                               indptr=val.indptr,
                                               indices=val.indices)
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
                    logger.info("not storing attribute %s" % (attrib))
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]


    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2,
             ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        `fname_or_handle` is either a string specifying the file name to
        save to, or an open file-like object which can be written to. If
        the object is a file handle, no special array handling will be
        performed; all attributes will be saved to the same file.

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object" % self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore,
                             pickle_protocol=pickle_protocol)
#endclass SaveLoad


def identity(p):
    """Identity fnc, for flows that don't accept lambda (pickling etc)."""
    return p


def get_max_id(corpus):
    """
    Return the highest feature id that appears in the corpus.

    For empty corpora (no features at all), return -1.

    """
    maxid = -1
    for document in corpus:
        maxid = max(maxid, max([-1] + [fieldid for fieldid, _ in document])) # [-1] to avoid exceptions from max(empty)
    return maxid


class FakeDict(object):
    """
    Objects of this class act as dictionaries that map integer->str(integer), for
    a specified range of integers <0, num_terms).

    This is meant to avoid allocating real dictionaries when `num_terms` is huge, which
    is a waste of memory.

    """
    def __init__(self, num_terms):
        self.num_terms = num_terms


    def __str__(self):
        return "FakeDict(num_terms=%s)" % self.num_terms


    def __getitem__(self, val):
        if 0 <= val < self.num_terms:
            return str(val)
        raise ValueError("internal id out of bounds (%s, expected <0..%s))" %
                         (val, self.num_terms))

    def iteritems(self):
        for i in xrange(self.num_terms):
            yield i, str(i)

    def keys(self):
        """
        Override the dict.keys() function, which is used to determine the maximum
        internal id of a corpus = the vocabulary dimensionality.

        HACK: To avoid materializing the whole `range(0, self.num_terms)`, this returns
        the highest id = `[self.num_terms - 1]` only.

        """
        return [self.num_terms - 1]

    def __len__(self):
        return self.num_terms

    def get(self, val, default=None):
        if 0 <= val < self.num_terms:
            return str(val)
        return default


def dict_from_corpus(corpus):
    """
    Scan corpus for all word ids that appear in it, then construct and return a mapping
    which maps each `wordId -> str(wordId)`.

    This function is used whenever *words* need to be displayed (as opposed to just
    their ids) but no wordId->word mapping was provided. The resulting mapping
    only covers words actually used in the corpus, up to the highest wordId found.

    """
    num_terms = 1 + get_max_id(corpus)
    id2word = FakeDict(num_terms)
    return id2word


def is_corpus(obj):
    """
    Check whether `obj` is a corpus. Return (is_corpus, new) 2-tuple, where
    `new is obj` if `obj` was an iterable, or `new` yields the same sequence as
    `obj` if it was an iterator.

    `obj` is a corpus if it supports iteration over documents, where a document
    is in turn anything that acts as a sequence of 2-tuples (int, float).

    Note: An "empty" corpus (empty input sequence) is ambiguous, so in this case the
    result is forcefully defined as `is_corpus=False`.

    """
    try:
        if 'Corpus' in obj.__class__.__name__:  # the most common case, quick hack
            return True, obj
    except:
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
        id1, val1 = next(iter(doc1))  # if obj is a 1D numpy array(scalars) instead of 2-tuples, it resolves to False here
        id1, val1 = int(id1), float(val1)  # must be a 2-tuple (integer, float)
    except Exception:
        return False, obj
    return True, obj


def get_my_ip():
    """
    Try to obtain our external ip (from the pyro nameserver's point of view)

    This tries to sidestep the issue of bogus `/etc/hosts` entries and other
    local misconfigurations, which often mess up hostname resolution.

    If all else fails, fall back to simple `socket.gethostbyname()` lookup.

    """
    import socket
    try:
        import Pyro4
        # we know the nameserver must exist, so use it as our anchor point
        ns = Pyro4.naming.locateNS()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ns._pyroUri.host, ns._pyroUri.port))
        result, port = s.getsockname()
    except:
        try:
            # see what ifconfig says about our default interface
            import commands
            result = commands.getoutput("ifconfig").split("\n")[1].split()[1][5:]
            if len(result.split('.')) != 4:
                raise Exception()
        except:
            # give up, leave the resolution to gethostbyname
            result = socket.gethostbyname(socket.gethostname())
    return result


class RepeatCorpus(SaveLoad):
    """
    Used in the tutorial on distributed computing and likely not useful anywhere else.

    """
    def __init__(self, corpus, reps):
        """
        Wrap a `corpus` as another corpus of length `reps`. This is achieved by
        repeating documents from `corpus` over and over again, until the requested
        length `len(result)==reps` is reached. Repetition is done
        on-the-fly=efficiently, via `itertools`.

        >>> corpus = [[(1, 0.5)], []] # 2 documents
        >>> list(RepeatCorpus(corpus, 5)) # repeat 2.5 times to get 5 documents
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)]]

        """
        self.corpus = corpus
        self.reps = reps

    def __iter__(self):
        return itertools.islice(itertools.cycle(self.corpus), self.reps)

class RepeatCorpusNTimes(SaveLoad):

    def __init__(self, corpus, n):
        """
        Repeat a `corpus` `n` times.

        >>> corpus = [[(1, 0.5)], []]
        >>> list(RepeatCorpusNTimes(corpus, 3)) # repeat 3 times
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]
        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in xrange(self.n):
            for document in self.corpus:
                yield document

class ClippedCorpus(SaveLoad):
    def __init__(self, corpus, max_docs=None):
        """
        Return a corpus that is the "head" of input iterable `corpus`.

        Any documents after `max_docs` are ignored. This effectively limits the
        length of the returned corpus to <= `max_docs`. Set `max_docs=None` for
        "no limit", effectively wrapping the entire input corpus.

        """
        self.corpus = corpus
        self.max_docs = max_docs

    def __iter__(self):
        return itertools.islice(self.corpus, self.max_docs)

    def __len__(self):
        return min(self.max_docs, len(self.corpus))

class SlicedCorpus(SaveLoad):
    def __init__(self, corpus, slice_):
        """
        Return a corpus that is the slice of input iterable `corpus`.

        Negative slicing can only be used if the corpus is indexable.
        Otherwise, the corpus will be iterated over.

        Slice can also be a np.ndarray to support fancy indexing.

        NOTE: calculating the size of a SlicedCorpus is expensive
        when using a slice as the corpus has to be iterated over once.
        Using a list or np.ndarray does not have this drawback, but
        consumes more memory.
        """
        self.corpus = corpus
        self.slice_ = slice_
        self.length = None

    def __iter__(self):
        if hasattr(self.corpus, 'index') and len(self.corpus.index) > 0:
            return (self.corpus.docbyoffset(i) for i in
                    self.corpus.index[self.slice_])
        else:
            return itertools.islice(self.corpus, self.slice_.start,
                                    self.slice_.stop, self.slice_.step)

    def __len__(self):
        # check cached length, calculate if needed
        if self.length is None:
            if isinstance(self.slice_, (list, np.ndarray)):
                self.length = len(self.slice_)
            else:
                self.length = sum(1 for x in self)

        return self.length

def safe_unichr(intval):
    try:
        return unichr(intval)
    except ValueError:
        # ValueError: unichr() arg not in range(0x10000) (narrow Python build)
        s = "\\U%08x" % intval
        # return UTF16 surrogate pair
        return s.decode('unicode-escape')

def decode_htmlentities(text):
    """
    Decode HTML entities in text, coded as hex, decimal or named.

    Adapted from http://github.com/sku/python-twitter-ircbot/blob/321d94e0e40d0acc92f5bf57d126b57369da70de/html_decode.py

    >>> u = u'E tu vivrai nel terrore - L&#x27;aldil&#xE0; (1981)'
    >>> print(decode_htmlentities(u).encode('UTF-8'))
    E tu vivrai nel terrore - L'aldilà (1981)
    >>> print(decode_htmlentities("l&#39;eau"))
    l'eau
    >>> print(decode_htmlentities("foo &lt; bar"))
    foo < bar

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
        except:
            # in case of errors, return original input
            return match.group()

    return RE_HTML_ENTITY.sub(substitute_entity, text)


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc) for doc in itertools.islice(it, int(chunksize))]]
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
            logger.debug("prepared another chunk of %i documents (qsize=%s)" %
                        (len(wrapped_chunk[0]), qsize))
            self.q.put(wrapped_chunk.pop(), block=True)
#endclass InputQueue


if os.name == 'nt':
    warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")

    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk
else:
    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        """
        Split a stream of values into smaller chunks.
        Each chunk is of length `chunksize`, except the last one which may be smaller.
        A once-only input stream (`corpus` from a generator) is ok, chunking is done
        efficiently via itertools.

        If `maxsize > 1`, don't wait idly in between successive chunk `yields`, but
        rather keep filling a short queue (of size at most `maxsize`) with forthcoming
        chunks in advance. This is realized by starting a separate process, and is
        meant to reduce I/O delays, which can be significant when `corpus` comes
        from a slow medium (like harddisk).

        If `maxsize==0`, don't fool around with parallelism and simply yield the chunksize
        via `chunkize_serial()` (no I/O optimizations).

        >>> for chunk in chunkize(range(10), 4): print(chunk)
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9]

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

    `protocol` defaults to 2 so pickled objects are compatible across
    Python 2.x and 3.x.

    """
    with smart_open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load pickled object from `fname`"""
    with smart_open(fname, 'rb') as f:
        # Because of loading from S3 load can't be used (missing readline in smart_open)
        if sys.version_info > (3, 0):
            return _pickle.load(f, encoding='latin1')
        else:
            return _pickle.loads(f.read())

def revdict(d):
    """
    Reverse a dictionary mapping.

    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).

    """
    return dict((v, k) for (k, v) in iteritems(d))


def toptexts(query, texts, index, n=10):
    """
    Debug fnc to help inspect the top `n` most similar documents (according to a
    similarity index `index`), to see if they are actually related to the query.

    `texts` is any object that can return something insightful for each document
    via `texts[docid]`, such as its fulltext or snippet.

    Return a list of 3-tuples (docid, doc's similarity to the query, texts[docid]).

    """
    sims = index[query] # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    result = []
    for topid, topcosine in sims[:n]: # only consider top-n most similar docs
        result.append((topid, topcosine, texts[topid]))
    return result


def randfname(prefix='gensim'):
    randpart = hex(random.randint(0, 0xffffff))[2:]
    return os.path.join(tempfile.gettempdir(), prefix + randpart)


def upload_chunked(server, docs, chunksize=1000, preprocess=None):
    """
    Memory-friendly upload of documents to a SimServer (or Pyro SimServer proxy).

    Use this function to train or index large collections -- avoid sending the
    entire corpus over the wire as a single Pyro in-memory object. The documents
    will be sent in smaller chunks, of `chunksize` documents each.

    """
    start = 0
    for chunk in grouper(docs, chunksize):
        end = start + len(chunk)
        logger.info("uploading documents %i-%i" % (start, end - 1))
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
    """
    Return a Pyro name server proxy.
    """
    import Pyro4
    try:
        return Pyro4.locateNS(host, port, broadcast, hmac_key)
    except Pyro4.errors.NamingError:
        raise RuntimeError("Pyro name server not found")


def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None, ns_conf={}):
    """
    Register object with name server (starting the name server if not running
    yet) and block until the daemon is terminated. The object is registered under
    `name`, or `name`+ some random suffix if `random_suffix` is set.

    """
    if random_suffix:
        name += '.' + hex(random.randint(0, 0xffffff))[2:]
    import Pyro4
    with getNS(**ns_conf) as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            # register server for remote access
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')" % (name, uri))
            daemon.requestLoop()


def has_pattern():
    """
    Function which returns a flag indicating whether pattern is installed or not
    """
    try:
        from pattern.en import parse
        return True
    except ImportError:
        return False


def lemmatize(content, allowed_tags=re.compile('(NN|VB|JJ|RB)'), light=False,
        stopwords=frozenset(), min_length=2, max_length=15):
    """
    This function is only available when the optional 'pattern' package is installed.

    Use the English lemmatizer from `pattern` to extract UTF8-encoded tokens in
    their base form=lemma, e.g. "are, is, being" -> "be" etc.
    This is a smarter version of stemming, taking word context into account.

    Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).

    >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
    ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']

    >>> lemmatize('The study ranks high.')
    ['study/NN', 'rank/VB', 'high/JJ']

    >>> lemmatize('The ranks study hard.')
    ['rank/NN', 'study/VB', 'hard/RB']

    """
    if not has_pattern():
        raise ImportError("Pattern library is not installed. Pattern library is needed in order to use lemmatize function")
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
    """
    Create a random gensim sparse vector. Each coordinate is nonzero with
    probability `prob_nnz`, each non-zero coordinate value is drawn from
    a Poisson distribution with parameter lambda equal to `lam`.

    """
    nnz = np.random.uniform(size=(dim,))
    data = [(i, float(np.random.poisson(lam=lam) + 1.0))
            for i in xrange(dim) if nnz[i] < prob_nnz]
    return data


def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    """
    Create a random gensim-style corpus, as a list of lists of (int, float) tuples,
    to be used as a mock corpus.

    """
    data = [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam)
            for _ in xrange(n_items)]
    return data


def prune_vocab(vocab, min_reduce, trim_rule=None):
    """
    Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.

    Modifies `vocab` in place, returns the sum of all counts that were pruned.

    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    logger.info("pruned out %i tokens with count <=%i (before %i, after %i)",
                old_len - len(vocab), min_reduce, old_len, len(vocab))
    return result


def qsize(queue):
    """Return the (approximate) queue size where available; -1 where not (OS X)."""
    try:
        return queue.qsize()
    except NotImplementedError:
        # OS X doesn't support qsize
        return -1

RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2


def keep_vocab_item(word, count, min_count, trim_rule=None):
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
    Backported from Python 2.7 as it's implemented as pure python on stdlib.
    >>> check_output(args=['/usr/bin/python', '--version'])
    Python 2.6.2
    Added extra KeyboardInterrupt handling
    """
    try:
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
     """
     Pick `n` items from dictionary `d` and return them as a list.
     The items are picked randomly if `use_random` is True, otherwise picked
     according to natural dict iteration.
     """
     selected_keys = random.sample(list(d), min(len(d), n)) if use_random else itertools.islice(iterkeys(d), n)
     return [(key, d[key]) for key in selected_keys]
