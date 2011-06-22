#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains various general utility functions.
"""

from __future__ import with_statement

import logging
import re
import unicodedata
import cPickle
import itertools
from functools import wraps # for `synchronous` function lock
from htmlentitydefs import name2codepoint as n2cp # for `decode_htmlentities`
import threading, time
from Queue import Queue, Empty



logger = logging.getLogger('gensim.utils')


PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)(x?)(\w+);', re.UNICODE)



def synchronous(tlockname):
    """
    A decorator to place an instance-based lock around a method.

    Adapted from http://code.activestate.com/recipes/577105-synchronization-decorator-for-class-methods/
    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = self.__getattribute__(tlockname)
            with tlock: # use lock as a context manager to perform safe acquire/release pairs
                return func(self, *args, **kwargs)
        return _synchronizer
    return _synched



def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.

    Return input string with accents removed, as unicode.

    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'
    """
    if not isinstance(text, unicode):
        text = unicode(text, 'utf8') # assume utf8 for byte strings, use default (strict) error handling
    norm = unicodedata.normalize("NFD", text)
    result = u''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def tokenize(text, lowercase=False, deacc=False, errors="strict", to_lower=False, lower=False):
    """
    Iteratively yield tokens as unicode strings, optionally also lowercasing them
    and removing accent marks.

    Input text may be either unicode or utf8-encoded byte string.

    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).

    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or to_lower or lower
    if not isinstance(text, unicode):
        text = unicode(text, encoding='utf8', errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def to_utf8(text, errors='strict'):
    """
    Like built-in `unicode.encode('utf8')`, but allow input to be bytestring,
    too (so this is a no-op if input already is a bytestring in utf8).
    """
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, 'utf8', errors=errors).encode('utf8')


def to_unicode(text, encoding='utf8', errors='strict'):
    """Like built-in `unicode`, but simply return input if `text` already is unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions, which un/pickle
    them to disk.

    This uses cPickle for de/serializing, so objects must not contains unpicklable
    attributes, such as lambda functions etc.
    """
    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see `save`).
        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))
        return unpickle(fname)

    def save(self, fname):
        """
        Save the object to file via pickling (also see `load`).
        """
        logger.info("saving %s object to %s" % (self.__class__.__name__, fname))
        pickle(self, fname)
#endclass SaveLoad


def identity(p):
    return p


def get_max_id(corpus):
    """
    Return highest feature id that appears in the corpus.

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
        `[self.num_terms - 1]` only.
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
    which maps each ``wordId -> str(wordId)``.

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
        if 'Corpus' in obj.__class__.__name__: # the most common case, quick hack
            return True, obj
    except:
        pass
    try:
        if hasattr(obj, 'next'):
            # the input is an iterator object, meaning once we call next()
            # that element could be gone forever. we must be careful to put
            # whatever we retrieve back again
            doc1 = obj.next()
            obj = itertools.chain([doc1], obj)
        else:
            doc1 = iter(obj).next() # empty corpus is resolved to False here
        if len(doc1) == 0: # sparse documents must have a __len__ function (list, tuple...)
            return True, obj # the first document is empty=>assume this is a corpus
        id1, val1 = iter(doc1).next() # if obj is a numpy array, it resolves to False here
        id1, val1 = int(id1), float(val1) # must be a 2-tuple (integer, float)
    except:
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
        import Pyro
        # we know the nameserver must exist, so use it as our anchor point
        ns = Pyro.naming.locateNS()
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


def decode_htmlentities(text):
    """
    Decode HTML entities in text, coded as hex, decimal or named.

    Adapted from http://github.com/sku/python-twitter-ircbot/blob/321d94e0e40d0acc92f5bf57d126b57369da70de/html_decode.py

    >>> u = u'E tu vivrai nel terrore - L&#x27;aldil&#xE0; (1981)'
    >>> print decode_htmlentities(u).encode('UTF-8')
    E tu vivrai nel terrore - L'aldilà (1981)
    >>> print decode_htmlentities("l&#39;eau")
    l'eau
    >>> print decode_htmlentities("foo &lt; bar")
    foo < bar

    """
    def substitute_entity(match):
        ent = match.group(3)
        if match.group(1) == "#":
            # decoding by number
            if match.group(2) == '':
                # number is in decimal
                return unichr(int(ent))
            elif match.group(2) == 'x':
                # number is in hex
                return unichr(int('0x' + ent, 16))
        else:
            # they were using a name
            cp = n2cp.get(ent)
            if cp:
                return unichr(cp)
            else:
                return match.group()

    try:
        return RE_HTML_ENTITY.sub(substitute_entity, text)
    except:
        # in case of errors, return input
        # e.g., ValueError: unichr() arg not in range(0x10000) (narrow Python build)
        return text


def chunkize_serial(corpus, chunksize):
    """
    Split a stream of values into smaller chunks.
    Each chunk is of length `chunksize`, except the last one which may be smaller.
    A once-only input stream (`corpus` from a generator) is ok.

    >>> for chunk in chunkize_serial(xrange(10), 4): print list(chunk)
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9]

    """
    if chunksize <= 0:
        raise ValueError("chunk size must be greater than zero")
    i = (val for val in corpus) # create generator
    while True:
        chunk = list(itertools.islice(i, int(chunksize))) # consume `chunksize` items from the generator
        if not chunk: # generator empty?
            break
        yield chunk


def chunkize(corpus, chunksize, maxsize=0):
    """
    Split a stream of values into smaller chunks.
    Each chunk is of length `chunksize`, except the last one which may be smaller.
    A once-only input stream (`corpus` from a generator) is ok, chunking is done
    efficiently via itertools.

    If `maxsize > 1`, don't wait idly in between successive chunk `yields`, but
    rather keep filling a short queue (of size at most `maxsize`) with forthcoming
    chunks in advance. This is realized by starting a separate thread, and is
    meant to reduce I/O delays, which can be significant when `corpus` comes
    from a slow medium (like harddisk).

    If `maxsize==0`, don't fool around with threads and simply yield the chunksize
    via `chunkize_serial()` (no I/O optimizations).

    >>> for chunk in chunkize(xrange(10), 4): print chunk
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9]

    """
    class InputQueue(threading.Thread):
        """
        Help class for threaded `chunkize()`.
        """
        def __init__(self, q, corpus, chunksize, maxsize):
            super(InputQueue, self).__init__()
            self.q = q
            self.maxsize = maxsize
            self.corpus = corpus
            self.chunksize = chunksize

        def run(self):
            import numpy # don't clutter the global namespace with a dependency on numpy
            i = (val for val in self.corpus) # create generator
            while True:
                # HACK XXX convert documents to numpy arrays, to save memory.
                # This also gives a scipy warning at runtime:
                # "UserWarning: indices array has non-integer dtype (float64)"
                chunk = [numpy.asarray(doc) for doc in itertools.islice(i, self.chunksize)] # consume `chunksize` items from the generator
                if not chunk: # generator empty?
                    break
                logger.info("prepared another chunk of %i documents (qsize=%i)" %
                            (len(chunk), self.q.qsize()))
                self.q.put(chunk, block=True)
    #endclass InputQueue

    assert chunksize > 0

    if maxsize > 0:
        q = Queue(maxsize=maxsize)
        thread = InputQueue(q, corpus, chunksize, maxsize=maxsize)
        thread.start()
        while thread.isAlive() or not q.empty():
            try:
                yield q.get(block=True, timeout=1)
            except Empty:
                pass
    else:
        for chunk in chunkize_serial(corpus, chunksize):
            yield chunk


def pickle(obj, fname, protocol=-1):
    """Pickle object `obj` to file `fname`."""
    with open(fname, 'wb') as fout: # 'b' for binary, needed on Windows
        cPickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load pickled object from `fname`"""
    return cPickle.load(open(fname, 'rb'))


def revdict(d):
    """
    Reverse a dictionary mapping.

    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary)."""
    return dict((v, k) for (k, v) in d.iteritems())


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

