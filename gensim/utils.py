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
logger = logging.getLogger('gensim.utils')

import re
import unicodedata
import os
import random
import cPickle
import itertools
import tempfile
from functools import wraps # for `synchronous` function lock
from htmlentitydefs import name2codepoint as n2cp # for `decode_htmlentities`
import multiprocessing
import shutil
import traceback


try:
    from pattern.en import parse
    logger.info("'pattern' package found; utils.lemmatize() is available for English")
    HAS_PATTERN = True
except ImportError:
    HAS_PATTERN = False


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
            tlock = getattr(self, tlockname)
            logger.debug("acquiring lock %r for %s" % (tlockname, func.func_name))

            with tlock: # use lock as a context manager to perform safe acquire/release pairs
                logger.debug("acquired lock %r for %s" % (tlockname, func.func_name))
                result = func(self, *args, **kwargs)
                logger.debug("releasing lock %r for %s" % (tlockname, func.func_name))
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


def simple_preprocess(doc, deacc=False):
    """
    Convert a document into a list of tokens.

    This lowercases, tokenizes, stems, normalizes etc. -- the output are final,
    utf8 encoded strings that won't be processed any further.
    """
    tokens = [token.encode('utf8') for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
            if 2 <= len(token) <= 15 and not token.startswith('_')]
    return tokens


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8.
    """
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


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print list(grouper(xrange(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    import numpy
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[numpy.array(doc) for doc in itertools.islice(it, int(chunksize))]]
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
        if self.as_numpy:
            import numpy # don't clutter the global namespace with a dependency on numpy
        it = iter(self.corpus)
        while True:
            chunk = itertools.islice(it, self.chunksize)
            if self.as_numpy:
                # HACK XXX convert documents to numpy arrays, to save memory.
                # This also gives a scipy warning at runtime:
                # "UserWarning: indices array has non-integer dtype (float64)"
                wrapped_chunk = [[numpy.asarray(doc) for doc in chunk]]
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
    logger.info("detected Windows; aliasing chunkize to chunkize_serial")

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

        >>> for chunk in chunkize(xrange(10), 4): print chunk
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


def smart_open(fname, mode='r'):
    from os import path
    _, ext = path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return make_closing(BZ2File)(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return make_closing(GzipFile)(fname, mode)
    return open(fname, mode)


def pickle(obj, fname, protocol=-1):
    """Pickle object `obj` to file `fname`."""
    with smart_open(fname, 'wb') as fout: # 'b' for binary, needed on Windows
        cPickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load pickled object from `fname`"""
    return cPickle.load(smart_open(fname, 'rb'))


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


def getNS():
    """
    Return a Pyro name server proxy. If there is no name server running,
    start one on 0.0.0.0 (all interfaces), as a background process.
    """
    import Pyro4
    try:
        return Pyro4.locateNS()
    except Pyro4.errors.NamingError:
        logger.info("Pyro name server not found; starting a new one")
    os.system("python -m Pyro4.naming -n 0.0.0.0 &")
    # TODO: spawn a proper daemon ala http://code.activestate.com/recipes/278731/ ?
    # like this, if there's an error somewhere, we'll never know... (and the loop
    # below will block). And it probably doesn't work on windows, either.
    while True:
        try:
            return Pyro4.locateNS()
        except:
            pass


def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None):
    """Register object with name server (starting the name server if not running
    yet) and block until the daemon is terminated. The object is registered under
    `name`, or `name`+ some random suffix if `random_suffix` is set."""
    if random_suffix:
        name += '.' + hex(random.randint(0, 0xffffff))[2:]
    import Pyro4
    with getNS() as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            # register server for remote access
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')" % (name, uri))
            daemon.requestLoop()


if HAS_PATTERN:
    def lemmatize(content, light=False, allowed_tags=re.compile('(NN|VB|JJ|RB)')):
        """
        This function is only available when the optional 'pattern' package is installed.

        Use the English lemmatizer from `pattern` to extract tokens in
        their base form=lemma, e.g. "are, is, being" -> "be" etc.
        This is a smarter version of stemming. Only consider nouns, verbs, adjectives
        and adverbs by default (=all other lemmas are discarded).

        >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
        ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']

        From http://www.clips.ua.ac.be/pages/pattern-en#parser :

            The parser is built on a Brill lexicon of tagged words and rules to
            improve the tags context-wise. With light=False, it uses Brill's contextual
            rules. With light=True it uses Jason Wiener's simpler ruleset. This
            ruleset is 5-10x faster but also 25% less accurate.

        """
        # tokenization in `pattern` is weird; it gets thrown off by non-letters,
        # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
        # FIXME this throws away all fancy parsing cues, including sentence structure,
        # abbreviations etc.
        content = u' '.join(tokenize(content, lower=True, errors='ignore'))

        # use simpler, modified pattern.text.en.text.parser.parse that doesn't
        # collapse the output at the end: https://github.com/piskvorky/pattern
        parsed = parse(content, lemmata=True, collapse=False, light=light)
        result = []
        for sentence in parsed:
            for token, tag, _, _, lemma in sentence:
                if 2 <= len(lemma) <= 15 and not lemma.startswith('_'):
                    if allowed_tags.match(tag):
                        lemma += "/" + tag[:2]
                        result.append(lemma.encode('utf8'))
        return result
#endif HAS_PATTERN
