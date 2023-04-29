#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# coding: utf-8
#
# Copyright (C) 2021 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# Code adapted from TinyFastSS (public domain), https://github.com/fujimotos/TinyFastSS

"""Fast approximate string similarity search using the FastSS algorithm."""

import itertools

from cpython.ref cimport PyObject


DEF MAX_WORD_LENGTH = 1000  # Maximum allowed word length, in characters. Must fit in the C `int` range.


cdef extern from *:
    """
    #define WIDTH int
    #define MAX_WORD_LENGTH 1000

    int ceditdist(PyObject * s1, PyObject * s2, WIDTH maximum) {
        WIDTH row1[MAX_WORD_LENGTH + 1];
        WIDTH row2[MAX_WORD_LENGTH + 1];
        WIDTH * CYTHON_RESTRICT pos_new;
        WIDTH * CYTHON_RESTRICT pos_old;
        int row_flip = 1;  /* Does pos_new represent row1 or row2? */
        int kind1 = PyUnicode_KIND(s1);  /* How many bytes per unicode codepoint? */
        int kind2 = PyUnicode_KIND(s2);

        WIDTH len_s1 = (WIDTH)PyUnicode_GET_LENGTH(s1);
        WIDTH len_s2 = (WIDTH)PyUnicode_GET_LENGTH(s2);
        if (len_s1 > len_s2) {
            PyObject * tmp = s1; s1 = s2; s2 = tmp;
            const WIDTH tmpi = len_s1; len_s1 = len_s2; len_s2 = tmpi;
        }
        if (len_s2 - len_s1 > maximum) return maximum + 1;
        if (len_s2 > MAX_WORD_LENGTH) return -1;
        void * s1_data = PyUnicode_DATA(s1);
        void * s2_data = PyUnicode_DATA(s2);

        WIDTH tmpi;
        for (tmpi = 0; tmpi <= len_s1; tmpi++) row2[tmpi] = tmpi;

        WIDTH i2;
        for (i2 = 0; i2 < len_s2; i2++) {
            int all_bad = i2 >= maximum;
            const Py_UCS4 ch = PyUnicode_READ(kind2, s2_data, i2);
            row_flip = 1 - row_flip;
            if (row_flip) {
                pos_new = row2; pos_old = row1;
            } else {
                pos_new = row1; pos_old = row2;
            }
            *pos_new = i2 + 1;

            WIDTH i1;
            for (i1 = 0; i1 < len_s1; i1++) {
                WIDTH val = *(pos_old++);
                if (ch != PyUnicode_READ(kind1, s1_data, i1)) {
                    const WIDTH _val1 = *pos_old;
                    const WIDTH _val2 = *pos_new;
                    if (_val1 < val) val = _val1;
                    if (_val2 < val) val = _val2;
                    val += 1;
                }
                *(++pos_new) = val;
                if (all_bad && val <= maximum) all_bad = 0;
            }
            if (all_bad) return maximum + 1;
        }

        return row_flip ? row2[len_s1] : row1[len_s1];
    }
    """
    int ceditdist(PyObject *s1, PyObject *s2, int maximum)


def editdist(s1: str, s2: str, max_dist=None):
    """
    Return the Levenshtein distance between two strings.

    Use `max_dist` to control the maximum distance you care about. If the actual distance is larger
    than `max_dist`, editdist will return early, with the value `max_dist+1`.
    This is a performance optimization – for example if anything above distance 2 is uninteresting
    to your application, call editdist with `max_dist=2` and ignore any return value greater than 2.

    Leave `max_dist=None` (default) to always return the full Levenshtein distance (slower).

    """
    if s1 == s2:
        return 0

    result = ceditdist(<PyObject *>s1, <PyObject *>s2, MAX_WORD_LENGTH if max_dist is None else int(max_dist))
    if result >= 0:
        return result
    elif result == -1:
        raise ValueError(f"editdist doesn't support strings longer than {MAX_WORD_LENGTH} characters")
    else:
        raise ValueError(f"editdist returned an error: {result}")


def indexkeys(word, max_dist):
    """Return the set of index keys ("variants") of a word.

    >>> indexkeys('aiu', 1)
    {'aiu', 'iu', 'au', 'ai'}
    """
    res = set()
    wordlen = len(word)
    limit = min(max_dist, wordlen) + 1

    for dist in range(limit):
        for variant in itertools.combinations(word, wordlen - dist):
            res.add(''.join(variant))

    return res


def set2bytes(s):
    """Serialize a set of unicode strings into bytes.

    >>> set2byte({u'a', u'b', u'c'})
    b'a\x00b\x00c'
    """
    return '\x00'.join(s).encode('utf8')


def bytes2set(b):
    """Deserialize bytes into a set of unicode strings.

    >>> bytes2set(b'a\x00b\x00c')
    {u'a', u'b', u'c'}
    """
    return set(b.decode('utf8').split('\x00')) if b else set()


class FastSS:
    """
    Fast implementation of FastSS (Fast Similarity Search): https://fastss.csg.uzh.ch/

    FastSS enables fuzzy search of a dynamic query (a word, string) against a static
    dictionary (a set of words, strings). The "fuziness" is configurable by means
    of a maximum edit distance (Levenshtein) between the query string and any of the
    dictionary words.

    """

    def __init__(self, words=None, max_dist=2):
        """
        Create a FastSS index. The index will contain encoded variants of all
        indexed words, allowing fast "fuzzy string similarity" queries.

        max_dist: maximum allowed edit distance of an indexed word to a query word. Keep
        max_dist<=3 for sane performance.

        """
        self.db = {}
        self.max_dist = max_dist
        if words:
            for word in words:
                self.add(word)

    def __str__(self):
        return "%s<max_dist=%s, db_size=%i>" % (self.__class__.__name__, self.max_dist, len(self.db), )

    def __contains__(self, word):
        bkey = word.encode('utf8')
        if bkey in self.db:
            return word in bytes2set(self.db[bkey])
        return False

    def add(self, word):
        """Add a string to the index."""
        for key in indexkeys(word, self.max_dist):
            bkey = key.encode('utf8')
            wordset = {word}

            if bkey in self.db:
                wordset |= bytes2set(self.db[bkey])

            self.db[bkey] = set2bytes(wordset)

    def query(self, word, max_dist=None):
        """Find all words from the index that are within max_dist of `word`."""
        if max_dist is None:
            max_dist = self.max_dist
        if max_dist > self.max_dist:
            raise ValueError(
                f"query max_dist={max_dist} cannot be greater than max_dist={self.max_dist} from the constructor"
            )

        res = {d: [] for d in range(max_dist + 1)}
        cands = set()

        for key in indexkeys(word, max_dist):
            bkey = key.encode('utf8')

            if bkey in self.db:
                cands.update(bytes2set(self.db[bkey]))

        for cand in cands:
            dist = editdist(word, cand, max_dist=max_dist)
            if dist <= max_dist:
                res[dist].append(cand)

        return res
