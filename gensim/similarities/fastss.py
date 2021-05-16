"""Create and query FastSS index for string simirality search.

Command-line usage:

  Create a new index from dictionary:

    python -m fastss -c index.dat <file>

  Query an index file:

    python -m fastss -q index.dat <query>

Create mode options:

  --maxdist  <N> maximum edit distance for the index (default: 2)
  --encoding <S> the encoding of the dictionary file.

Note:

  For creating an index, you need to pass a dictionary file which
  contains a list of words in a one-word-per-line manner. If <file>
  argument is omitted, it tries to read from stdin.
"""

import struct
import itertools


#
# Constants

ENCODING = 'utf-8'
DELIMITER = b'\x00'


#
# Utils

def editdist(s1, s2):
    r"""Return the Levenshtein distance between two strings.

    >>> editdist('aiu', 'aie')
    1
    """

    matrix = {}

    for i in range(len(s1) + 1):
        matrix[(i, 0)] = i
    for j in range(len(s2) + 1):
        matrix[(0, j)] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                matrix[(i, j)] = matrix[(i - 1, j - 1)]
            else:
                matrix[(i, j)] = min(
                    matrix[(i - 1, j)],
                    matrix[(i, j - 1)],
                    matrix[(i - 1, j - 1)]
                ) + 1

    return matrix[(i, j)]


def indexkeys(word, max_dist):
    r"""Return the set of index keys ("variants") of a word.

    >>> indexkeys('aiu', 1)
    {'aiu', 'iu', 'au', 'ai'}
    """

    res = set()
    wordlen = len(word)
    limit = min(max_dist, wordlen) + 1

    for dist in range(limit):
        variants = itertools.combinations(word, wordlen - dist)

        for variant in variants:
            res.add(''.join(variant))

    return res


def int2byte(i):
    r"""Encode a positive int (<= 256) into a 8-bit byte.

    >>> int2byte(1)
    b'\x01'
    """
    return struct.pack('B', i)


def byte2int(b):
    r"""Decode a 8-bit byte into an integer.

    >>> byte2int(b'\x01')
    1
    """
    return struct.unpack('B', b)[0]


def set2bytes(s):
    r"""Serialize a set of unicode strings into bytes.

    >>> set2byte({u'a', u'b', u'c'})
    b'a\x00b\x00c'
    """
    lis = []
    for uword in sorted(s):
        bword = uword.encode(ENCODING)
        lis.append(bword)
    return DELIMITER.join(lis)


def bytes2set(b):
    r"""Deserialize bytes into a set of unicode strings.

    >>> int2byte(b'a\x00b\x00c')
    {u'a', u'b', u'c'}
    """
    if not b:
        return set()

    lis = b.split(DELIMITER)
    return set(bword.decode(ENCODING) for bword in lis)


#
# FastSS class

class FastSS:
    def __init__(self, max_dist):
        r"""Open an FastSS index.

        max_dist: the uppser threshold of edit distance for the index. Only
                  effective when creating a new index file.
        """
        self.db = {}
        self.max_dist = max_dist

    def __str__(self):
        return "%s<max_dist=%s, dictionary=%i" % (self.__class__.__name__, self.max_dist, len(self.db), )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __contains__(self, word):
        bkey = word.encode(ENCODING)
        if bkey in self.db:
            return word in bytes2set(self.db[bkey])
        return False

    def close(self):
        pass

    def add(self, word):
        for key in indexkeys(word, self.max_dist):
            bkey = key.encode(ENCODING)
            wordset = {word}

            if bkey in self.db:
                wordset |= bytes2set(self.db[bkey])

            self.db[bkey] = set2bytes(wordset)

    def query(self, word, max_dist):
        res = {d: [] for d in range(max_dist + 1)}
        cands = set()

        for key in indexkeys(word, max_dist):
            bkey = key.encode(ENCODING)

            if bkey in self.db:
                cands.update(bytes2set(self.db[bkey]))

        for cand in cands:
            dist = editdist(word, cand)
            if dist <= max_dist:
                res[dist].append(cand)

        return res
