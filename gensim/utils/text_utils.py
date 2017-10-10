#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import re
import string

from gensim import utils
from six.moves import xrange


class PorterStemmer(object):
    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[0],
        b[1] ... ending at b[k]. k is readjusted downwards as the stemming
        progresses.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.j = 0   # j is a general offset into the string

    def _cons(self, i):
        """True <=> b[i] is a consonant."""
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == 'y':
            return i == 0 or not self._cons(i - 1)
        return True

    def _m(self):
        """Returns the number of consonant sequences between 0 and j.

        If c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        i = 0
        while True:
            if i > self.j:
                return 0
            if not self._cons(i):
                break
            i += 1
        i += 1
        n = 0
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while 1:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        """True <=> 0,...j contains a vowel"""
        return not all(self._cons(i) for i in xrange(self.j + 1))

    def _doublec(self, j):
        """True <=> j,(j-1) contain a double consonant."""
        return j > 0 and self.b[j] == self.b[j - 1] and self._cons(j)

    def _cvc(self, i):
        """True <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. This is used when trying to
        restore an e at the end of a short word, e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s):
        """True <=> 0,...k ends with the string s."""
        if s[-1] != self.b[self.k]:  # tiny speed-up
            return 0
        length = len(s)
        if length > (self.k + 1):
            return 0
        if self.b[self.k - length + 1:self.k + 1] != s:
            return 0
        self.j = self.k - length
        return 1

    def _setto(self, s):
        """Set (j+1),...k to the characters in the string s, adjusting k."""
        self.b = self.b[:self.j + 1] + s
        self.k = len(self.b) - 1

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        """Get rid of plurals and -ed or -ing. E.g.,

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                if self.b[self.k - 1] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        """Turn terminal y to i when there is another vowel in the stem."""
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i'

    def _step2(self):
        """Map double suffices to single ones.

        So, -ization ( = -ize plus -ation) maps to -ize etc. Note that the
        string before the suffix must give _m() > 0.
        """
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif ch == 'c':
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif ch == 'e':
            if self._ends("izer"):
                self._r("ize")
        elif ch == 'l':
            if self._ends("bli"):
                self._r("ble")  # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self._ends("abli"):      self._r("able")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif ch == 'o':
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif ch == 's':
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif ch == 't':
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")
        elif ch == 'g':  # --DEPARTURE--
            if self._ends("logi"):
                self._r("log")
        # To match the published algorithm, delete this phrase

    def _step3(self):
        """Deal with -ic-, -full, -ness etc. Similar strategy to _step2."""
        ch = self.b[self.k]
        if ch == 'e':
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif ch == 'i':
            if self._ends("iciti"):
                self._r("ic")
        elif ch == 'l':
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif ch == 's':
            if self._ends("ness"):
                self._r("")

    def _step4(self):
        """_step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        ch = self.b[self.k - 1]
        if ch == 'a':
            if not self._ends("al"):
                return
        elif ch == 'c':
            if not self._ends("ance") and not self._ends("ence"):
                return
        elif ch == 'e':
            if not self._ends("er"):
                return
        elif ch == 'i':
            if not self._ends("ic"):
                return
        elif ch == 'l':
            if not self._ends("able") and not self._ends("ible"):
                return
        elif ch == 'n':
            if self._ends("ant"):
                pass
            elif self._ends("ement"):
                pass
            elif self._ends("ment"):
                pass
            elif self._ends("ent"):
                pass
            else:
                return
        elif ch == 'o':
            if self._ends("ion") and self.b[self.j] in "st":
                pass
            elif self._ends("ou"):
                pass
            # takes care of -ous
            else:
                return
        elif ch == 's':
            if not self._ends("ism"):
                return
        elif ch == 't':
            if not self._ends("ate") and not self._ends("iti"):
                return
        elif ch == 'u':
            if not self._ends("ous"):
                return
        elif ch == 'v':
            if not self._ends("ive"):
                return
        elif ch == 'z':
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self):
        """Remove a final -e if _m() > 1, and change -ll to -l if m() > 1.
        """
        k = self.j = self.k
        if self.b[k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, w):
        """Stem the word w, return the stemmed form."""
        w = w.lower()
        k = len(w) - 1
        if k <= 1:
            return w  # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.b = w
        self.k = k

        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]

    def stem_sentence(self, txt):
        return " ".join(self.stem(x) for x in txt.split())

    def stem_documents(self, docs):
        return [self.stem_sentence(x) for x in docs]


STOPWORDS = frozenset({
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along',
    'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another',
    'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became',
    'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
    'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co',
    'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn',
    'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere',
    'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few',
    'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four',
    'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her',
    'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
    'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'just',
    'keep', 'kg', 'km', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'make', 'many', 'may', 'me',
    'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my',
    'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone',
    'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or',
    'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps',
    'please', 'put', 'quite', 'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem', 'seemed',
    'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty',
    'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
    'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
    'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those',
    'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards',
    'twelve', 'twenty', 'two', 'un', 'under', 'unless', 'until', 'up', 'upon', 'us', 'used', 'using', 'various',
    'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
    'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither',
    'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you',
    'your', 'yours', 'yourself', 'yourselves'
})


def remove_stopwords(s):
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)


def strip_punctuation(s):
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(" ", s)


strip_punctuation2 = strip_punctuation
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)


def strip_tags(s):
    s = utils.to_unicode(s)
    return RE_TAGS.sub("", s)


def strip_short(s, minsize=3):
    s = utils.to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)


RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)


def strip_numeric(s):
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub("", s)


RE_NONALPHA = re.compile(r"\W", re.UNICODE)


def strip_non_alphanum(s):
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(" ", s)


RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def strip_multiple_whitespaces(s):
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)


RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)


def split_alphanum(s):
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)


def stem_text(text):
    """
    Return lowercase and (porter-)stemmed version of string `text`.
    """
    text = utils.to_unicode(text)
    p = PorterStemmer()
    return ' '.join(p.stem(word) for word in text.split())


stem = stem_text
DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short, stem_text
]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    return [preprocess_string(d) for d in docs]
