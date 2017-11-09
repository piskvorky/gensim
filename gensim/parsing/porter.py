#!/usr/bin/env python

"""Porter Stemming Algorithm
This is the Porter stemming algorithm, ported to Python from the
version coded up in ANSI C by the author. It may be be regarded
as canonical, in that it follows the algorithm presented in

Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
no. 3, pp 130-137,

only differing from it at the points maked --DEPARTURE-- below.

See also http://www.tartarus.org/~martin/PorterStemmer

The algorithm as described in the paper could be exactly replicated
by adjusting the points of DEPARTURE, but this is barely necessary,
because (a) the points of DEPARTURE are definitely improvements, and
(b) no encoding of the Porter stemmer I have seen is anything like
as exact as this version, even with the points of DEPARTURE!

Vivake Gupta (v@nano.com)

Release 1: January 2001

Further adjustments by Santiago Bruno (bananabruno@gmail.com)
to allow word input not restricted to one word per line, leading
to:

Release 2: July 2008

Optimizations and cleanup of the code by Lars Buitinck, July 2012.

Notes:
------

The main part of the stemming algorithm (https://en.wikipedia.org/wiki/Stemming)
starts in :func:`~gensim.parsing.porter.PorterStemmer`.


Attributes
--------
b : str : is a buffer holding a word to be stemmed. The letters are in b[0], b[1] ... ending at b[k].
k : int : is readjusted downwards as the stemming progresses.
j : int : is word length.


Examples:
---------

>>> from gensim.parsing.porter import PorterStemmer
>>> p = PorterStemmer()
>>> text = "Cats and ponies have meeting"
>>> p.stem_sentence(text)
'cat and poni have meet'

>>> from gensim.parsing.porter import PorterStemmer
>>> p = PorterStemmer()
>>> docs = ["Cats and ponies", "have meeting"]
>>> p.stem_documents(docs)
['cat and poni', 'have meet']

"""


from six.moves import xrange


class PorterStemmer(object):
    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[0],
        b[1] ... ending at b[k]. k is readjusted downwards as the stemming
        progresses.


        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> print "b (word) = ", p.b, " ,k (readjusted downwards as the stemming progresses) = ", p.k, " ,j (word length) = ", p.j
        b =    ,k =  0  ,j =  0

        """
        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.j = 0   # j is a general offset into the string

    def _cons(self, i):
        """Take b[i], check if it is a consonant.

        Parameters
        ----------
        i : int

        Returns
        -------
        bool
            True, if b[i] is a consonant, otherwise - False.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "hi"
        >>> p._cons(1)
        False

        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "meow"
        >>> p._cons(3)
        True

        """
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == 'y':
            return i == 0 or not self._cons(i - 1)
        return True

    def _m(self):
        """Return the number of consonant sequences between 0 and j.

        If c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3

        Returns
        -------
        int
            The number of consonant sequences between 0 and j.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "<bm>aobm<ao>"
        >>> p.j = 11
        >>> p._m()
        2

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
        """Check if b[i] (i = 0,...j) contains a vowel.

        Returns
        -------
        bool
            True, if b contains a vowel, otherwise - False.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "gnsm"
        >>> p.j = 3
        >>> p._vowelinstem()
        False

        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "gensim"
        >>> p.j = 5
        >>> p._vowelinstem()
        True

        """
        return not all(self._cons(i) for i in xrange(self.j + 1))

    def _doublec(self, j):
        """Check if b[j], b[j-1] contain a double consonant.

        Parameters
        ----------
        j : int

        Returns
        -------
        bool
            True, if b[j], b[j-1] contain a double consonant , otherwise - False.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "real"
        >>> p.j = 3
        >>> p._doublec(3)
        False

        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "really"
        >>> p.j = 5
        >>> p._doublec(4)
        True

        """
        return j > 0 and self.b[j] == self.b[j - 1] and self._cons(j)

    def _cvc(self, i):
        """Check if b[i-2], b[i-1], b[i] have the form consonant - vowel - consonant
        and also if the second c is not w,x or y. This is used when trying to
        restore an e at the end of a short word, e.g.
        cav(e), lov(e), hop(e), crim(e), but
        snow, box, tray.

        Parameters
        ----------
        i : int

        Returns
        -------
        bool
            True, if b[i-2], b[i-1], b[i] have the form consonant - vowel - consonant , otherwise - False.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "lib"
        >>> p.j = 2
        >>> p._cvc(2)
        True

        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "dll"
        >>> p.j = 2
        >>> p._cvc(2)
        False

        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "wow"
        >>> p.j = 2
        >>> p._cvc(2)
        False

        """
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s):
        """Check if b[0],...b[k] ends with the string s.

        Parameters
        ----------
        s : str

        Returns
        -------
        int
            1, if b[0],...b[k] ends with the string s, otherwise - 0.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> p.b = "cowboy"
        >>> p.j = 5
        >>> p.k = 2
        >>> p._ends("cow")
        1

        """
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
        """Set (j+1),...k to the characters in the string s, adjusting k.

        Parameters
        ----------
        s : str

        """
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
        """Stem the word w, return the stemmed form.

        Parameters
        ----------
        w : str

        Returns
        -------
        str
            Stemmed version of  w.

        Examples
        --------
        >>> from gensim.parsing.porter import PorterStemmer
        >>> p = PorterStemmer()
        >>> text = "Matting ponies"
        >>> p.stem(text)
        'matting poni'

        """
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


if __name__ == '__main__':
    import sys

    p = PorterStemmer()

    for f in sys.argv[1:]:
        with open(f) as infile:
            for line in infile:
                print(p.stem_sentence(line))
