#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import re
import string
import glob

from gensim import utils
from gensim.parsing.porter import PorterStemmer


# improved list from Stone, Denis, Kwantes (2010)
STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)

RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)

RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)

RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)

RE_NONALPHA = re.compile(r"\W", re.UNICODE)

RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)

RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)

RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def remove_stopwords(s):
    """Take string, remove all words those are among stopwords.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without stopwords.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import remove_stopwords
    >>> s = "Better late than never, but better never late."
    >>> remove_stopwords(s)
    u'Better late never, better late.'

    """
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


def strip_punctuation(s):
    """Take string, replace all punctuation characters with spaces.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without punctuation characters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_punctuation
    >>> s = "A semicolon is a stronger break than a comma, but not as much as a full stop!"
    >>> strip_punctuation(s)
    u'A semicolon is a stronger break than a comma  but not as much as a full stop '

    """
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(" ", s)


# unicode.translate cannot delete characters like str can
strip_punctuation2 = strip_punctuation

# def strip_punctuation2(s):
#     s = utils.to_unicode(s)
#     return s.translate(None, string.punctuation)


def strip_tags(s):
    """Take string and remove tags.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without tags.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_tags
    >>> s = "<i>Hello</i> <b>World</b>!"
    >>> strip_tags(s)
    u'Hello World!'

    """
    s = utils.to_unicode(s)
    return RE_TAGS.sub("", s)


def strip_short(s, minsize=3):
    """Take string and remove words with length lesser than minsize (default = 3).

    Parameters
    ----------
    s : str
    minsize : int, optional

    Returns
    -------
    str
        Unicode string without words with length lesser than minsize.


    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_short
    >>> s = "salut les amis du 59"
    >>> strip_short(s)
    u'salut les amis'

    >>> from gensim.parsing.preprocessing import strip_short
    >>> s = "one two three four five six seven eight nine ten"
    >>> strip_short(s,5)
    u'three seven eight'

    """
    s = utils.to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_numeric(s):
    """Take string and remove digits from it.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without digits.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_numeric
    >>> s = "0text24gensim365test"
    >>> strip_numeric(s)
    u'textgensimtest'

    """
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub("", s)


def strip_non_alphanum(s):
    """Take string and remove not a word characters from it.
    (Word characters - alphanumeric & underscore)

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without not a word characters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_non_alphanum
    >>> s = "if-you#can%read$this&then@this#method^works"
    >>> strip_non_alphanum(s)
    u'if you can read this then this method works'

    """
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(" ", s)


def strip_multiple_whitespaces(s):
    r"""Take string, remove repeating in a row whitespace characters (spaces, tabs, line breaks) from it
    and turns tabs & line breaks into spaces.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without repeating in a row whitespace characters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_multiple_whitespaces
    >>> s = "salut" + '\r' + " les" + '\n' + "         loulous!"
    >>> strip_multiple_whitespaces(s)
    u'salut les loulous!'

    """
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)


def split_alphanum(s):
    """Take string, add spaces between digits & letters.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with spaces between digits & letters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import split_alphanum
    >>> s = "24.0hours7 days365 a1b2c3"
    >>> split_alphanum(s)
    u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)


def stem_text(text):
    """Take string, tranform it into lowercase and (porter-)stemmed version.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Lowercase and (porter-)stemmed version of string `text`.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import stem_text
    >>> text = "While it is quite useful to be able to search a large collection of documents almost instantly for a joint occurrence of a collection of exact words, for many searching purposes, a little fuzziness would help. "
    >>> stem_text(text)
    u'while it is quit us to be abl to search a larg collect of document almost instantli for a joint occurr of a collect of exact words, for mani search purposes, a littl fuzzi would help.'

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
    """Take string, apply list of chosen filters to it, where filters are methods from this module.
    Default list of filters consists of: strip_tags, strip_punctuation, strip_multiple_whitespaces,
    strip_numeric, remove_stopwords, strip_short, stem_text. <function <lambda>> in signature means
    that we use lambda function for applying methods to filters.

    Parameters
    ----------
    s : str
    filters: list, optional

    Returns
    -------
    list
        List of unicode strings.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import preprocess_string
    >>> s = "<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"
    >>> preprocess_string(s)
    [u'hel', u'rld', u'weather', u'todai', u'isn']

    >>> from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation
    >>> s = "<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"
    >>> CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
    >>> preprocess_string(s, CUSTOM_FILTERS)
    [u'hel', u'9lo', u'wo9', u'rld', u'th3', u'weather', u'is', u'really', u'g00d', u'today', u'isn', u't', u'it']

    """
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    """Take list of strings, then apply default filters to every string.

    Parameters
    ----------
    docs : list

    Returns
    -------
    list
        List of lists, filled by unicode strings.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import preprocess_documents
    >>> s = ["<i>Hel 9lo</i> <b>Wo9 rld</b>!", "Th3     weather_is really g00d today, isn't it?"]
    >>> preprocess_documents(s)
    [[u'hel', u'rld'], [u'weather', u'todai', u'isn']]

    """
    return [preprocess_string(d) for d in docs]


def read_file(path):
    with utils.smart_open(path) as fin:
        return fin.read()


def read_files(pattern):
    return [read_file(fname) for fname in glob.glob(pattern)]
