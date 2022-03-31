#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains methods for parsing and preprocessing strings.

Examples
--------

.. sourcecode:: pycon

    >>> from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
    >>> remove_stopwords("Better late than never, but better never late.")
    u'Better late never, better late.'
    >>>
    >>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
    [u'hel', u'rld', u'weather', u'todai', u'isn']

"""

import re
import string
import glob

from gensim import utils
from gensim.parsing.porter import PorterStemmer


STOPWORDS = frozenset([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fifty', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once'
])


RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_NONALPHA = re.compile(r"\W", re.UNICODE)
RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def remove_stopwords(s, stopwords=None):
    """Remove :const:`~gensim.parsing.preprocessing.STOPWORDS` from `s`.

    Parameters
    ----------
    s : str
    stopwords : iterable of str, optional
        Sequence of stopwords
        If None - using :const:`~gensim.parsing.preprocessing.STOPWORDS`

    Returns
    -------
    str
        Unicode string without `stopwords`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import remove_stopwords
        >>> remove_stopwords("Better late than never, but better never late.")
        u'Better late never, better late.'

    """
    s = utils.to_unicode(s)
    return " ".join(remove_stopword_tokens(s.split(), stopwords))


def remove_stopword_tokens(tokens, stopwords=None):
    """Remove stopword tokens using list `stopwords`.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    stopwords : iterable of str, optional
        Sequence of stopwords
        If None - using :const:`~gensim.parsing.preprocessing.STOPWORDS`

    Returns
    -------
    list of str
        List of tokens without `stopwords`.

    """
    if stopwords is None:
        stopwords = STOPWORDS
    return [token for token in tokens if token not in stopwords]


def strip_punctuation(s):
    """Replace ASCII punctuation characters with spaces in `s` using :const:`~gensim.parsing.preprocessing.RE_PUNCT`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without punctuation characters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_punctuation
        >>> strip_punctuation("A semicolon is a stronger break than a comma, but not as much as a full stop!")
        u'A semicolon is a stronger break than a comma  but not as much as a full stop '

    """
    s = utils.to_unicode(s)
    # For unicode enhancement options see https://github.com/RaRe-Technologies/gensim/issues/2962
    return RE_PUNCT.sub(" ", s)


def strip_tags(s):
    """Remove tags from `s` using :const:`~gensim.parsing.preprocessing.RE_TAGS`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without tags.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_tags
        >>> strip_tags("<i>Hello</i> <b>World</b>!")
        u'Hello World!'

    """
    s = utils.to_unicode(s)
    return RE_TAGS.sub("", s)


def strip_short(s, minsize=3):
    """Remove words with length lesser than `minsize` from `s`.

    Parameters
    ----------
    s : str
    minsize : int, optional

    Returns
    -------
    str
        Unicode string without short words.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_short
        >>> strip_short("salut les amis du 59")
        u'salut les amis'
        >>>
        >>> strip_short("one two three four five six seven eight nine ten", minsize=5)
        u'three seven eight'

    """
    s = utils.to_unicode(s)
    return " ".join(remove_short_tokens(s.split(), minsize))


def remove_short_tokens(tokens, minsize=3):
    """Remove tokens shorter than `minsize` chars.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    minsize : int, optimal
        Minimal length of token (include).

    Returns
    -------
    list of str
        List of tokens without short tokens.
    """

    return [token for token in tokens if len(token) >= minsize]


def strip_numeric(s):
    """Remove digits from `s` using :const:`~gensim.parsing.preprocessing.RE_NUMERIC`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode  string without digits.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_numeric
        >>> strip_numeric("0text24gensim365test")
        u'textgensimtest'

    """
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub("", s)


def strip_non_alphanum(s):
    """Remove non-alphabetic characters from `s` using :const:`~gensim.parsing.preprocessing.RE_NONALPHA`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with alphabetic characters only.

    Notes
    -----
    Word characters - alphanumeric & underscore.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_non_alphanum
        >>> strip_non_alphanum("if-you#can%read$this&then@this#method^works")
        u'if you can read this then this method works'

    """
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(" ", s)


def strip_multiple_whitespaces(s):
    r"""Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
    and turns tabs & line breaks into spaces using :const:`~gensim.parsing.preprocessing.RE_WHITESPACE`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without repeating in a row whitespace characters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_multiple_whitespaces
        >>> strip_multiple_whitespaces("salut" + '\r' + " les" + '\n' + "         loulous!")
        u'salut les loulous!'

    """
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)


def split_alphanum(s):
    """Add spaces between digits & letters in `s` using :const:`~gensim.parsing.preprocessing.RE_AL_NUM`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with spaces between digits & letters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import split_alphanum
        >>> split_alphanum("24.0hours7 days365 a1b2c3")
        u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)


def stem_text(text):
    """Transform `s` into lowercase and stem it.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Unicode lowercased and porter-stemmed version of string `text`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import stem_text
        >>> stem_text("While it is quite useful to be able to search a large collection of documents almost instantly.")
        u'while it is quit us to be abl to search a larg collect of document almost instantly.'

    """
    text = utils.to_unicode(text)
    p = PorterStemmer()
    return ' '.join(p.stem(word) for word in text.split())


stem = stem_text


def lower_to_unicode(text, encoding='utf8', errors='strict'):
    """Lowercase `text` and convert to unicode, using :func:`gensim.utils.any2unicode`.

    Parameters
    ----------
    text : str
        Input text.
    encoding : str, optional
        Encoding that will be used for conversion.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).

    Returns
    -------
    str
        Unicode version of `text`.

    See Also
    --------
    :func:`gensim.utils.any2unicode`
        Convert any string to unicode-string.

    """
    return utils.to_unicode(text.lower(), encoding, errors)


def split_on_space(s):
    """Split line by spaces, used in :class:`gensim.corpora.lowcorpus.LowCorpus`.

    Parameters
    ----------
    s : str
        Some line.

    Returns
    -------
    list of str
        List of tokens from `s`.

    """
    return [word for word in utils.to_unicode(s).strip().split(' ') if word]


DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short, stem_text
]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    """Apply list of chosen filters to `s`.

    Default list of filters:

    * :func:`~gensim.parsing.preprocessing.strip_tags`,
    * :func:`~gensim.parsing.preprocessing.strip_punctuation`,
    * :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces`,
    * :func:`~gensim.parsing.preprocessing.strip_numeric`,
    * :func:`~gensim.parsing.preprocessing.remove_stopwords`,
    * :func:`~gensim.parsing.preprocessing.strip_short`,
    * :func:`~gensim.parsing.preprocessing.stem_text`.

    Parameters
    ----------
    s : str
    filters: list of functions, optional

    Returns
    -------
    list of str
        Processed strings (cleaned).

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import preprocess_string
        >>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
        [u'hel', u'rld', u'weather', u'todai', u'isn']
        >>>
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
    """Apply :const:`~gensim.parsing.preprocessing.DEFAULT_FILTERS` to the documents strings.

    Parameters
    ----------
    docs : list of str

    Returns
    -------
    list of list of str
        Processed documents split by whitespace.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import preprocess_documents
        >>> preprocess_documents(["<i>Hel 9lo</i> <b>Wo9 rld</b>!", "Th3     weather_is really g00d today, isn't it?"])
        [[u'hel', u'rld'], [u'weather', u'todai', u'isn']]

    """
    return [preprocess_string(d) for d in docs]


def read_file(path):
    with utils.open(path, 'rb') as fin:
        return fin.read()


def read_files(pattern):
    return [read_file(fname) for fname in glob.glob(pattern)]
