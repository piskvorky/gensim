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


def remove_stopwords(s):
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)


def strip_punctuation(s):
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(" ", s)


# unicode.translate cannot delete characters like str can
strip_punctuation2 = strip_punctuation
# def strip_punctuation2(s):
#     s = utils.to_unicode(s)
#     return s.translate(None, string.punctuation)


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

DEFAULT_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    return [preprocess_string(d) for d in docs]


def read_file(path):
    with utils.smart_open(path) as fin:
        return fin.read()


def read_files(pattern):
    return [read_file(fname) for fname in glob.glob(pattern)]
