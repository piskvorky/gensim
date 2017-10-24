#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.summarization.syntactic_unit import SyntacticUnit
from gensim.parsing.preprocessing import preprocess_documents
from gensim.utils import tokenize
from six.moves import xrange
import re
import logging

logger = logging.getLogger('summa.preprocessing.cleaner')

try:
    from pattern.en import tag
    logger.info("'pattern' package found; tag filters are available for English")
    HAS_PATTERN = True
except ImportError:
    logger.info("'pattern' package not found; tag filters are not available for English")
    HAS_PATTERN = False


SEPARATOR = r"@"
RE_SENTENCE = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE | re.DOTALL)
AB_SENIOR = re.compile("([A-Z][a-z]{1,2}\.)\s(\w)", re.UNICODE)
AB_ACRONYM = re.compile("(\.[a-zA-Z]\.)\s(\w)", re.UNICODE)
AB_ACRONYM_LETTERS = re.compile("([a-zA-Z])\.([a-zA-Z])\.", re.UNICODE)
UNDO_AB_SENIOR = re.compile("([A-Z][a-z]{1,2}\.)" + SEPARATOR + "(\w)", re.UNICODE)
UNDO_AB_ACRONYM = re.compile("(\.[a-zA-Z]\.)" + SEPARATOR + "(\w)", re.UNICODE)


def split_sentences(text):
    processed = replace_abbreviations(text)
    return [undo_replacement(sentence) for sentence in get_sentences(processed)]


def replace_abbreviations(text):
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])


def undo_replacement(sentence):
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def get_sentences(text):
    for match in RE_SENTENCE.finditer(text):
        yield match.group()


def merge_syntactic_units(original_units, filtered_units, tags=None):
    units = []
    for i in xrange(len(original_units)):
        if filtered_units[i] == '':
            continue

        text = original_units[i]
        token = filtered_units[i]
        tag = tags[i][1] if tags else None
        sentence = SyntacticUnit(text, token, tag)
        sentence.index = i

        units.append(sentence)

    return units


def join_words(words, separator=" "):
    return separator.join(words)


def clean_text_by_sentences(text):
    """ Tokenizes a given text into sentences, applying filters and lemmatizing them.
    Returns a SyntacticUnit list. """
    original_sentences = split_sentences(text)
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return merge_syntactic_units(original_sentences, filtered_sentences)


def clean_text_by_word(text, deacc=True):
    """ Tokenizes a given text into words, applying filters and lemmatizing them.
    Returns a dict of word -> syntacticUnit. """
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    original_words = list(tokenize(text_without_acronyms, to_lower=True, deacc=deacc))
    filtered_words = [join_words(word_list, "") for word_list in preprocess_documents(original_words)]
    if HAS_PATTERN:
        tags = tag(join_words(original_words))  # tag needs the context of the words in the text
    else:
        tags = None
    units = merge_syntactic_units(original_words, filtered_words, tags)
    return {unit.text: unit for unit in units}


def tokenize_by_word(text):
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    return tokenize(text_without_acronyms, to_lower=True, deacc=True)
