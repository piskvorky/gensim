#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Text Cleaner

This module contains functions and processors used for processing text, 
extracting sentences from text, working with acronyms and abbreviations.
"""


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


SEPARATOR = r'@'
"""str: special separator used in abbreviations."""
RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)  # backup (\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)
"""SRE_Pattern: pattern to split text to sentences."""
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
"""SRE_Pattern: pattern for detecting abbreviations. (Example: Sgt. Pepper)"""
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
"""SRE_Pattern: one more pattern for detecting acronyms."""
AB_ACRONYM_LETTERS = re.compile(r'([a-zA-Z])\.([a-zA-Z])\.', re.UNICODE)
"""SRE_Pattern: one more pattern for detecting acronyms. 
(Example: P.S. I love you)"""
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + SEPARATOR + r'(\w)', re.UNICODE)
"""SRE_Pattern: Pattern like AB_SENIOR but with SEPARATOR between abbreviation 
and next word"""
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + SEPARATOR + r'(\w)', re.UNICODE)
"""SRE_Pattern: Pattern like AB_ACRONYM but with SEPARATOR between abbreviation 
and next word"""


def split_sentences(text):
    """Splits and returns list of sentences from given text. It preserves 
    abbreviations set in `AB_SENIOR` and `AB_ACRONYM`. 

    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    str:
        List of sentences from text.
    """
    processed = replace_abbreviations(text)
    return [undo_replacement(sentence) for sentence in get_sentences(processed)]


def replace_abbreviations(text):
    """Replaces blank space to @ separator after abbreviation and next word.

    Parameters
    ----------
    sentence : str
        Input sentence.
    
    Returns
    -------
    str:
        Sentence with changed separator.
        
    Example
    -------
    >>> replace_abbreviations("God bless you, please, Mrs. Robinson")
    God bless you, please, Mrs.@Robinson
    """
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])


def undo_replacement(sentence):
    """Replaces `@` separator back to blank space after each abbreviation.

    Parameters
    ----------
    sentence : str
        Input sentence.
    
    Returns
    -------
    str:
        Sentence with changed separator.
        
    Example
    -------
    >>> undo_replacement("God bless you, please, Mrs.@Robinson")
    God bless you, please, Mrs. Robinson
    """
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def replace_with_separator(text, separator, regexs):
    """Returns text with replaced separator if provided regular expressions 
    were matched.

    Parameters
    ----------
    text : str
        Input text.
    separator : str
        The separator between words to be replaced.
    regexs : str
        List of regular expressions.
    
    Returns
    -------
    str
        Text with replaced separators.
    """
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def get_sentences(text):
    """Sentence generator from provided text. Sentence pattern set in `RE_SENTENCE`.

    Parameters
    ----------
    text : str
        Input text.
    
    Yields
    ------
    str
        Single sentence extracted from text.

    Example
    -------
    >>> text = "Does this text contains two sentences? Yes, it is."
    >>> for sentence in get_sentences(text):
    >>>     print(sentence)
    Does this text contains two sentences?
    Yes, it is.
    """
    for match in RE_SENTENCE.finditer(text):
        yield match.group()


def merge_syntactic_units(original_units, filtered_units, tags=None):
    """Processes given sentences and its filtered (tokenized) copies into 
    SyntacticUnit type. Also adds tags if they are provided to produced units.
    Returns a SyntacticUnit list. 

    Parameters
    ----------
    original_units : list
        List of original sentences.
    filtered_units : list
        List of tokenized sentences.
    tags : list
        List of strings used as tags for each unit. None as deafault.
    
    Returns
    -------
    list
        SyntacticUnit for each input item.
    """
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
    """Merges words to a string using separator (blank space as default).

    Parameters
    ----------
    words : list
        List of words.
    separator : str
        The separator bertween elements. Blank set as default.
    
    Returns
    -------
    str
        String of merged words with separator between them.
    """
    return separator.join(words)


def clean_text_by_sentences(text):
    """Tokenizes a given text into sentences, applying filters and lemmatizing them.
    Returns a SyntacticUnit list. 

    Parameters
    ----------
    text : list
        Input text.
    
    Returns
    -------
    list
        SyntacticUnit objects for each sentence.
    """
    original_sentences = split_sentences(text)
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return merge_syntactic_units(original_sentences, filtered_sentences)


def clean_text_by_word(text, deacc=True):
    """Tokenizes a given text into words, applying filters and lemmatizing them.
    Returns a dictionary of word -> syntacticUnit. Note that different words may
    lead to same processed unit.

    Parameters
    ----------
    text : list
        Input text.
    deacc : bool
        Remove accentuation (default True).
    
    Returns
    -------
    dictionary
        Word as key, SyntacticUnit as value of dictionary.

    Example
    -------
    >>> from gensim.summarization.textcleaner import clean_text_by_word
    >>> clean_text_by_word("God helps those who help themselves")
    {'god': Original unit: 'god' *-*-*-* Processed unit: 'god',
    'help': Original unit: 'help' *-*-*-* Processed unit: 'help',
    'helps': Original unit: 'helps' *-*-*-* Processed unit: 'help'}

    """
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
    """Tokenizes input text. Before tokenizing transforms text to lower case and
    removes accentuation and acronyms set `AB_ACRONYM_LETTERS`. 
    Returns generator of words.

    Parameters
    ----------
    text : list
        Input text.
    
    Returns
    -------
    generator
        Words contained in processed text.

    """
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    return tokenize(text_without_acronyms, to_lower=True, deacc=True)
