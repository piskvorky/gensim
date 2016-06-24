#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""ShakesCorpus is an iterable over the lines of The Complete Workds of William Shakespeare

This module provides the ShakesCorpus class and functions for segmenting
Gutenberg Project books with formatting similar to the text file used for this Corpus.

source text: 'gensim/test/test_data/shakespeare-complete-works.txt.gz'
source meta-data: 'gensim/test/test_data/shakespeare-complete-works-meta.json'
"""
from __future__ import with_statement

import gzip
import re
import json
import logging

# from six import string_types


from gensim import utils
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary

logger = logging.getLogger('gensim.corpora.shakescorpus')

# FIXME:
# module_path = os.path.dirname(__file__)

PATH_SHAKESPEARE = utils.datapath('shakespeare-complete-works.txt.gz')
DICT_ROMAN2INT = {'I': 1, 'II': 2, 'III': 3, 'IV': 4,  'V': 5,
                  'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
for num_X in range(1, 5):
    for s, num in DICT_ROMAN2INT.items():
        DICT_ROMAN2INT['X' * num_X + s] = 10 + num
PATH_SHAKES = utils.datapath('shakespeare-complete-works.txt.gz')
META_SHAKES = json.load(open(utils.datapath('shakespeare-complete-works-meta.json'), 'rU'))

RE_TITLE = re.compile(r'(([-;,\'A-Z]+[ ]?){3,8})')
RE_TITLE_LINE = re.compile(r'^' + RE_TITLE.pattern + r'$')
RE_GUTEN_LINE = re.compile(r'^\*\*\*\ START\ OF\ THIS\ PROJECT\ GUTENBERG\ EBOOK\ ' +
                           RE_TITLE.pattern + r'\ \*\*\*$')
RE_ACT_SCENE_LINE = re.compile(r'^((ACT\ [IV]+)[.]?\ (SCENE\ [0-9]{1,2})[.]?)$')
RE_YEAR_LINE = re.compile(r'^1[56][0-9]{2}$')
RE_THE_END = re.compile(r'^THE[ ]END$')
RE_BY_LINE = re.compile(r'^((by\ )?(William\ Shakespeare))$', re.IGNORECASE)


def generate_lines(input_file,
                   start=0,
                   stop=float('inf')):
    """Generate (yield) lines in a gzipped file (*.txt.gz) one line at a time"""
    with gzip.GzipFile(input_file, 'rU') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= stop:
                break
            yield line.rstrip()


def segment_shakespeare_works(input_file=PATH_SHAKESPEARE, verbose=False):
    """Find start and end of each volume within _Complete Works of William Shakespeare_

    """
    works = [{}]
    meta = {}
    j = 0
    for i, line in enumerate(generate_lines(input_file=input_file)):
        if 'title' not in meta:
            match = RE_GUTEN_LINE.match(line)
            if match:
                meta['title'] = match.groups()[0]
                meta['body_start'] = i
            continue
        if j >= len(works):
            works += [{}]
        if not len(works[j]):
            match = RE_YEAR_LINE.match(line)
            if match:
                if verbose:
                    print(" year {:02d}, {}: {}".format(j, i, match.group()))
                works[j]['year'] = int(match.group())
                works[j]['start'] = i
        elif len(works[j]) == 2:
            match = RE_TITLE_LINE.match(line)
            if match:
                if verbose:
                    print("title {:02d}, {}: {}".format(j, i, match.groups()[0]))
                works[j]['title'] = match.groups()[0]
                works[j]['title_lineno'] = i
        elif len(works[j]) == 4:
            match = RE_BY_LINE.match(line)
            if match:
                if verbose:
                    print("   by {:02d}, {}: {}".format(j, i, match.group()))
                works[j]['by'] = match.groups()[2]
                works[j]['by_lineno'] = i
        elif len(works[j]) > 4:
            match = RE_ACT_SCENE_LINE.match(line)
            if match:
                section_meta = {
                    'start': i,
                    'title': match.groups()[0],
                    'act_roman': match.groups()[1].split()[-1],
                    'act': int(DICT_ROMAN2INT[match.groups()[1].split()[-1]]),
                    'scene': int(match.groups()[2].split()[-1]),
                }
                works[j]['sections'] = works[j].get('sections', []) + [section_meta]
            else:
                match = RE_THE_END.match(line)
                if match and 'GUTENBERG' not in match.group().upper():
                    if verbose:
                        print(" stop {:02d}, {}: {}".format(j, i, match.group()))
                    works[j]['stop'] = i
                    j += 1
    if not len(works[-1]):
        works = works[:-1]
    meta['volumes'] = works
    return meta


class ShakesCorpus(TextCorpus):
    """Iterable, memory-efficient sequence of BOWs (bag of words vectors) for each line in Shakespeare's words"""
    def __init__(self, input_file=PATH_SHAKES, lemmatize=False, lowercase=False, dictionary=None, filter_namespaces=('0',), metadata=False):
        """Initialize a Corpus of the lines in Shakespeare's Collected Works

        Unless a dictionary is provided, this scans the corpus once, to determine its vocabulary.
        This Corpus should not be used with any other input_file than that provided in gensim/test/test_data.

        >>> shakes = ShakesCorpus()
        >>> for i, tokens in enumerate(shakes.get_texts()):
        ...     print(i, tokens)
        ...     if i >= 4:
        ...         break
        (0, [])
        (1, [])
        (2, [u'THE', u'SONNETS'])
        (3, [])
        (4, [u'by', u'William', u'Shakespeare'])
        >>> for i, vec in enumerate(shakes):
        ...     print(i, vec)
        ...     if i >= 4:
        ...         break
        (0, [])
        (1, [])
        (2, [(0, 1), (1, 1)])
        (3, [])
        (4, [(2, 1), (3, 1), (4, 1)])
        """
        if input_file is None:
            raise(ValueError('ShakesCorpus requires an input document which it preprocesses to compute ' +
                             'the Dictionary and `book_meta` information (title, sections, etc).'))
        super(ShakesCorpus, self).__init__(input=None, metadata=metadata)
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        if input_file is None:
            self.book_meta = dict(META_SHAKES)
        else:
            logger.warn('This ShakesCorpus is only intended for use with the gzipped text file from the ' +
                        'Gutenberg project which comes with gensim.')
            self.book_meta = segment_shakespeare_works(input_file)
        self.input_file_path = input_file or PATH_SHAKES
        self.dictionary = Dictionary(self.get_texts(metadata=False))

    def get_texts(self, metadata=None):
        """Iterate over the lines of "The Complete Works of William Shakespeare".

        This yields lists of strings (**texts**) rather than vectors (vectorized bags-of-words).
        And the **texts** yielded are lines rather than entire plays or sonnets.
        If you want vectors, use the corpus interface instead of this method.

        >>> shakes = ShakesCorpus(lowercase=True)
        >>> for i, tokens in enumerate(shakes.get_texts()):
        ...     print(i, tokens)
        ...     if i >= 4:
        ...         break
        (0, [])
        (1, [])
        (2, [u'the', u'sonnets'])
        (3, [])
        (4, [u'by', u'william', u'shakespeare'])
        """
        if metadata is None:
            metadata = self.metadata
        self.input_file = gzip.GzipFile(self.input_file_path)
        volume_num = 0
        with self.input_file as lines:
            for lineno, line in enumerate(lines):
                if volume_num >= len(self.book_meta['volumes']):
                    raise StopIteration()
                if lineno < self.book_meta['volumes'][volume_num]['start']:
                    continue
                if lineno < self.book_meta['volumes'][volume_num]['stop']:
                    # act_num, scene_num = 0, 0  # FIXME: use self.book_meta['volumes'][volume_num]['sections']
                    if metadata:
                        # FIXME: use self.lemmatize
                        toks = self.tokenize(line, lowercase=self.lowercase)
                        yield (toks, (lineno,))
                    else:
                        toks = self.tokenize(line, lowercase=self.lowercase)
                        yield toks
                else:
                    volume_num += 1  # don't yield the "THE END" line?

    def tokenize(self, line, **kwargs):
        return list(utils.tokenize(line, **kwargs))

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.get_texts())
        return self.length

# endclass TextCorpus
