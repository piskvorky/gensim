#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Wikipedia corpus.

This wraps a compressed XML Wikipedia dump on disk so that iterating over the 
corpus yields directly bag-of-words sparse vectors.

The input compressed file is processed incrementally, so that it need not be 
uncompressed in whole.
"""


import logging
import itertools
import os.path
import re
import bz2

from gensim import interfaces, matutils, utils
from gensim.corpora import dictionary # for constructing word->id mappings



logger = logging.getLogger('wikicorpus')
logger.setLevel(logging.INFO)


ARTICLE_MIN_CHARS = 500


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE)
RE_P3 = re.compile("{{([^}{]*)}}", re.UNICODE)
RE_P4 = re.compile("{{([^}]*)}}", re.UNICODE)
RE_P5 = re.compile("\[([^][|]*?):\/\/(.*?)\]", re.UNICODE)
RE_P6 = re.compile("\[([^]|[:]*)\|([^][]*)\]", re.UNICODE)
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)


def filterWiki(raw):
    """
    Filter out wiki markup from utf8 string `raw`, leaving only text.
    """
    # the parsing is far form perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.decode_htmlentities(unicode(raw, 'utf8', 'ignore'))
    text = re.sub(RE_P0, "", text) # remove comments
    text = re.sub(RE_P1, '', text) # remove footnotes
    text = re.sub(RE_P2, "", text) # remove the last list (=languages)
    text = re.sub(RE_P3, '', text) # remove templates
    text = re.sub(RE_P4, '', text) # remove templates (no recursion, only 2-level)
    text = re.sub(RE_P5, '', text) # remove urls
    text = re.sub(RE_P6, '\\2', text) # simplify links, keep description only
    text = re.sub(RE_P7, '\\3', text) # simplify images, keep description only
    text = re.sub(RE_P8, '\\3', text) # simpligy files, keep description only
    return text


def tokenize(content):
    """
    Tokenize a piece of text from wikipedia. The input string `content` is assumed
    to be mark-up free (see `filterWiki()`).
    
    Return tokens as utf8 bytestrings. 
    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic etc.)
    return [token.encode('utf8') for token in utils.tokenize(content, lower = True, errors = 'ignore') 
            if len(token) <= 15]



class WikiCorpus(interfaces.CorpusABC):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a (read-only) corpus.
    
    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.
    
    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id, takes almost 7h
    >>> wiki.saveAsText('wiki_en_vocab200k') # another 7.5h, creates a file in MatrixMarket format plus file with id->word
    
    """
    def __init__(self, fname, noBelow = 20, keep_words = 200000, dictionary = None):
        """
        Initialize the corpus. This scans the corpus once, to determine its 
        vocabulary (only the first `keep_words` most frequent words that 
        appear in at least `noBelow` documents are kept).
        """
        self.fname = fname
        if dictionary is None:
            self.dictionary = dictionary.Dictionary(self.getArticles())
            self.dictionary.filterExtremes(noBelow = noBelow, noAbove = 0.1, keepN = keep_words)
        else:
            self.dictionary = dictionary

    
    def __len__(self):
        return self.numDocs


    def __iter__(self):
        """
        The function that defines a corpus -- iterating over the corpus yields 
        vectors, one for each document.
        """
        for docNo, text in enumerate(self.getArticles()):
            yield self.dictionary.doc2bow(text, allowUpdate = False)

        
    def saveDictionary(self, fname):
        """
        Store id->word mapping to a file, in format `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.
        """
        logger.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for token, tokenId in sorted(self.dictionary.token2id.iteritems()):
            fout.write("%i\t%s\t%i\n" % (tokenId, token, self.dictionary.docFreq[tokenId]))
        fout.close()
    
    
    @staticmethod
    def loadDictionary(fname):
        """
        Load previously stored mapping between words and their ids.
        
        The result can be used as the `id2word` parameter for input to transformations.
        """
        result = {}
        for lineNo, line in enumerate(open(fname)):
            cols = line[:-1].split('\t')
            if len(cols) == 2:
                wordId, word = cols
            elif len(cols) == 3:
                wordId, word, docFreq = cols
            else:
                continue
            result[int(wordId)] = word # docFreq not used
        return result
    
    
    def saveAsText(self, fname):
        """
        Store the corpus to disk, in a human-readable text format.
        
        This actually saves two files:
        
        1. Document-term co-occurence frequency counts (bag-of-words), as 
           a Matrix Market file `fname_bow.mm`.
        2. Token to integer mapping, as a text file `fname_wordids.txt`.
        
        """
        self.saveDictionary(fname + '_wordids.txt')
        matutils.MmWriter.writeCorpus(fname + '_bow.mm', self, progressCnt = 10000)
        
    
    def getArticles(self):
        """
        Iterate over the dump, returning text version of each article.
        
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).
        """
        articles, intext = 0, False
        for lineno, line in enumerate(bz2.BZ2File(self.fname)):
            if line.startswith('      <text'):
                intext = True
                lines = []
            elif intext:
                lines.append(line)
            pos = line.find('</text>') # can be on the same line as <text>
            if pos >= 0:
                intext = False
                if not lines:
                    continue
                lines[-1] = line[:pos]
                text = filterWiki(''.join(lines))
                if len(text) > ARTICLE_MIN_CHARS: # article redirects are removed here
                    articles += 1
                    yield tokenize(text) # split text into tokens
        
        self.numDocs = articles # cache corpus length
#endclass WikiCorpus

