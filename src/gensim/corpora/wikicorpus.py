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
uncompressed and may be of arbitrary size.
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
    return [token.encode('utf8') for token in utils.tokenize(content, lower = True, errors = 'ignore') 
            if len(token) <= 15]



class WikiCorpus(interfaces.CorpusABC):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a corpus.
    
    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.
    """
    
    def __init__(self, fname):
        """
        Initialize the corpus. This scans the corpus once to determine its vocabulary.
        
        **note** The scan takes more than 9 hours on the June 2010 wiki dump (
        `enwiki-20100622-pages-articles.xml.bz2` of 6GB)!
        """
        self.fname = fname
        self.buildDictionary()

    
    def __len__(self):
        raise RuntimeError("len(wiki) is too costly!")


    def __iter__(self):
        """
        The function that defines a corpus -- iterating over the corpus yields 
        vectors, one for each document.
        """
        for docNo, text in enumerate(self.getArticles()):
            yield self.dictionary.doc2bow(tokenize(text), allowUpdate = False)

    
    def buildDictionary(self):
        """
        Populate dictionary mapping and statistics.
        
        This is done by sequentially retrieving the article fulltexts, splitting
        them into tokens and converting tokens to their ids (creating new ids as 
        necessary).
        """
        logger.info("creating dictionary from wiki dump %s" % self.fname)
        self.dictionary = dictionary.Dictionary(id2word = False)
        numPositions = 0
        for docNo, text in enumerate(self.getArticles()):
            if docNo % 1000 == 0:
                logger.info("PROGRESS: at document #%i, %s" % (docNo, self.dictionary))
            words = tokenize(text)
            numPositions += len(words)
            # convert to bag-of-words, but ignore the result -- here we only care about updating token ids
            _ = self.dictionary.doc2bow(tokenize(text), allowUpdate = True)
        logger.info("built %s from %i documents (total %i corpus positions)" % 
                     (self.dictionary, docNo + 1, numPositions))
    
    
    def saveDictionary(self, fname):
        """
        Store id->word mapping to a file.
        """
        logger.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for token, tokenId in self.dictionary.token2id.iteritems():
            fout.write("%i\t%s\n" % (tokenId, token))
        fout.close()
    
    
    @staticmethod
    def loadDictionary(fname):
        """
        Load previously stored mapping between words and their ids.
        
        The result can be used as the `id2word` parameter for input to transformations.
        """
        result = {}
        for lineNo, line in enumerate(open(fname)):
            pair = line[:-1].split('\t')
            if len(pair) != 2:
                continue
            wordId, word = pair
            result[int(wordId)] = word
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
        matutils.MmWriter.writeCorpus(fname + '_bow.mm', self)
        
    
    def getArticles(self):
        """
        Iterate over the dump, returning text version of each article.
        
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).
        """
        intext, lines = False, []
        for lineno, line in enumerate(bz2.BZ2File(self.fname)):
        #for lineno, line in enumerate(open(self.fname, 'rb')):
            if line.startswith('      <text'):
                intext = True
            elif intext:
                lines.append(line)
            pos = line.find('</text>') # can be on the same line as <text>
            if pos >= 0:
                intext = False
                if not lines:
                    continue
                lines[-1] = line[:pos]
                text = filterWiki(''.join(lines))
                lines = []
                if len(text) > ARTICLE_MIN_CHARS: # article redirects are removed here
                    yield text
#endclass WikiCorpus

