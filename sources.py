#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
This module contains implementations (= different classes) which encapsulate the
idea of a document source. 

A document source is basically a collection of articles sharing the same format, 
same location (type of access), same way of parsing them etc.

Different sources can be aggregated into a single corpus, which is what the 
DmlCorpus class does (see the corpus.py module).
"""

import logging
import os
import os.path
import re

import utils


PAT_TAG = re.compile('<(.*?)>(.*)</.*?>')


class ArticleSource(object):
    """
    Objects of this class describe a single source of articles.
    
    A source is an abstraction over where the documents reside (the findArticles() 
    method), how to retrieve their fulltexts, their metadata, how to tokenize the 
    articles and how to normalize the tokens.
    
    What is NOT abstracted away (ie. must hold for all sources) is the idea of
    article identifiers (URIs), which uniquely identify each article within 
    one source. 

    This class is just an ABC interface; see eg. DmlSource or ArxmlivSource classes 
    for concrete instances.
    """
    def __init__(self, sourceId):
        self.sourceId = sourceId
    
    def __str__(self):
        return self.sourceId

    def findArticles(self):
        raise NotImplementedError('Abstract Base Class')
    
    def getContent(self, uri):
        raise NotImplementedError('Abstract Base Class')

    def getMeta(self, uri):
        raise NotImplementedError('Abstract Base Class')

    def tokenize(self, content):
        raise NotImplementedError('Abstract Base Class')
    
    def wordNormalizer(self, word):
        raise NotImplementedError('Abstract Base Class')
#endclass Source



class DmlSource(ArticleSource):
    """
    Article source for articles in DML-CZ format: 
    1) articles = directories starting with '#'
    2) content is stored in fulltext.txt
    3) metadata are stored in meta.xml 
    
    Article URI is currently (a part of) the article's path on filesystem.
    
    See the ArticleSource class for general info on sources. 
    """
    def __init__(self, sourceId, baseDir):
        self.sourceId = sourceId
        self.baseDir = os.path.normpath(baseDir)
    
    def __str__(self):
        return self.sourceId
    
    
    def idFromDir(self, path):
        assert len(path) > len(self.baseDir)
        return path[len(self.baseDir) + 1 : ] # by default, internal id is the filesystem path following the base dir
    
    
    def isArticle(self, path):
        # in order to be valid, the article directory must start with '#'
        if not os.path.basename(path).startswith('#'): # all dml-cz articles are stored in directories starting with '#'
            return False
        # and contain the fulltext.txt file
        if not os.path.exists(os.path.join(path, 'fulltext.txt')):
            return False
        # and also the meta.xml file
        if not os.path.exists(os.path.join(path, 'meta.xml')):
            return False
        return True
    
    
    def findArticles(self):
        dirTotal = artAccepted = 0
        logging.info("looking for '%s' articles inside %s" % (self.sourceId, self.baseDir))
        for root, dirs, files in os.walk(self.baseDir):
            dirTotal += 1
            root = os.path.normpath(root)
            if self.isArticle(root):
                artAccepted += 1
                yield self.idFromDir(root)
    
        logging.info('%i directories processed, found %i articles' % 
                     (dirTotal, artAccepted))
    
    
    def getContent(self, uri):
        """
        Return article content as a single large string.
        """
        filename = os.path.join(self.baseDir, uri, 'fulltext.txt')
        return open(filename).read()
    
    
    def getMeta(self, uri):
        """
        Return article metadata as a attribute->value dictionary.
        """
        filename = os.path.join(self.baseDir, uri, 'meta.xml')
        return parseMeta(filename)
    
    
    def tokenize(self, content):
        return [token.encode('utf8') for token in utils.tokenize(content)]
    
    
    def wordNormalizer(self, word):
        wordU = unicode(word, 'utf8')
        return wordU.lower().encode('utf8') # lowercase and then convert back to bytestring
#endclass DmlSource



def parseMeta(xmlfile):
    """
    Parse out all fields from meta.xml, return them as a dictionary.
    """
    result = {}
    xml = open(xmlfile)
    for line in xml:
        if line.find('<article>') >= 0: # skip until the beginning of <article> tag
            break
    for line in xml:
        if line.find('</article>') >= 0: # end of <article>, we're done
            break
        p = re.search(PAT_TAG, line) # HAX assumes one element = one line; proper xml parsing probably better... but who cares
        if p:
            name, cont = p.groups()
            name = name.split()[0]
            name, cont = name.strip(), cont.strip()
            if name == 'msc':
                if len(cont) != 5:
                    logging.warning('invalid MSC=%s in %s' % (cont, xmlfile))
                result.setdefault('msc', []).append(cont)
                continue
            if name == 'idMR':
                cont = cont[2:] # omit MR from MR123456
            if name and cont:
                result[name] = cont
    xml.close()
    return result
