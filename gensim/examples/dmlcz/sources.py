#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains implementations (= different classes) which encapsulate the
idea of a Digital Library document source.

A document source is basically a collection of articles sharing the same format,
same location (type of access), same way of parsing them etc.

Different sources can be aggregated into a single corpus, which is what the
`DmlCorpus` class does (see the `dmlcorpus` module).
"""

import logging
import os
import os.path
import re

import xml.sax  # for parsing arxmliv articles

from gensim import utils

import sys
if sys.version_info[0] >= 3:
    unicode = str

PAT_TAG = re.compile(r'<(.*?)>(.*)</.*?>')
logger = logging.getLogger('gensim.corpora.sources')


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

    def normalizeWord(self, word):
        raise NotImplementedError('Abstract Base Class')
# endclass ArticleSource


class DmlSource(ArticleSource):
    """
    Article source for articles in DML format (DML-CZ, Numdam):
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

    @classmethod
    def parseDmlMeta(cls, xmlfile):
        """
        Parse out all fields from meta.xml, return them as a dictionary.
        """
        result = {}
        xml = open(xmlfile)
        for line in xml:
            if line.find('<article>') >= 0:  # skip until the beginning of <article> tag
                break
        for line in xml:
            if line.find('</article>') >= 0:  # end of <article>, we're done
                break
            p = re.search(PAT_TAG, line)  # HAX assumes one element = one line; proper xml parsing probably better... but who cares
            if p:
                name, cont = p.groups()
                name = name.split()[0]
                name, cont = name.strip(), cont.strip()
                if name == 'msc':
                    if len(cont) != 5:
                        logger.warning('invalid MSC=%s in %s', cont, xmlfile)
                    result.setdefault('msc', []).append(cont)
                    continue
                if name == 'idMR':
                    cont = cont[2:]  # omit MR from MR123456
                if name and cont:
                    result[name] = cont
        xml.close()
        return result

    def idFromDir(self, path):
        assert len(path) > len(self.baseDir)
        intId = path[1 + path.rfind('#'):]
        pathId = path[1 + len(self.baseDir):]
        return (intId, pathId)

    def isArticle(self, path):
        # in order to be valid, the article directory must start with '#'
        if not os.path.basename(path).startswith('#'):
            return False
        # and contain the fulltext.txt file
        if not os.path.exists(os.path.join(path, 'fulltext.txt')):
            logger.info('missing fulltext in %s', path)
            return False
        # and also the meta.xml file
        if not os.path.exists(os.path.join(path, 'meta.xml')):
            logger.info('missing meta.xml in %s', path)
            return False
        return True

    def findArticles(self):
        dirTotal = artAccepted = 0
        logger.info("looking for '%s' articles inside %s", self.sourceId, self.baseDir)
        for root, dirs, files in os.walk(self.baseDir):
            dirTotal += 1
            root = os.path.normpath(root)
            if self.isArticle(root):
                artAccepted += 1
                yield self.idFromDir(root)
        logger.info('%i directories processed, found %i articles', dirTotal, artAccepted)

    def getContent(self, uri):
        """
        Return article content as a single large string.
        """
        intId, pathId = uri
        filename = os.path.join(self.baseDir, pathId, 'fulltext.txt')
        return open(filename).read()

    def getMeta(self, uri):
        """
        Return article metadata as a attribute->value dictionary.
        """
        intId, pathId = uri
        filename = os.path.join(self.baseDir, pathId, 'meta.xml')
        return DmlSource.parseDmlMeta(filename)

    def tokenize(self, content):
        return [token.encode('utf8') for token in utils.tokenize(content, errors='ignore') if not token.isdigit()]

    def normalizeWord(self, word):
        wordU = unicode(word, 'utf8')
        return wordU.lower().encode('utf8')  # lowercase and then convert back to bytestring
# endclass DmlSource


class DmlCzSource(DmlSource):
    """
    Article source for articles in DML-CZ format:
    1) articles = directories starting with '#'
    2) content is stored in fulltext.txt or fulltext_dspace.txt
    3) there exists a dspace_id file, containing internal dmlcz id
    3) metadata are stored in meta.xml

    See the ArticleSource class for general info on sources.
    """

    def idFromDir(self, path):
        assert len(path) > len(self.baseDir)
        dmlczId = open(os.path.join(path, 'dspace_id')).read().strip()
        pathId = path[1 + len(self.baseDir):]
        return (dmlczId, pathId)

    def isArticle(self, path):
        # in order to be valid, the article directory must start with '#'
        if not os.path.basename(path).startswith('#'):
            return False
        # and contain a dspace_id file
        if not (os.path.exists(os.path.join(path, 'dspace_id'))):
            logger.info('missing dspace_id in %s', path)
            return False
        # and contain either fulltext.txt or fulltext_dspace.txt file
        if not (os.path.exists(os.path.join(path, 'fulltext.txt')) or os.path.exists(os.path.join(path, 'fulltext-dspace.txt'))):
            logger.info('missing fulltext in %s', path)
            return False
        # and contain the meta.xml file
        if not os.path.exists(os.path.join(path, 'meta.xml')):
            logger.info('missing meta.xml in %s', path)
            return False
        return True

    def getContent(self, uri):
        """
        Return article content as a single large string.
        """
        intId, pathId = uri
        filename1 = os.path.join(self.baseDir, pathId, 'fulltext.txt')
        filename2 = os.path.join(self.baseDir, pathId, 'fulltext-dspace.txt')

        if os.path.exists(filename1) and os.path.exists(filename2):
            # if both fulltext and dspace files exist, pick the larger one
            if os.path.getsize(filename1) < os.path.getsize(filename2):
                filename = filename2
            else:
                filename = filename1
        elif os.path.exists(filename1):
            filename = filename1
        else:
            assert os.path.exists(filename2)
            filename = filename2
        return open(filename).read()
# endclass DmlCzSource


class ArxmlivSource(ArticleSource):
    """
    Article source for articles in arxmliv format:
    1) articles = directories starting with '#'
    2) content is stored in tex.xml
    3) metadata in special tags within tex.xml

    Article URI is currently (a part of) the article's path on filesystem.

    See the ArticleSource class for general info on sources.
    """
    class ArxmlivContentHandler(xml.sax.handler.ContentHandler):
        def __init__(self):
            self.path = ['']  # help structure for sax event parsing
            self.tokens = []  # will contain tokens once parsing is finished

        def startElement(self, name, attr):
            # for math tokens, we only care about Math elements directly below <p>
            if name == 'Math' and self.path[-1] == 'p' and attr.get('mode', '') == 'inline':
                tex = attr.get('tex', '')
                if tex and not tex.isdigit():
                    self.tokens.append('$%s$' % tex.encode('utf8'))
            self.path.append(name)

        def endElement(self, name):
            self.path.pop()

        def characters(self, text):
            # for text, we only care about tokens directly within the <p> tag
            if self.path[-1] == 'p':
                tokens = [token.encode('utf8') for token in utils.tokenize(text, errors='ignore') if not token.isdigit()]
                self.tokens.extend(tokens)
    # endclass ArxmlivHandler

    class ArxmlivErrorHandler(xml.sax.handler.ErrorHandler):
        # Python2.5 implementation of xml.sax is broken -- character streams and
        # byte encodings of InputSource are ignored, bad things sometimes happen
        # in buffering of multi-byte files (such as utf8), characters get cut in
        # the middle, resulting in invalid tokens...
        # This is not really a problem with arxmliv xml files themselves, so ignore
        # these errors silently.
        def error(self, exception):
            pass

        warning = fatalError = error
    # endclass ArxmlivErrorHandler

    def __init__(self, sourceId, baseDir):
        self.sourceId = sourceId
        self.baseDir = os.path.normpath(baseDir)

    def __str__(self):
        return self.sourceId

    def idFromDir(self, path):
        assert len(path) > len(self.baseDir)
        intId = path[1 + path.rfind('#'):]
        pathId = path[1 + len(self.baseDir):]
        return (intId, pathId)

    def isArticle(self, path):
        # in order to be valid, the article directory must start with '#'
        if not os.path.basename(path).startswith('#'):
            return False
        # and contain the tex.xml file
        if not os.path.exists(os.path.join(path, 'tex.xml')):
            logger.warning('missing tex.xml in %s', path)
            return False
        return True

    def findArticles(self):
        dirTotal = artAccepted = 0
        logger.info("looking for '%s' articles inside %s", self.sourceId, self.baseDir)
        for root, dirs, files in os.walk(self.baseDir):
            dirTotal += 1
            root = os.path.normpath(root)
            if self.isArticle(root):
                artAccepted += 1
                yield self.idFromDir(root)
        logger.info('%i directories processed, found %i articles', dirTotal, artAccepted)

    def getContent(self, uri):
        """
        Return article content as a single large string.
        """
        intId, pathId = uri
        filename = os.path.join(self.baseDir, pathId, 'tex.xml')
        return open(filename).read()

    def getMeta(self, uri):
        """
        Return article metadata as an attribute->value dictionary.
        """
#        intId, pathId = uri
#        filename = os.path.join(self.baseDir, pathId, 'tex.xml')
        return {'language': 'eng'}  # TODO maybe parse out some meta; but currently not needed for anything...

    def tokenize(self, content):
        """
        Parse tokens out of xml. There are two types of token: normal text and
        mathematics. Both are returned interspersed in a single list, in the same
        order as they appeared in the content.

        The math tokens will be returned in the form $tex_expression$, ie. with
        a dollar sign prefix and suffix.
        """
        handler = ArxmlivSource.ArxmlivContentHandler()
        xml.sax.parseString(content, handler, ArxmlivSource.ArxmlivErrorHandler())
        return handler.tokens

    def normalizeWord(self, word):
        if word[0] == '$':  # ignore math tokens
            return word
        wordU = unicode(word, 'utf8')
        return wordU.lower().encode('utf8')  # lowercase and then convert back to bytestring
# endclass ArxmlivSource
