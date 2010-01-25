#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
USAGE: %s
    Process all articles in directories specified \
in the common.py config file. The directories must be in DML-CZ format. Program \
searches for articles with fulltext.txt and meta.xml files. \

This script has to be run prior to running build_tfidf.py. Its output are \
database files, which serves as input to build_tfidf.py.

Example: ./build_database.py 2>&1 | tee ~/xrehurek/results/build_database.log
"""


import logging
import sys
import os.path
import cPickle
import itertools

import common
import utils_iddb



class Token(object):
    __slots__ = ['token', 'intId', 'surfaceForms']

    def __init__(self, token, intId, surfaceForms):
        self.token = token # preprocessed word form (string)
        self.intId = intId # id of the form (integer)
        self.surfaceForms = set(surfaceForms) # set of all words which map to this token after preprocessing
#endclass Token


class Dictionary(object):
    def __init__(self):
        self.id2token = {} # tokenId -> token
        self.token2id = {} # token -> tokenId
        self.word2id = {} # surface form (word as appearing in text) -> tokenId
        self.docFreq = {} # tokenId -> in how many documents this token appeared
    
    
    def addToken(self, token):
        if token.intId in self.id2token:
            logging.debug("overwriting old token %s (id %i); is this intended?" %
                          (token.token, token.intId))
        self.id2token[token.intId] = token
        self.token2id[token.token] = token.intId
        for surfaceForm in token.surfaceForms:
            self.word2id[surfaceForm] = token.intId
    
    
    def doc2bow(self, document, wordNormalizer = lambda word: word.strip().lower()):
        """
        Convert document (list of words) into bag-of-words format = list of 
        (tokenId, tokenCount) 2-tuples.
        
        Also update dictionary in the process; create ids for new words etc. At
        the same time update document frequencies -- for each word appearing in 
        this document, increase its self.docFreq by one.
        """
        # construct (word, frequency) mapping. in python3 this is done simply 
        # using Counter(), but here i use itertools.groupby()
        result = {}
        for word, group in itertools.groupby(sorted(document)):
            frequency = len(list(group)) # how many times does this word appear in the input document

            # determine the Token object of this word, creating it if necessary
            tokenId = self.word2id.get(word, None)
            if tokenId is None: 
                # first time we see this surface form
                wordNorm = wordNormalizer(word)
                tokenId = self.token2id.get(wordNorm, None)
                if tokenId is None: 
                    # first time we see this token (normalized form)
                    tokenId = len(self.token2id)
                    token = Token(wordNorm, tokenId, set([word]))
                    self.addToken(token)
                else:
                    token = self.id2token[tokenId]
                    # add original word to the set of observed surface forms
                    token.surfaceForms.add(word)
                    self.word2id[word] = tokenId
            else:
                token = self.id2token[tokenId]
            # post condition -- now both tokenId and token object are properly set
            
            # update how many times a token appeared in the document
            result[tokenId] = result.get(tokenId, 0) + frequency
            
        # increase document count for each unique token that appeared in the document
        for tokenId in tokenId, frequency in result.iteritems():
            self.docFreq[tokenId] = self.docFreq.get(tokenId, 0) + 1
        
        return sorted(result.iteritems()) # return tokenIds in ascending order

    
    def __len__(self):
        assert len(self.token2id) == len(self.id2token)
        return len(self.token2id)
#endclass Dictionary


class DmlCorpus(object):
    def __init__(self, config = None):
        self.documents = []
        self.config = config
        self.dictionary = Dictionary()
        if self.config is not None:
            self.processConfig(config)
            self.buildDictionary()

    @staticmethod
    def load(fname):
        logging.info("loading DmlCorpus object from %s" % fname)
        return cPickle.load(open(fname))


    def save(self, fname):
        logging.info("saving DmlCorpus object to %s" % (self, fname))
        f = open(fname, 'w')
        cPickle.dump(self, f)
        f.close()
    
    
    def buildDictionary(self):
        """
        Populate dictionary mapping and statistics.
        
        This is done by sequentially retrieving the article fulltexts, splitting
        them into tokens and converting tokens to their ids (creating new ids as 
        necessary).
        """
        logging.info("creating dictionary for %i articles" % len(self.documents))
        for sourceId, docUri in self.documents:
            source = self.config.sources[sourceId]

            contents = source.getFulltext(docUri)
            words = source.tokenize(contents)
            _ = self.dictionary.doc2bow(words, source.wordNormalizer) # ignore the bow vector -- here we are only interested in updating the dictionary
    
    
    def pruneDictionary(self):
#        self.dictionary = dictionary without too frequent/rare terms
        pass
    
    
    def processConfig(self, config):
        """
        Parse the directories specified in the config, looking for suitable articles.
        
        This updates the self.documents var, which keeps a list of (source id, 
        article uri) 2-tuples. Each tuple is a unique identifier of one article.
        
        Note that some articles are ignored based on config settings (for example 
        if the article's language doesn't match any language specified in the 
        config etc.).
        """
        self.config = config
        logging.info("processing config %s" % config)
        for sourceId, source in config.sources.iteritems():
            logging.info("processing source '%s'" % sourceId)
            accepted = []
            for articleUri in source.findArticles():
                meta = source.getMeta(articleUri) # retrieve metadata (= dictionary of key->value)
                if config.acceptArticle(meta):
                    accepted.append((sourceId, articleUri))
            logging.info("accepted %i articles for source '%s'" % 
                         (len(accepted), sourceId))
            self.documents.extend(accepted)

        if not self.documents:
            logging.warning('no articles at all found from the config; something went wrong!')
        
        logging.info("accepted total of %i articles for %s" % 
                     (len(self.documents), str(config)))
#endclass DmlCorpus


class DmlConfig(object):
    def __init__(self, acceptLangs = None):
        self.sources = {}
        if acceptLangs is None:
            acceptLangs = set(['any'])
        self.acceptLangs = set(acceptLangs)
        logging.info('initialized DmlConfig, accepting %i language(s): [%s]' %
                     (len(self.acceptLangs), ', '.join(self.acceptLangs)))
    
    def acceptArticle(self, metadata):
        lang = metadata.get('language', 'unk') # if there is no language field in the metadata, set language to 'unk' = unknown
        if 'any' not in self.acceptLangs and lang not in self.acceptLangs:
            return False
        return True
    
    def addSource(self, source):
        sourceId = str(source)
        assert sourceId not in self.sources, "source %s already present in the config!" % sourceId
        self.sources[sourceId] = source
        
    def __str__(self):
        return ("DmlConfig(sources=[%s], acceptLangs=[%s])" % 
                (', '.join(self.sources.iterkeys()), ', '.join(self.acceptLangs)))
#endclass DmlConfig


class DmlSource(object):
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
        logging.info("looking for articles for '%s' inside %s" % (self.sourceId, self.baseDir))
        for root, dirs, files in os.walk(self.baseDir):
            dirTotal += 1
            root = os.path.normpath(root)
            if self.isArticle(root):
                artAccepted += 1
                yield self.idFromDir(root)
    
        logging.info('%i directories processed, found %i articles' % 
                     (dirTotal, artAccepted))
    
    def getFulltext(self, uri):
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
        return utils_iddb.parseMeta(filename)
#endclass DmlSource



if __name__ == '__main__':
    logging.basicConfig(level = common.PRINT_LEVEL)
    logging.root.level = common.PRINT_LEVEL
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 1:
        print globals()['__doc__'] % (program)
        sys.exit(1)
    inputs = common.INPUT_PATHS
    prefix = common.PREFIX
    
    # build individual input databases
    # each input database contains ALL articles (not only those with the selected language)
    for id, path in inputs.iteritems():
        iddb.create_maindb(id, path) # create main article databases (all languages)
        gc.collect()
    
    logging.info("finished running %s" % program)
