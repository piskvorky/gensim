#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz


"""
This module implements classes which present DML-CZ articles as a corpus (the 
DmlCorpus class). 

This corpus supports iteration over the bag-of-words representation of its constituent
articles, without the need to keep all articles in memory at the same time.
This allows the corpus to be much larger than the available RAM.

Also included is the DmlConfig class, which encapsulates parameters necessary for
initialization of this corpus. It allows articles to come from different sources,
locations and be in different formats (see the sources module).
"""


import logging
import itertools
import os.path

import matutils
import utils
import dictionary



class DmlConfig(object):
    """
    DmlConfig contains parameters necessary for the abstraction of a 'corpus of 
    articles' (see the DmlCorpus class).
    
    Articles may come from different sources (=different locations on disk/netword,
    different file formats etc), so the main purpose of DmlConfig is to keep all
    sources in one place (= the self.sources attribute).
    
    Apart from glueing together sources, DmlConfig also decides where to store
    output files and which articles to accept for the corpus (= additional filter 
    over the sources).
    """
    def __init__(self, configId, resultDir, acceptLangs = None):
        self.resultDir = resultDir # output files will be stored in this directory 
        self.configId = configId # configId is a string that is used as filename prefix for all files, so keep it simple
        self.sources = {} # all article sources; see sources.DmlSource class for an example of source
        if acceptLangs is None: # which languages to accept
            acceptLangs = set(['any']) # if not specified, accept all languages (including unknown/unspecified)
        self.acceptLangs = set(acceptLangs)
        logging.info('initialized %s' % self)
        
    
    def resultFile(self, fname):
        return os.path.join(self.resultDir, self.configId + '_' + fname)
    
    
    def acceptArticle(self, metadata):
        lang = metadata.get('language', 'unk') # if there was no language field in the article metadata, set language to 'unk' = unknown
        if 'any' not in self.acceptLangs and lang not in self.acceptLangs:
            return False
        return True
    
    
    def addSource(self, source):
        sourceId = str(source)
        assert sourceId not in self.sources, "source %s already present in the config!" % sourceId
        self.sources[sourceId] = source
    
    
    def __str__(self):
        return ("DmlConfig(id=%s, sources=[%s], acceptLangs=[%s])" % 
                (self.configId, ', '.join(self.sources.iterkeys()), ', '.join(self.acceptLangs)))
#endclass DmlConfig



class DmlCorpus(utils.SaveLoad):
    """
    DmlCorpus implements a collection of articles. It is initialized via a DmlConfig
    object, which holds information about where to look for the articles and how 
    to process them.
    
    Apart from being a regular corpus (bag-of-words iterable with a len() function),
    DmlCorpus has methods for building and storing its dictionary (mapping between
    words and their ids).
    """
    def __init__(self):
        self.documents = []
        self.config = None
        self.dictionary = dictionary.Dictionary()

    
    def __len__(self):
        return len(self.documents)


    def __iter__(self):
        """
        The function that defines a corpus -- iterating over the corpus yields 
        bag-of-words vectors, one for each document.
        
        A bag-of-words vector is simply a list of (tokenId, tokenCount) 2-tuples.
        """
        for docNo, (sourceId, docUri) in enumerate(self.documents):
            source = self.config.sources[sourceId]

            contents = source.getContent(docUri)
            words = source.tokenize(contents)
            yield self.dictionary.doc2bow(words, source.normalizeWord, allowUpdate = False)

    
    def buildDictionary(self):
        """
        Populate dictionary mapping and statistics.
        
        This is done by sequentially retrieving the article fulltexts, splitting
        them into tokens and converting tokens to their ids (creating new ids as 
        necessary).
        """
        logging.info("creating dictionary from %i articles" % len(self.documents))
        self.dictionary = dictionary.Dictionary()
        for docNo, (sourceId, docUri) in enumerate(self.documents):
            source = self.config.sources[sourceId]
            contents = source.getContent(docUri)
            words = source.tokenize(contents)
            _ = self.dictionary.doc2bow(words, source.normalizeWord, allowUpdate = True) # ignore the result, here we only care about updating token ids
        logging.info("built %s" % self.dictionary)

    
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
        self.documents = []
        logging.info("processing config %s" % config)
        for sourceId, source in config.sources.iteritems():
            logging.info("processing source '%s'" % sourceId)
            accepted = []
            for articleUri in source.findArticles():
                meta = source.getMeta(articleUri) # retrieve metadata (= dictionary of key->value)
                if config.acceptArticle(meta): # do additional filtering on articles, based on the article's metadata
                    accepted.append((sourceId, articleUri))
            logging.info("accepted %i articles for source '%s'" % 
                         (len(accepted), sourceId))
            self.documents.extend(accepted)

        if not self.documents:
            logging.warning('no articles at all found from the config; something went wrong!')
        
        logging.info("accepted total of %i articles for %s" % 
                     (len(self.documents), str(config)))
        
    def saveAsBow(self):
        """
        Store the corpus to a file, as a term-document matrix with bag-of-words 
        counts in Matrix Market format.
        
        The exact path and filename is determined from the config, but always ends 
        in '*bow.mm'.
        """
        # determine matrix shape and density (only needed for the MM format headers, 
        # which are irrelevant in Python anyway, so it is a HUGE overkill to do 
        # an extra corpus sweep just for that...)
        numDocs, numTerms, numNnz = matutils.MmWriter.determineNnz(self)
        
        # write bow to a file in matrix market format
        outfile = matutils.MmWriter(self.config.resultFile('bow.mm'))
        outfile.writeHeaders(numDocs, numTerms, numNnz)
        outfile.writeCorpus(self)
        outfile.close()
#endclass DmlCorpus

