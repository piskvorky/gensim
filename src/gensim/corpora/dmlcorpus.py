#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus for the DML-CZ project.
"""


import logging
import itertools
import os.path

from gensim import interfaces, matutils
import dictionary # for constructing word->id mappings


class DmlConfig(object):
    """
    DmlConfig contains parameters necessary for the abstraction of a 'corpus of 
    articles' (see the DmlCorpus class).
    
    Articles may come from different sources (=different locations on disk/netword,
    different file formats etc), so the main purpose of DmlConfig is to keep all
    sources in one place (= the self.sources attribute).
    
    Apart from glueing together sources, DmlConfig also decides where to store
    output files and which articles to accept for the corpus (= an additional filter 
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



class DmlCorpus(interfaces.CorpusABC):
    """
    DmlCorpus implements a collection of articles. It is initialized via a DmlConfig
    object, which holds information about where to look for the articles and how 
    to process them.
    
    Apart from being a regular corpus (bag-of-words iterable with a len() method),
    DmlCorpus has methods for building a dictionary (mapping between words and 
    their ids).
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
        numPositions = 0
        for docNo, (sourceId, docUri) in enumerate(self.documents):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: at document #%i/%i (%s, %s)" % 
                             (docNo, len(self.documents), sourceId, docUri))
            source = self.config.sources[sourceId]
            contents = source.getContent(docUri)
            words = source.tokenize(contents)
            numPositions += len(words)

            # convert to bag-of-words, but ignore the result -- here we only care about updating token ids
            _ = self.dictionary.doc2bow(words, source.normalizeWord, allowUpdate = True)
        logging.info("built %s from %i documents (total %i corpus positions)" % 
                     (self.dictionary, len(self.documents), numPositions))

    
    def processConfig(self, config, shuffle = False):
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
        
        if shuffle:
            logging.info("shuffling the documents for random order")
            import random
            random.shuffle(self.documents)
        
        logging.info("accepted total of %i articles for %s" % 
                     (len(self.documents), str(config)))

    
    def saveDictionary(self, fname):
        logging.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for tokenId, token in self.dictionary.id2token.iteritems():
            fout.write("%i\t%s\n" % (tokenId, token.token))
        fout.close()
    
    @staticmethod
    def loadDictionary(fname):
        words = [line.split('\t')[1].strip() for line in open(fname) if len(line.split('\t')) == 2]
        id2word = dict(itertools.izip(xrange(len(words)), words))
        return id2word
    
    def saveDocuments(self, fname):
        logging.info("saving documents mapping to %s" % fname)
        fout = open(fname, 'w')
        for docNo, docId in enumerate(self.documents):
            sourceId, docUri = docId
            intId, pathId = docUri
            fout.write("%i\t%s\n" % (docNo, repr(docId)))
        fout.close()

    
    def saveAsText(self, normalizeTfidf = False):
        """
        Store the corpus to disk, in a human-readable text format.
        
        This actually saves multiple files:
          1. Pure document-term co-occurence frequency counts, as a Matrix Market file. 
          2. Token to integer mapping, as a text file.
          3. Document to document URI mapping, as a text file.
        
        The exact filesystem paths and filenames are determined from the config.
        """
        self.saveDictionary(self.config.resultFile('wordids.txt'))
        self.saveDocuments(self.config.resultFile('docids.txt'))
        matutils.MmWriter.writeCorpus(self.config.resultFile('bow.mm'), self)
#endclass DmlCorpus

