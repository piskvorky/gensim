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
import cPickle
import os.path


import matutils


class Token(object):
    __slots__ = ['token', 'intId', 'surfaceForms']

    def __init__(self, token, intId, surfaceForms):
        self.token = token # preprocessed word form (string)
        self.intId = intId # id of the form (integer)
        self.surfaceForms = set(surfaceForms) # set of all words which map to this token after preprocessing
#endclass Token


class Dictionary(object):
    """
    Dictionary encapsulates mappings between words, their normalized forms and ids
    of those normalized forms.
    
    An example of a word is 'answering', normalized word here could be for example
    'answer' and the id 42.
    
    Main function is doc2bow, which coverts a collection of words to its bow 
    representation, optionally also updating the mappings with new words and ids.
    """
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
    
    
    def doc2bow(self, document, wordNormalizer = lambda word: word, allowUpdate = False):
        """
        Convert document (list of words) into bag-of-words format = list of 
        (tokenId, tokenCount) 2-tuples.
        
        If update is set, then also update dictionary in the process: create ids 
        for new words etc. At the same time update document frequencies -- for 
        each word appearing in this document, increase its self.docFreq by one.
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
                    if not allowUpdate: # if we aren't allowed to create new tokens, continue with the next word
                        continue
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
        for tokenId in result.iterkeys():
            self.docFreq[tokenId] = self.docFreq.get(tokenId, 0) + 1
        
        return sorted(result.iteritems()) # return tokenIds in ascending order

    
    def __len__(self):
        assert len(self.token2id) == len(self.id2token)
        return len(self.token2id)


    def __str__(self):
        return ("Dictionary(%i tokens covering %i surface forms)" %
                (len(self), len(self.word2id)))
#endclass Dictionary



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



class DmlCorpus(object):
    """
    DmlCorpus implements a collection of articles. It is initialized via a DmlConfig
    object, which holds information about where to look for the articles and how 
    to process them.
    
    Apart from being a regular corpus (bag-of-words iterable with a len() function),
    DmlCorpus has methods for building and storing its dictionary (mapping between
    words and their ids).
    """
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
    
    
    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        for docNo, (sourceId, docUri) in enumerate(self.documents):
            source = self.config.sources[sourceId]

            contents = source.getContent(docUri)
            words = source.tokenize(contents)
            yield self.dictionary.doc2bow(words, source.wordNormalizer, allowUpdate = False)
    
    
    def buildDictionary(self):
        """
        Populate dictionary mapping and statistics.
        
        This is done by sequentially retrieving the article fulltexts, splitting
        them into tokens and converting tokens to their ids (creating new ids as 
        necessary).
        """
        self.dictionary = Dictionary()
        logging.info("creating dictionary from %i articles" % len(self.documents))
        for docNo, (sourceId, docUri) in enumerate(self.documents):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: at article %i/%i" % (docNo, len(self.documents)))
            
            source = self.config.sources[sourceId]

            contents = source.getContent(docUri)
            words = source.tokenize(contents)
            self.dictionary.doc2bow(words, source.wordNormalizer, allowUpdate = True) # ignore the resulting bow vector -- here we are only interested in updating the dictionary
    
    
    def pruneDictionary(self):
#        self.dictionary = dictionary without too frequent/rare terms
#        rebuild dictionary (shrink id gaps)
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
        # so it is a HUGE overkill to do an extra corpus sweep just for that...)
        numDocs, numTerms, numNnz = matutils.MmWriter.determineNnz(self)
        
        # write bow to a file in matrix market format
        outfile = matutils.MmWriter(self.config.resultFile('bow.mm'))
        outfile.writeHeaders(numDocs, numTerms, numNnz)
        outfile.writeCorpus(self)
        outfile.close()
#endclass DmlCorpus

