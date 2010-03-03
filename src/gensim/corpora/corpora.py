#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements classes for I/O over various corpus formats in the Vector 
Space Model.

A corpus is any object which supports iteration over the bag-of-words representation \
of its constituent documents, without the need to keep all articles in memory at \
the same time.
This allows corpora to be much larger than the available RAM.

A trivial example of a corpus containing two documents would be:
>>> corpus = [[(1, 2), (3, 10)], [(0, 1)]]
The first document contains 12 words of two types (ids 1 and 3), the second document 
a single word of another type (id 0).

The word ids may actually refer to concept ids, it is up to application how to \
interpret these document vectors.

There are several corpora classes included in this module, which differ in the way \
they are initialized:

1) LowCorpus: initialized from a single file, where each line is one document = list of \
words separated by space (used by GibbsLda++ soft).

2) MmCorpus: initialized from a single file in Matrix Market coordinate format (standard \
sparse matrix format).

3) DmlCorpus: more complex, can be initialized from many different sources (local 
filesystem/network) and different data formats, includes methods for creating 
word->wordId mappings.

Also in this module is the DmlConfig class, which encapsulates parameters necessary for \
initialization of the DmlCorpus.
"""


import logging
import itertools
import os.path

from gensim import interfaces, matutils
import dictionary # for constructing word->id mappings



class CorpusLow(interfaces.CorpusABC):
    """
    List_Of_Words corpus handles input in GibbsLda++ format.
    
    Quoting http://gibbslda.sourceforge.net/#3.2_Input_Data_Format :
    Both data for training/estimating the model and new data (i.e., previously 
    unseen data) have the same format as follows:

    [M]
    [document1]
    [document2]
    ...
    [documentM]

    in which the first line is the total number for documents [M]. Each line 
    after that is one document. [documenti] is the ith document of the dataset 
    that consists of a list of Ni words/terms.

    [documenti] = [wordi1] [wordi2] ... [wordiNi]

    in which all [wordij] (i=1..M, j=1..Ni) are text strings and they are separated 
    by the blank character.
    """
    def __len__(self):
        return self.numDocs

    
    def __init__(self, fname, id2word = None, line2words = lambda line: line.strip().split(' ')):
        """
        Initialize the corpus from a file.
        
        id2word and line2words are optional parameters. 
        
        If provided, id2word is a dictionary mapping between wordIds (integers) 
        and words (strings). If not provided, the mapping is constructed from 
        the documents.
        
        line2words is a function which converts lines into tokens. Default is 
        splitting on spaces.
        """
        logging.info("loading corpus from %s" % fname)
        
        self.fname = fname # input file, see class doc for format
        self.line2words = line2words # how to translate lines into words (simply split on space by default)
        self.numDocs = int(open(fname).readline()) # the first line in input data is the number of documents (integer). throws exception on bad input.
        
        if not id2word:
            # build a list of all word types in the corpus (distinct words)
            logging.info("extracting vocabulary from the corpus")
            allTerms = set()
            self.useWordIds = False # return documents as (word, wordCount) 2-tuples
            for doc in self:
                allTerms.update(word for word, wordCnt in doc)
            allTerms = sorted(allTerms) # sort the list of all words; rank in that list = word's integer id
            self.id2word = dict(zip(xrange(len(allTerms)), allTerms)) # build a mapping of word id(int) -> word (string)
        else:
            logging.info("using provided word mapping (%i ids)" % len(id2word))
            self.id2word = id2word
        self.word2id = dict((v, k) for k, v in self.id2word.iteritems())
        self.numTerms = len(self.word2id)
        self.useWordIds = True # return documents as (wordIndex, wordCount) 2-tuples
        
        logging.info("loaded corpus with %i documents and %i terms from %s" % 
                     (self.numDocs, self.numTerms, fname))

    
    def __iter__(self):
        """
        Iterate over the corpus, returning one bag-of-words vector at a time.
        """
        for lineNo, line in enumerate(open(self.fname)):
            if lineNo > 0: # ignore the first line = number of documents
                # convert document line to words, using the function supplied in constructor
                words = self.line2words(line)
                
                if self.useWordIds:
                    # get all distinct terms in this document, ignore unknown words
                    uniqWords = set(words).intersection(self.word2id.iterkeys())
                    
                    # the following creates a unique list of words *in the same order*
                    # as they were in the input. when iterating over the documents,
                    # the (word, count) pairs will appear in the same order as they
                    # were in the input (bar duplicates), which looks better. 
                    # if this was not needed, we might as well have used useWords = set(words)
                    useWords, marker = [], set()
                    for word in words:
                        if (word in uniqWords) and (word not in marker):
                            useWords.append(word)
                            marker.add(word)
                    # construct a list of (wordIndex, wordFrequency) 2-tuples
                    doc = zip(map(self.word2id.get, useWords), map(words.count, useWords)) # using list.count is suboptimal but speed of this whole function is irrelevant
                else:
                    uniqWords = set(words)
                    # construct a list of (word, wordFrequency) 2-tuples
                    doc = zip(uniqWords, map(words.count, uniqWords)) # using list.count is suboptimal but that's irrelevant at this point
                
                # return the document, then forget it and move on to the next one
                # note that this way, only one doc is stored in memory at a time, not the whole corpus
                yield doc
    
    
    def saveAsBlei(self, fname = None):
        """
        Save the corpus in a format compatible with Blei's LDA-C.
        
        There are actually two files saved: INPUT.blei and INPUT.blei.vocab, where
        INPUT is the filename of the original LOW corpus.
        """
        if fname is None:
            fname = self.fname + '.blei'
        
        logging.info("converting corpus to Blei's format: %s" % fname)
        fout = open(fname, 'w')
        for doc in self:
            fout.write("%i %s\n" % (len(doc), ' '.join("%i:%i" % p for p in doc)))
        fout.close()
        
        # write out vocabulary, in a format compatible with Blei's topics.py script
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary to %s" % fnameVocab)
        fout = open(fnameVocab + '.vocab', 'w')
        for word, wordId in sorted(self.word2id.iteritems(), key = lambda item: item[1]):
            fout.write("%s\n" % (word))
        fout.close()
#endclass CorpusLow



class MmCorpus(matutils.MmReader, interfaces.CorpusABC):
    def __iter__(self):
        """
        Interpret a matrix in Matrix Market format as a corpus.
        
        This simply wraps the iterative reader of MM format to comply with the corpus 
        interface (only return documents as bag-of-words, drop documentId).
        """
        for docId, doc in super(MmCorpus, self).__iter__():
            yield doc # get rid of docId, return the bow vector only

    def saveAsBlei(self, fname = None):
        """
        Save the corpus in a format compatible with Blei's LDA-C.
        """
        if fname is None:
            fname = self.fname + '.blei'
        
        logging.info("converting MM corpus from %s to Blei's format in %s" % 
                     (self.fname, fname))
        fout = open(fname, 'w')
        for doc in self:
            fout.write("%i %s\n" % (len(doc), ' '.join("%i:%i" % p for p in doc)))
        fout.close()
        
        # write out vocabulary, in a format compatible with Blei's topics.py script
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary to %s" % fnameVocab)
        fout = open(fnameVocab + '.vocab', 'w')
        for wordId in xrange(self.numTerms):
            fout.write("%s\n" % str(wordId))
        fout.close()
#endclass MmCorpus



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
        Store the corpus to a file in Matrix Market format.
        
        This actually saves multiple files:
        1) pure document-term co-occurence frequency counts,
        2) token to integer mapping
        3) document to document URI mapping
        
        The exact filesystem paths and filenames are determined from the config, 
        see source.
        """
        self.saveDictionary(self.config.resultFile('wordids.txt'))
        self.saveDocuments(self.config.resultFile('docids.txt'))
        matutils.MmWriter.writeCorpus(self.config.resultFile('bow.mm'), self)
#endclass DmlCorpus

