#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in GibbsLda++ format of List-Of-Words.
"""


import logging

from gensim import interfaces, utils



def splitOnSpace(s):
    return s.strip().split(' ')


class LowCorpus(interfaces.CorpusABC):
    """
    List_Of_Words corpus handles input in GibbsLda++ format.
    
    Quoting http://gibbslda.sourceforge.net/#3.2_Input_Data_Format::
    
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
    def __init__(self, fname, id2word = None, line2words = splitOnSpace):
        """
        Initialize the corpus from a file.
        
        `id2word` and `line2words` are optional parameters. 
        
        If provided, `id2word` is a dictionary mapping between wordIds (integers) 
        and words (strings). If not provided, the mapping is constructed from 
        the documents.
        
        `line2words` is a function which converts lines into tokens. Defaults to 
        simple splitting on spaces.
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

    
    def __len__(self):
        return self.numDocs

    
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
    
    
    @staticmethod
    def saveCorpus(fname, corpus, id2word = None):
        """
        Save a corpus in the List-of-words format.
        """
        if id2word is None:
            logging.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dictFromCorpus(corpus)
        
        logging.info("storing corpus in List-Of-Words format: %s" % fname)
        truncated = 0
        fout = open(fname, 'w')
        fout.write('%i\n' % len(corpus))
        for doc in corpus:
            words = []
            for wordId, value in doc:
                if abs(int(value) - value) > 1e-6:
                    truncated += 1
                words.extend([str(id2word[wordId])] * int(value))
            fout.write('%s\n' % ' '.join(words))
        fout.close()
        
        if truncated:
            logging.warning("List-of-words format can only save vectors with \
            integer entries; %i float entries were truncated to integer value" % 
            truncated)
#endclass LowCorpus

