#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements the concept of Dictionary -- a mapping between words and 
their integer ids.

Dictionaries can be created from a corpus and can later be pruned according to
document frequency (removing (un)common words via the :func:`Dictionary.filterExtremes` method), 
save/loaded from disk via :func:`Dictionary.save` and :func:`Dictionary.load` methods etc.
"""



import logging
import itertools
import random

from gensim import utils


logger = logging.getLogger('dictionary')
logger.setLevel(logging.INFO)


class Dictionary(utils.SaveLoad):
    """
    Dictionary encapsulates mappings between normalized words and their integer ids.
    
    The main function is `doc2bow`, which converts a collection of words to its 
    bag-of-words representation, optionally also updating the dictionary mapping 
    with newly encountered words and their ids.
    """
    def __init__(self, documents = None):
        self.token2id = {} # token -> tokenId
        self.docFreq = {} # tokenId -> in how many documents this token appeared
        self.numDocs = 0 # number of documents processed
        self.numPos = 0 # total number of corpus positions
        
        if documents:
            self.addDocuments(documents)
    
    
    # TODO expensive, only here for historical reasons; maybe deprecate?
    id2token = property(lambda self: dict((id, token) for token, id in self.token2id.iteritems())) 
    id2word = id2token
    

    def __len__(self):
        """
        Return the number of token->id mappings in the dictionary.
        """
        return len(self.token2id)


    def __str__(self):
        return ("Dictionary(%i unique tokens)" % len(self))


    @staticmethod
    def fromDocuments(documents):
        return Dictionary(documents = documents)


    def addDocuments(self, documents):
        """
        Build dictionary from a collection of documents. Each document is a list 
        of tokens (**tokenized and normalized** utf-8 encoded strings).
        
        This is only a convenience wrapper for calling `doc2bow` on each document
        with `allowUpdate=True`.
        
        >>> print Dictionary.fromDocuments(["máma mele maso".split(), "ema má máma".split()])
        Dictionary(5 unique tokens)
        """
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                logging.info("adding document #%i to %s" % (docno, self))
            _ = self.doc2bow(document, allowUpdate = True) # ignore the result, here we only care about updating token ids
        logger.info("built %s from %i documents (total %i corpus positions)" % 
                     (self, self.numDocs, self.numPos))
        
    
    def doc2bow(self, document, allowUpdate = False):
        """
        Convert `document` (a list of words) into the bag-of-words format = list of 
        `(tokenId, tokenCount)` 2-tuples. Each word is assumed to be a 
        **tokenized and normalized** utf-8 encoded string.
        
        If `allowUpdate` is set, then also update of dictionary in the process: create ids 
        for new words. At the same time, update document frequencies -- for 
        each word appearing in this document, increase its `self.docFreq` by one.
        
        If `allowUpdate` is **not** set, this function is `const`, i.e. read-only.
        """
        result = {}
        document = sorted(document)
        # construct (word, frequency) mapping. in python3 this is done simply 
        # using Counter(), but here i use itertools.groupby() for the job
        for wordNorm, group in itertools.groupby(sorted(document)):
            frequency = len(list(group)) # how many times does this word appear in the input document

            tokenId = self.token2id.get(wordNorm, None)
            if tokenId is None: 
                # first time we see this token (~normalized form)
                if not allowUpdate: # if we aren't allowed to create new tokens, continue with the next unique token
                    continue
                tokenId = len(self.token2id)
                self.token2id[wordNorm] = tokenId # new id = number of ids made so far; NOTE this assumes there are no gaps in the id sequence!
            
            # update how many times a token appeared in the document
            result[tokenId] = frequency
        
        if allowUpdate:
            self.numDocs += 1
            self.numPos += len(document)
            # increase document count for each unique token that appeared in the document
            for tokenId in result.iterkeys():
                self.docFreq[tokenId] = self.docFreq.get(tokenId, 0) + 1
        
        return sorted(result.iteritems()) # return tokenIds, in ascending id order


    def filterExtremes(self, noBelow = 5, noAbove = 0.5, keepN = None):
        """
        Filter out tokens that appear in
        
        1. less than `noBelow` documents (absolute number) or 
        2. more than `noAbove` documents (fraction of total corpus size, *not* 
           absolute number).
        3. after (1) and (2), keep only the first `keepN' most frequent tokens (or
           all if `None`).
        
        After the pruning, shrink resulting gaps in word ids. 
        
        **Note**: Due to the gap shrinking, the same word may have a different 
        word id before and after the call to this function!
        """
        noAboveAbs = int(noAbove * self.numDocs) # convert fractional threshold to absolute threshold
        
        # determine which tokens to keep
        goodIds = (v for v in self.token2id.itervalues() if noBelow <= self.docFreq[v] <= noAboveAbs)
        goodIds = sorted(goodIds, key = self.docFreq.get, reverse = True)
        if keepN is not None:
            goodIds = goodIds[:keepN]
        logger.info("keeping %i tokens which were in more than %i and less than %i (=%.1f%%) documents" %
                     (len(goodIds), noBelow, noAboveAbs, 100.0 * noAbove))
        
        # do the actual filtering, then rebuild dictionary to remove gaps in ids
        self.filterTokens(goodIds = goodIds)
        self.rebuildDictionary()
        logger.info("resulting dictionary: %s" % self)

    
    def filterTokens(self, badIds = None, goodIds = None):
        """
        Remove the selected `badIds` tokens from all dictionary mappings, or, keep
        selected `goodIds` in the mapping and remove the rest.
        
        `badIds` and `goodIds` are collections of word ids to be removed.
        """
        if badIds is not None:
            badIds = set(badIds)
            self.token2id = dict((token, tokenId) for token, tokenId in self.token2id.iteritems() if tokenId not in badIds)
            self.docFreq = dict((tokenId, freq) for tokenId, freq in self.docFreq.iteritems() if tokenId not in badIds)
        if goodIds is not None:
            goodIds = set(goodIds)
            self.token2id = dict((token, tokenId) for token, tokenId in self.token2id.iteritems() if tokenId in goodIds)
            self.docFreq = dict((tokenId, freq) for tokenId, freq in self.docFreq.iteritems() if tokenId in goodIds)

    
    def rebuildDictionary(self):
        """
        Assign new word ids to all words. 
        
        This is done to make the ids more compact, e.g. after some tokens have 
        been removed via :func:`filterTokens` and there are gaps in the id series.
        Calling this method will remove the gaps.
        """
        logger.debug("rebuilding dictionary, shrinking gaps")
        
        # build mapping from old id -> new id
        idmap = dict(itertools.izip(self.token2id.itervalues(), xrange(len(self.token2id))))
        
        # reassign mappings to new ids
        self.token2id = dict((token, idmap[tokenId]) for token, tokenId in self.token2id.iteritems())
        self.docFreq = dict((idmap[tokenId], freq) for tokenId, freq in self.docFreq.iteritems())
#endclass Dictionary

