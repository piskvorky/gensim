#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements the concept of Dictionary -- a mapping between words and 
their internal ids.

Dictionaries can be created from a corpus and can later be pruned according to
document frequency (removing (un)common words via the :func:`Dictionary.filterExtremes` method), 
save/loaded from disk via :func:`Dictionary.save` and :func:`Dictionary.load` methods etc.
"""



import logging
import itertools
import random

from gensim import utils



class Token(object):
    """
    Object representing a single token.
    """
    __slots__ = ['token', 'intId'] # keep in slots to save a little memory
    
    def __init__(self, token, intId):
        self.token = token # postprocessed word form (string)
        self.intId = intId # id of the token (integer)
    
    # provide getstate/setstate methods, so that Token objects can be pickled
    def __getstate__(self):
        return [(name, self.__getattribute__(name)) for name in self.__slots__]
    
    def __setstate__(self, state):
        for key, val in state:
            self.__setattr__(key, val)

    def __str__(self):
        return ("Token(id=%i, norm='%s')" % (self.intId, self.token))
#endclass Token


class Dictionary(utils.SaveLoad):
    """
    Dictionary encapsulates mappings between normalized words and their integer ids.
    
    The main function is `doc2bow`, which coverts a collection of words to its bow 
    representation, optionally also updating the dictionary mapping with new 
    words and their ids.
    """
    def __init__(self):
        self.id2token = {} # tokenId -> token
        self.token2id = {} # token -> tokenId
        self.docFreq = {} # tokenId -> in how many documents this token appeared
        self.numDocs = 0
    

    def __len__(self):
        """
        Return the number of token->id mappings in the dictionary.
        """
        assert len(self.token2id) == len(self.id2token)
        return len(self.token2id)


    def __str__(self):
        return ("Dictionary(%i unique tokens)" % len(self))

    @staticmethod
    def fromDocuments(documents):
        """
        Build dictionary from a collection of documents. Each document is a list 
        of tokens (ie. **tokenized and normalized** utf-8 encoded strings).
        
        >>> print Dictionary.fromDocuments(["máma mele maso".split(), "ema má mama".split()])
        Dictionary(6 unique tokens)
        
        """
        result = Dictionary()
        for document in documents:
            _ = result.doc2bow(document, allowUpdate = True) # ignore the result, here we only care about updating token ids
        return result


    def addToken(self, token):
        if token.intId in self.id2token:
            logging.debug("overwriting old token %s (id %i); is this intended?" %
                          (token.token, token.intId))
        self.id2token[token.intId] = token
        self.token2id[token.token] = token.intId
    
    
    def doc2bow(self, document, allowUpdate = False):
        """
        Convert `document` (a list of words) into the bag-of-words format = list of 
        `(tokenId, tokenCount)` 2-tuples. Each word is assumed to be a 
        **tokenized and normalized** utf-8 encoded string.
        
        If `allowUpdate` is set, then also update of dictionary in the process: create ids 
        for new words. At the same time, update document frequencies -- for 
        each word appearing in this document, increase its self.docFreq by one.
        
        If `allowUpdate` is **not** set, this function is `const`, ie. read-only.
        """
        result = {}
        
        # construct (word, frequency) mapping. in python3 this is done simply 
        # using Counter(), but here i use itertools.groupby() for the job
        for wordNorm, group in itertools.groupby(sorted(document)):
            frequency = len(list(group)) # how many times does this word appear in the input document

            # determine the Token object of this word, creating it if necessary
            tokenId = self.token2id.get(wordNorm, None)
            if tokenId is None: 
                # first time we see this token (normalized form)
                if not allowUpdate: # if we aren't allowed to create new tokens, continue with the next unique token
                    continue
                tokenId = len(self.token2id) # new id = number of ids made so far; NOTE this assumes there are no gaps in the id sequence!
                token = Token(wordNorm, tokenId)
                self.addToken(token)
            else:
                token = self.id2token[tokenId]
            # post condition -- now both tokenId and token object are properly set
            
            # update how many times a token appeared in the document
            result[tokenId] = result.get(tokenId, 0) + frequency
            
        if allowUpdate:
            self.numDocs += 1
            # increase document count for each unique token that appeared in the document
            for tokenId in result.iterkeys():
                self.docFreq[tokenId] = self.docFreq.get(tokenId, 0) + 1
        
        return sorted(result.iteritems()) # return tokenIds, in ascending id order


    def filterExtremes(self, noBelow = 5, noAbove = 0.5):
        """
        Filter out tokens that appear in
        
        1. less than `noBelow` documents (absolute number) or 
        2. more than `noAbove` documents (fraction of total corpus size, *not* 
           absolute number).
        
        After the pruning, shrink resulting gaps in word ids. 
        
        **Note**: The same word may have a different word id before and after
        the call to this function!
        """
        noAboveAbs = int(noAbove * self.numDocs) # convert fractional threshold to absolute threshold
        
        # determine which tokens to drop
        badIds = [tokenId for tokenId, docFreq in self.docFreq.iteritems() if docFreq < noBelow or docFreq > noAboveAbs]
        logging.info("removing %i tokens which were in less than %i or more than %i (=%.1f%%) documents" %
                     (len(badIds), noBelow, noAboveAbs, 100.0 * noAbove))
        
        # print some sanity check debug info
        if len(badIds) >= 10:
            someIds = random.sample(badIds, 10) # choose 10 random ids that will be removed
            someTokenFreqs = [(self.id2token[tokenId].token, self.docFreq[tokenId]) for tokenId in someIds]
            logging.info("document frequencies of some of the removed tokens: [%s]" % 
                         ', '.join("%s:%i" % i for i in someTokenFreqs))
        
        # do the actual filtering, then rebuild dictionary to remove gaps in ids
        self.filterTokens(badIds)
        self.rebuildDictionary()
        logging.info("resulting dictionary: %s" % self)

    
    def filterTokens(self, badIds):
        """
        Remove the selected tokens from all dictionary mappings.
        
        `badIds` is a collection of word ids to be removed.
        """
        badIds = set(badIds)
        self.id2token = dict((tokenId, token) for tokenId, token in self.id2token.iteritems() if tokenId not in badIds)
        self.token2id = dict((token, tokenId) for token, tokenId in self.token2id.iteritems() if tokenId not in badIds)
        self.word2id = dict((word, tokenId) for word, tokenId in self.word2id.iteritems() if tokenId not in badIds)
        self.docFreq = dict((tokenId, freq) for tokenId, freq in self.docFreq.iteritems() if tokenId not in badIds)

    
    def rebuildDictionary(self):
        """
        Assign new word ids to all words. 
        
        This is done to make the ids more compact, ie. after some tokens have 
        been removed via :func:`filterTokens` and there are gaps in the id series.
        Calling this method will remove the gaps.
        """
        logging.debug("rebuilding dictionary, shrinking gaps")
        
        # build mapping from old id -> new id
        idmap = dict(itertools.izip(self.id2token.iterkeys(), xrange(len(self.id2token))))
        
        # reassign all mappings to new ids
        self.id2token = dict((idmap[tokenId], token) for tokenId, token in self.id2token.iteritems())
        self.token2id = dict((token, idmap[tokenId]) for token, tokenId in self.token2id.iteritems())
        self.word2id = dict((word, idmap[tokenId]) for word, tokenId in self.word2id.iteritems())
        self.docFreq = dict((idmap[tokenId], freq) for tokenId, freq in self.docFreq.iteritems())
        
        # also change ids inside Token objects
        for tokenId, token in self.id2token.iteritems():
            token.intId = idmap[token.intId]
            assert token.intId == tokenId # make sure that the mapping matches
#endclass Dictionary

