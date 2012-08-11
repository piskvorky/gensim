#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements the concept of HashDictionary -- a drop-in replacement for gensim.corpora.dictionary
their integer ids.

"""

from __future__ import with_statement

import codecs                    # for unicode output
import zlib
import logging
import itertools
import UserDict

from gensim import utils


logger = logging.getLogger('gensim.corpora.hashdictionary')


class RestrictedHash:
    """
    Mimics a dict, using a restricted hash.
    """
    def __init__(self, key_range=32000, myhash=hash, maintain_reverse=True, debug=False):
        """
        Initialize a RestrictedHash with given key range and hash function.

        maintain_reverse determines whether to keep a dict mapping the inverse hash function..
        """
        self.key_range = key_range
        self.myhash = myhash
        self.debug = debug
        self.maintain_reverse = maintain_reverse
        self.reverse = {}
        self.debug_reverse = {}

    def __len__(self):
        """
        Reports the size of the domain of possible keys.
        """
        return self.key_range

    def __iter__(self):
        """
        Iterates over the hashes which have been calculated
        """
        for v in self.reverse.values():
            yield v

    def __getitem__(self, key):
        """
        Calculate the hash on submitted key.

        If maintain_reverse, we also keep track of the inverse hash.
        """
        h = self.restricted_hash(key)
        if self.maintain_reverse and not self.reverse.get(h, None):
            self.reverse[h] = key
            if self.debug:
                if self.debug_reverse.get(h, None):
                   self.debug_reverse[h] = self.debug_reverse[h].add(key)
                else:
                    self.debug_reverse[h] = set([key])
        return h

    def itervalues(self):
        return self.reverse.keys()

    def iteritems(self):
        return dict((v,k) for k, v in self.reverse.iteritems())

    def values(self):
        return self.reverse.keys()

    def keys(self):
        return self.reverse.values()

    def subset(self, key_subset):
        self.reverse = dict((k,v) for k, v in self.reverse.iteritems() if k in key_subset)

    def restricted_hash(self, key):
        """Calculates the hash mod the range"""
        return self.myhash(key) % self.key_range



class HashDictionary(utils.SaveLoad, UserDict.DictMixin):
    """
    HashDictionary is a drop-in replacement for Dictionary; see it for more info.

    The main function is `doc2bow`, which converts a collection of words to its
    bag-of-words representation: a list of (word_id, word_frequency) 2-tuples

    """
    def __init__(self, documents=None, id_range=32000, myhash=hash, debug=False):
        self.token2id = RestrictedHash(key_range=id_range, myhash=myhash, debug=debug)
        self.id2token = self.token2id.reverse # reverse mapping for token2id; only formed on request, to save memory
        self.dfs = {} # document frequencies: tokenId -> in how many documents this token appeared
        self.num_docs = 0 # number of documents processed
        self.num_pos = 0 # total number of corpus positions
        self.num_nnz = 0 # total number of non-zeroes in the BOW matrix

        if documents is not None:
            self.add_documents(documents)


    def __getitem__(self, tokenid):
        return self.id2token[tokenid]

    def keys(self):
        """Return a list of all token ids."""
        return self.token2id.keys()


    def __len__(self):
        """
        Return the number of token->id mappings seen.
        """
        return len(self.token2id)


    def __str__(self):
        return ("HashDictionary(%i id range)" % len(self))


    @staticmethod
    def from_documents(documents):
        return HashDictionary(documents=documents)


    def add_documents(self, documents):
        """
        Build dictionary from a collection of documents. Each document is a list
        of tokens = **tokenized and normalized** utf-8 encoded strings.

        This is only a convenience wrapper for calling `doc2bow` on each document
        with `allow_update=True`.

        >>> print Dictionary(["máma mele maso".split(), "ema má máma".split()])
        Dictionary(5 unique tokens)
        """
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                logger.info("adding document #%i to %s" % (docno, self))
            _ = self.doc2bow(document, allow_update=True) # ignore the result, here we only care about updating token ids
        logger.info("built %s from %i documents (total %i corpus positions)" %
                     (self, self.num_docs, self.num_pos))


    def doc2bow(self, document, allow_update=False, return_missing=False):
        """
        Convert `document` (a list of words) into the bag-of-words format = list
        of `(token_id, token_count)` 2-tuples. Each word is assumed to be a
        **tokenized and normalized** utf-8 encoded string. No further preprocessing
        is done on the words in `document`; apply tokenization, stemming etc. before
        calling this method.

        If `allow_update` is set, then also update dictionary in the process: create
        ids for new words. At the same time, update document frequencies -- for
        each word appearing in this document, increase its document frequency (`self.dfs`)
        by one.

        If `allow_update` is **not** set, this function is `const`, aka read-only.

        """
        result = {}
        missing = {}
        document = sorted(document)
        # construct (word, frequency) mapping. in python3 this is done simply
        # using Counter(), but here i use itertools.groupby() for the job
        for word_norm, group in itertools.groupby(document):
            frequency = len(list(group)) # how many times does this word appear in the input document
            tokenid = self.token2id[word_norm]
                # first time we see this token (~normalized form)
#            if not allow_update: # if we aren't allowed to create new tokens, continue with the next unique token

            # update how many times a token appeared in the document
            result[tokenid] = frequency

        if allow_update:
            self.num_docs += 1
            self.num_pos += len(document)
            self.num_nnz += len(result)
            # increase document count for each unique token that appeared in the document
            for tokenid in result.iterkeys():
                self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1

        # return tokenids, in ascending id order
        result = sorted(result.iteritems())
        if return_missing:
            return result, missing
        else:
            return result


    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000):
        """
        Filter out tokens that appear in

        1. less than `no_below` documents (absolute number) or
        2. more than `no_above` documents (fraction of total corpus size, *not*
           absolute number).
        3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
           keep all if `None`).

        After the pruning, shrink resulting gaps in word ids.

        **Note**: Due to the gap shrinking, the same word may have a different
        word id before and after the call to this function!
        """
        no_above_abs = int(no_above * self.num_docs) # convert fractional threshold to absolute threshold

        # determine which tokens to keep
        good_ids = (v for v in self.token2id.itervalues() if no_below <= self.dfs[v] <= no_above_abs)
        good_ids = sorted(good_ids, key=self.dfs.get, reverse=True)

        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        self.token2id.subset(key_subset=good_ids)
        self.dfs = dict((tokenid, freq) for tokenid, freq in self.dfs.iteritems()
                        if tokenid in good_ids)
        logger.info("keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents" %
                     (len(good_ids), no_below, no_above_abs, 100.0 * no_above))


    def save_as_text(self, fname):
        """
        Save this Dictionary to a text file, in format:
        `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.

        Note: use `save`/`load` to store in binary format instead (pickle).
        """
        logger.info("saving hashdictionary mapping to %s" % fname)
        with codecs.open(fname, 'wb',encoding='utf-8') as fout:
            for token, tokenid in sorted(self.token2id.iteritems()):
                fout.write("%i\t%s\t%i\n" % (tokenid, token, self.dfs.get(tokenid, 0)))

    @staticmethod
    def load_from_text(fname):
        """
        Load a previously stored HashDictionary from a text file.
        Mirror function to `save_as_text`.
        """
        result = HashDictionary()
        with open(fname, 'rb') as f:
            for lineno, line in enumerate(f):
                try:
                    wordid, word, docfreq = line[:-1].split('\t')
                except Exception:
                    raise ValueError("invalid line in dictionary file %s: %s"
                                     % (fname, line.strip()))
                wordid = int(wordid)
                result.token2id[word] = wordid
                result.dfs[wordid] = int(docfreq)
        return result
#endclass HashDictionary
