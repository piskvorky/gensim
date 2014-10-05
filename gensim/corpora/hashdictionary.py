#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Homer Strong, Radim Rehurek
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements the `"hashing trick" <http://en.wikipedia.org/wiki/Hashing-Trick>`_ --
a mapping between words and their integer ids using a fixed, static mapping. The
static mapping has a constant memory footprint, regardless of the number of word-types (features)
in your corpus, so it's suitable for processing extremely large corpora.

The ids are computed as `hash(word) % id_range`, where `hash` is a user-configurable
function (adler32 by default). Using HashDictionary, new words can be represented immediately,
without an extra pass through the corpus to collect all the ids first. This is another
advantage: HashDictionary can be used with non-repeatable (once-only) streams of documents.

A disadvantage of HashDictionary is that, unlike plain :class:`Dictionary`, several words may map
to the same id, causing hash collisions. The word<->id mapping is no longer a bijection.

"""

from __future__ import with_statement

import logging
import itertools
import zlib

from gensim import utils
from six import iteritems, iterkeys


logger = logging.getLogger('gensim.corpora.hashdictionary')



class HashDictionary(utils.SaveLoad, dict):
    """
    HashDictionary encapsulates the mapping between normalized words and their
    integer ids.

    Unlike `Dictionary`, building a `HashDictionary` before using it is not a necessary
    step. The documents can be computed immediately, from an uninitialized `HashDictionary`,
    without seeing the rest of the corpus first.

    The main function is `doc2bow`, which converts a collection of words to its
    bag-of-words representation: a list of (word_id, word_frequency) 2-tuples.

    """
    def __init__(self, documents=None, id_range=32000, myhash=zlib.adler32, debug=True):
        """
        By default, keep track of debug statistics and mappings. If you find yourself
        running out of memory (or are sure you don't need the debug info), set
        `debug=False`.
        """
        self.myhash = myhash # hash fnc: string->integer
        self.id_range = id_range # hash range: id = myhash(key) % id_range
        self.debug = debug

        # the following (potentially massive!) dictionaries are only formed if `debug` is True
        self.token2id = {}
        self.id2token = {} # reverse mapping int->set(words)
        self.dfs = {} # token_id -> how many documents this token_id appeared in
        self.dfs_debug = {} # token_string->how many documents this word appeared in

        self.num_docs = 0 # number of documents processed
        self.num_pos = 0 # total number of corpus positions
        self.num_nnz = 0 # total number of non-zeroes in the BOW matrix
        self.allow_update = True

        if documents is not None:
            self.add_documents(documents)


    def __getitem__(self, tokenid):
        """
        Return all words that have mapped to the given id so far, as a set.

        Only works if `self.debug` was enabled.
        """
        return self.id2token.get(tokenid, set())


    def restricted_hash(self, token):
        """
        Calculate id of the given token. Also keep track of what words were mapped
        to what ids, for debugging reasons.
        """
        h = self.myhash(utils.to_utf8(token)) % self.id_range
        if self.debug:
            self.token2id[token] = h
            self.id2token.setdefault(h, set()).add(token)
        return h


    def __len__(self):
        """
        Return the number of distinct ids = the entire dictionary size.
        """
        return self.id_range


    def keys(self):
        """Return a list of all token ids."""
        return range(len(self))


    def __str__(self):
        return ("HashDictionary(%i id range)" % len(self))


    @staticmethod
    def from_documents(*args, **kwargs):
        return HashDictionary(*args, **kwargs)


    def add_documents(self, documents):
        """
        Build dictionary from a collection of documents. Each document is a list
        of tokens = **tokenized and normalized** utf-8 encoded strings.

        This is only a convenience wrapper for calling `doc2bow` on each document
        with `allow_update=True`.
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

        If `allow_update` or `self.allow_update` is set, then also update dictionary
        in the process: update overall corpus statistics and document frequencies.
        For each id appearing in this document, increase its document frequency
        (`self.dfs`) by one.

        """
        result = {}
        missing = {}
        document = sorted(document) # convert the input to plain list (needed below)
        for word_norm, group in itertools.groupby(document):
            frequency = len(list(group)) # how many times does this word appear in the input document
            tokenid = self.restricted_hash(word_norm)
            result[tokenid] = result.get(tokenid, 0) + frequency
            if self.debug:
                # increment document count for each unique token that appeared in the document
                self.dfs_debug[word_norm] = self.dfs_debug.get(word_norm, 0) + 1

        if allow_update or self.allow_update:
            self.num_docs += 1
            self.num_pos += len(document)
            self.num_nnz += len(result)
            if self.debug:
                # increment document count for each unique tokenid that appeared in the document
                # done here, because several words may map to the same tokenid
                for tokenid in iterkeys(result):
                    self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1

        # return tokenids, in ascending id order
        result = sorted(iteritems(result))
        if return_missing:
            return result, missing
        else:
            return result


    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000):
        """
        Remove document frequency statistics for tokens that appear in

        1. less than `no_below` documents (absolute number) or
        2. more than `no_above` documents (fraction of total corpus size, *not*
           absolute number).
        3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
           keep all if `None`).

        **Note:** since HashDictionary's id range is fixed and doesn't depend on
        the number of tokens seen, this doesn't really "remove" anything. It only
        clears some supplementary statistics, for easier debugging and a smaller RAM
        footprint.
        """
        no_above_abs = int(no_above * self.num_docs) # convert fractional threshold to absolute threshold
        ok = [item for item in iteritems(self.dfs_debug)
                   if no_below <= item[1] <= no_above_abs]
        ok = frozenset(word for word, freq in sorted(ok, key=lambda item: -item[1])[:keep_n])

        self.dfs_debug = dict((word, freq)
                              for word, freq in iteritems(self.dfs_debug)
                              if word in ok)
        self.token2id = dict((token, tokenid)
                             for token, tokenid in iteritems(self.token2id)
                             if token in self.dfs_debug)
        self.id2token = dict((tokenid, set(token for token in tokens
                                                 if token in self.dfs_debug))
                             for tokenid, tokens in iteritems(self.id2token))
        self.dfs = dict((tokenid, freq)
                        for tokenid, freq in iteritems(self.dfs)
                        if self.id2token.get(tokenid, set()))

        # for word->document frequency
        logger.info("kept statistics for which were in no less than %i and no more than %i (=%.1f%%) documents" %
            (no_below, no_above_abs, 100.0 * no_above))


    def save_as_text(self, fname):
        """
        Save this HashDictionary to a text file, for easier debugging.

        The format is:
        `id[TAB]document frequency of this id[TAB]tab-separated set of words in UTF8 that map to this id[NEWLINE]`.

        Note: use `save`/`load` to store in binary format instead (pickle).
        """
        logger.info("saving HashDictionary mapping to %s" % fname)
        with utils.smart_open(fname, 'wb') as fout:
            for tokenid in self.keys():
                words = sorted(self[tokenid])
                if words:
                    words_df = [(word, self.dfs_debug.get(word, 0)) for word in words]
                    words_df = ["%s(%i)" % item for item in sorted(words_df, key=lambda item: -item[1])]
                    fout.write(utils.to_utf8("%i\t%i\t%s\n" %
                        (tokenid, self.dfs.get(tokenid, 0), '\t'.join(words_df))))
#endclass HashDictionary
