#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Homer Strong, Radim Rehurek
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Implements the `"hashing trick" <http://en.wikipedia.org/wiki/Hashing-Trick>`_ -- a mapping between words
and their integer ids using a fixed, static mapping (hash function).

Notes
-----

The static mapping has a constant memory footprint, regardless of the number of word-types (features) in your corpus,
so it's suitable for processing extremely large corpora. The ids are computed as `hash(word) %% id_range`,
where `hash` is a user-configurable function (`zlib.adler32` by default).

Advantages:

* New words can be represented immediately, without an extra pass through the corpus
  to collect all the ids first.
* Can be used with non-repeatable (once-only) streams of documents.
* Able to represent any token (not only those present in training documents)

Disadvantages:

* Multiple words may map to the same id, causing hash collisions. The word <-> id mapping is no longer a bijection.

"""

from __future__ import with_statement

import logging
import itertools
import zlib

from gensim import utils
from six import iteritems, iterkeys


logger = logging.getLogger(__name__)


class HashDictionary(utils.SaveLoad, dict):
    """Mapping between words and their integer ids, using a hashing function.

    Unlike :class:`~gensim.corpora.dictionary.Dictionary`,
    building a :class:`~gensim.corpora.hashdictionary.HashDictionary` before using it **isn't a necessary step**.

    You can start converting words to ids immediately, without training on a corpus.

    Examples
    --------
    >>> from gensim.corpora import HashDictionary
    >>>
    >>> dct = HashDictionary(debug=False)  # needs no training corpus!
    >>>
    >>> texts = [['human', 'interface', 'computer']]
    >>> dct.doc2bow(texts[0])
    [(10608, 1), (12466, 1), (31002, 1)]

    """
    def __init__(self, documents=None, id_range=32000, myhash=zlib.adler32, debug=True):
        """

        Parameters
        ----------
        documents : iterable of iterable of str, optional
            Iterable of documents. If given, used to collect additional corpus statistics.
            :class:`~gensim.corpora.hashdictionary.HashDictionary` can work
            without these statistics (optional parameter).
        id_range : int, optional
            Number of hash-values in table, used as `id = myhash(key) %% id_range`.
        myhash : function, optional
            Hash function, should support interface `myhash(str) -> int`, uses `zlib.adler32` by default.
        debug : bool, optional
            Store which tokens have mapped to a given id? **Will use a lot of RAM**.
            If you find yourself running out of memory (or not sure that you really need raw tokens),
            keep `debug=False`.

        """
        self.myhash = myhash  # hash fnc: string->integer
        self.id_range = id_range  # hash range: id = myhash(key) % id_range
        self.debug = debug

        # the following (potentially massive!) dictionaries are only formed if `debug` is True
        self.token2id = {}
        self.id2token = {}  # reverse mapping int->set(words)
        self.dfs = {}  # token_id -> how many documents this token_id appeared in
        self.dfs_debug = {}  # token_string->how many documents this word appeared in

        self.num_docs = 0  # number of documents processed
        self.num_pos = 0  # total number of corpus positions
        self.num_nnz = 0  # total number of non-zeroes in the BOW matrix
        self.allow_update = True

        if documents is not None:
            self.add_documents(documents)

    def __getitem__(self, tokenid):
        """Get all words that have mapped to the given id so far, as a set.

        Warnings
        --------
        Works only if you initialized your :class:`~gensim.corpora.hashdictionary.HashDictionary` object
        with `debug=True`.

        Parameters
        ----------
        tokenid : int
            Token identifier (result of hashing).

        Return
        ------
        set of str
            Set of all words that have mapped to this id.

        """
        return self.id2token.get(tokenid, set())

    def restricted_hash(self, token):
        """Calculate id of the given token.
        Also keep track of what words were mapped to what ids, if `debug=True` was set in the constructor.

        Parameters
        ----------
        token : str
            Input token.

        Return
        ------
        int
            Hash value of `token`.

        """
        h = self.myhash(utils.to_utf8(token)) % self.id_range
        if self.debug:
            self.token2id[token] = h
            self.id2token.setdefault(h, set()).add(token)
        return h

    def __len__(self):
        """Get the number of distinct ids = the entire dictionary size."""
        return self.id_range

    def keys(self):
        """Get a list of all token ids."""
        return range(len(self))

    def __str__(self):
        return "HashDictionary(%i id range)" % len(self)

    @staticmethod
    def from_documents(*args, **kwargs):
        return HashDictionary(*args, **kwargs)

    def add_documents(self, documents):
        """Collect corpus statistics from a corpus.

        Warnings
        --------
        Useful only if `debug=True`, to build the reverse `id=>set(words)` mapping.

        Notes
        -----
        This is only a convenience wrapper for calling `doc2bow` on each document with `allow_update=True`.

        Parameters
        ----------
        documents : iterable of list of str
            Collection of documents.

        Examples
        --------

        >>> from gensim.corpora import HashDictionary
        >>>
        >>> dct = HashDictionary(debug=True)  # needs no training corpus!
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> "sparta" in dct.token2id
        False
        >>> dct.add_documents([["this", "is", "sparta"], ["just", "joking"]])
        >>> "sparta" in dct.token2id
        True

        """
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                logger.info("adding document #%i to %s", docno, self)
            self.doc2bow(document, allow_update=True)  # ignore the result, here we only care about updating token ids
        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            self, self.num_docs, self.num_pos
        )

    def doc2bow(self, document, allow_update=False, return_missing=False):
        """Convert a sequence of words `document` into the bag-of-words format of `[(word_id, word_count)]`
        (e.g. `[(1, 4), (150, 1), (2005, 2)]`).

        Notes
        -----
        Each word is assumed to be a **tokenized and normalized** string. No further preprocessing
        is done on the words in `document`: you have to apply tokenization, stemming etc before calling this method.

        If `allow_update` or `self.allow_update` is set, then also update the dictionary in the process: update overall
        corpus statistics and document frequencies. For each id appearing in this document, increase its document
        frequency (`self.dfs`) by one.

        Parameters
        ----------
        document : sequence of str
            A sequence of word tokens = **tokenized and normalized** strings.
        allow_update : bool, optional
            Update corpus statistics and if `debug=True`, also the reverse id=>word mapping?
        return_missing : bool, optional
            Not used. Only here for compatibility with the Dictionary class.

        Return
        ------
        list of (int, int)
            Document in Bag-of-words (BoW) format.

        Examples
        --------
        >>> from gensim.corpora import HashDictionary
        >>>
        >>> dct = HashDictionary()
        >>> dct.doc2bow(["this", "is", "máma"])
        [(1721, 1), (5280, 1), (22493, 1)]

        """
        result = {}
        missing = {}
        document = sorted(document)  # convert the input to plain list (needed below)
        for word_norm, group in itertools.groupby(document):
            frequency = len(list(group))  # how many times does this word appear in the input document
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
        """Filter tokens in the debug dictionary by their frequency.

        Since :class:`~gensim.corpora.hashdictionary.HashDictionary` id range is fixed and doesn't depend on the number
        of tokens seen, this doesn't really "remove" anything. It only clears some
        internal corpus statistics, for easier debugging and a smaller RAM footprint.

        Warnings
        --------
        Only makes sense when `debug=True`.

        Parameters
        ----------
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents.
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
        keep_n : int, optional
            Keep only the first `keep_n` most frequent tokens.

        Notes
        -----
        For tokens that appear in:

        #. Less than `no_below` documents (absolute number) or \n
        #. More than `no_above` documents (fraction of total corpus size, **not absolute number**).
        #. After (1) and (2), keep only the first `keep_n` most frequent tokens (or keep all if `None`).

        """
        no_above_abs = int(no_above * self.num_docs)  # convert fractional threshold to absolute threshold
        ok = [item for item in iteritems(self.dfs_debug) if no_below <= item[1] <= no_above_abs]
        ok = frozenset(word for word, freq in sorted(ok, key=lambda x: -x[1])[:keep_n])

        self.dfs_debug = {word: freq for word, freq in iteritems(self.dfs_debug) if word in ok}
        self.token2id = {token: tokenid for token, tokenid in iteritems(self.token2id) if token in self.dfs_debug}
        self.id2token = {
            tokenid: {token for token in tokens if token in self.dfs_debug}
            for tokenid, tokens in iteritems(self.id2token)
        }
        self.dfs = {tokenid: freq for tokenid, freq in iteritems(self.dfs) if self.id2token.get(tokenid, set())}

        # for word->document frequency
        logger.info(
            "kept statistics for which were in no less than %i and no more than %i (=%.1f%%) documents",
            no_below, no_above_abs, 100.0 * no_above
        )

    def save_as_text(self, fname):
        """Save the debug token=>id mapping to a text file.

        Warnings
        --------
        Only makes sense when `debug=True`, for debugging.

        Parameters
        ----------
        fname : str
            Path to output file.

        Notes
        -----
        The format is:
        `id[TAB]document frequency of this id[TAB]tab-separated set of words in UTF8 that map to this id[NEWLINE]`.


        Examples
        --------
        >>> from gensim.corpora import HashDictionary
        >>> from gensim.test.utils import get_tmpfile
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> data = HashDictionary(corpus)
        >>> data.save_as_text(get_tmpfile("dictionary_in_text_format"))

        """
        logger.info("saving %s mapping to %s" % (self, fname))
        with utils.smart_open(fname, 'wb') as fout:
            for tokenid in self.keys():
                words = sorted(self[tokenid])
                if words:
                    words_df = [(word, self.dfs_debug.get(word, 0)) for word in words]
                    words_df = ["%s(%i)" % item for item in sorted(words_df, key=lambda x: -x[1])]
                    words_df = '\t'.join(words_df)
                    fout.write(utils.to_utf8("%i\t%i\t%s\n" % (tokenid, self.dfs.get(tokenid, 0), words_df)))
