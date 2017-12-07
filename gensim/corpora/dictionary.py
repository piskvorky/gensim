#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module implements the concept of Dictionary -- a mapping between words and
their integer ids.

Dictionaries can be created from a corpus and can later be pruned according to
document frequency (removing (un)common words via the :func:`Dictionary.filter_extremes` method),
save/loaded from disk (via :func:`Dictionary.save` and :func:`Dictionary.load` methods), merged
with other dictionary (:func:`Dictionary.merge_with`) etc.
"""

from __future__ import with_statement

from collections import Mapping, defaultdict
import sys
import logging
import itertools

from gensim import utils

from six import PY3, iteritems, iterkeys, itervalues, string_types
from six.moves import xrange
from six.moves import zip as izip

if sys.version_info[0] >= 3:
    unicode = str


logger = logging.getLogger('gensim.corpora.dictionary')


class Dictionary(utils.SaveLoad, Mapping):
    """
    Dictionary encapsulates the mapping between normalized words and their integer ids.

    The main function is `doc2bow`, which converts a collection of words to its
    bag-of-words representation: a list of (word_id, word_frequency) 2-tuples.
    """
    def __init__(self, documents=None, prune_at=2000000):
        """
        If `documents` are given, use them to initialize Dictionary (see `add_documents()`).
        """
        self.token2id = {}  # token -> tokenId
        self.id2token = {}  # reverse mapping for token2id; only formed on request, to save memory
        self.dfs = {}  # document frequencies: tokenId -> in how many documents this token appeared

        self.num_docs = 0  # number of documents processed
        self.num_pos = 0  # total number of corpus positions
        self.num_nnz = 0  # total number of non-zeroes in the BOW matrix

        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)

    def __getitem__(self, tokenid):
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = utils.revdict(self.token2id)
        return self.id2token[tokenid]  # will throw for non-existent ids

    def __iter__(self):
        return iter(self.keys())

    if PY3:
        # restore Py2-style dict API
        iterkeys = __iter__

        def iteritems(self):
            return self.items()

        def itervalues(self):
            return self.values()

    def keys(self):
        """Return a list of all token ids."""
        return list(self.token2id.values())

    def __len__(self):
        """
        Return the number of token->id mappings in the dictionary.
        """
        return len(self.token2id)

    def __str__(self):
        some_keys = list(itertools.islice(iterkeys(self.token2id), 5))
        return "Dictionary(%i unique tokens: %s%s)" % (len(self), some_keys, '...' if len(self) > 5 else '')

    @staticmethod
    def from_documents(documents):
        return Dictionary(documents=documents)

    def add_documents(self, documents, prune_at=2000000):
        """
        Update dictionary from a collection of documents. Each document is a list
        of tokens = **tokenized and normalized** strings (either utf8 or unicode).

        This is a convenience wrapper for calling `doc2bow` on each document
        with `allow_update=True`, which also prunes infrequent words, keeping the
        total number of unique words <= `prune_at`. This is to save memory on very
        large inputs. To disable this pruning, set `prune_at=None`.

        >>> print(Dictionary(["máma mele maso".split(), "ema má máma".split()]))
        Dictionary(5 unique tokens)
        """
        for docno, document in enumerate(documents):
            # log progress & run a regular check for pruning, once every 10k docs
            if docno % 10000 == 0:
                if prune_at is not None and len(self) > prune_at:
                    self.filter_extremes(no_below=0, no_above=1.0, keep_n=prune_at)
                logger.info("adding document #%i to %s", docno, self)

            # update Dictionary with the document
            self.doc2bow(document, allow_update=True)  # ignore the result, here we only care about updating token ids

        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            self, self.num_docs, self.num_pos
        )

    def doc2bow(self, document, allow_update=False, return_missing=False):
        """
        Convert `document` (a list of words) into the bag-of-words format = list
        of `(token_id, token_count)` 2-tuples. Each word is assumed to be a
        **tokenized and normalized** string (either unicode or utf8-encoded). No further preprocessing
        is done on the words in `document`; apply tokenization, stemming etc. before
        calling this method.

        If `allow_update` is set, then also update dictionary in the process: create
        ids for new words. At the same time, update document frequencies -- for
        each word appearing in this document, increase its document frequency (`self.dfs`)
        by one.

        If `allow_update` is **not** set, this function is `const`, aka read-only.
        """
        if isinstance(document, string_types):
            raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")

        # Construct (word, frequency) mapping.
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, unicode) else unicode(w, 'utf-8')] += 1

        token2id = self.token2id
        if allow_update or return_missing:
            missing = {w: freq for w, freq in iteritems(counter) if w not in token2id}
            if allow_update:
                for w in missing:
                    # new id = number of ids made so far;
                    # NOTE this assumes there are no gaps in the id sequence!
                    token2id[w] = len(token2id)

        result = {token2id[w]: freq for w, freq in iteritems(counter) if w in token2id}

        if allow_update:
            self.num_docs += 1
            self.num_pos += sum(itervalues(counter))
            self.num_nnz += len(result)
            # increase document count for each unique token that appeared in the document
            dfs = self.dfs
            for tokenid in iterkeys(result):
                dfs[tokenid] = dfs.get(tokenid, 0) + 1

        # return tokenids, in ascending id order
        result = sorted(iteritems(result))
        if return_missing:
            return result, missing
        else:
            return result

    def doc2idx(self, document, unknown_word_index=-1):
        """Convert `document` (a list of words) into a list of indexes = list of `token_id`.

        Each word is assumed to be a **tokenized and normalized** string (either unicode or utf8-encoded).
        No further preprocessing is done on the words in `document`; apply tokenization, stemming etc. before calling
        this method.

        Replace all unknown words i.e, words not in the dictionary with the index as set via `unknown_word_index`,
        defaults to -1.

        Notes
        -----
        This function is `const`, aka read-only

        Parameters
        ----------
        document : list of str
            Tokenized, normalized and preprocessed words
        unknown_word_index : int, optional
            Index to use for words not in the dictionary.

        Returns
        -------
        list of int
            Indexes in the dictionary for words in the `document` preserving the order of words

        Examples
        --------
        >>> dictionary_obj = Dictionary()
        >>> dictionary_obj.token2id = {'computer': 0, 'human': 1, 'response': 2, 'survey': 3}
        >>> dictionary_obj.doc2idx(document=['human', 'computer', 'interface'], unknown_word_index=-1)
        [1, 0, -1]

        """
        if isinstance(document, string_types):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")

        document = [word if isinstance(word, unicode) else unicode(word, 'utf-8') for word in document]
        return [self.token2id.get(word, unknown_word_index) for word in document]

    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        """
        Filter out tokens that appear in

        1. less than `no_below` documents (absolute number) or
        2. more than `no_above` documents (fraction of total corpus size, *not*
           absolute number).
        3. if tokens are given in keep_tokens (list of strings), they will be kept regardless of
           the `no_below` and `no_above` settings
        4. after (1), (2) and (3), keep only the first `keep_n` most frequent tokens (or
           keep all if `None`).

        After the pruning, shrink resulting gaps in word ids.

        **Note**: Due to the gap shrinking, the same word may have a different
        word id before and after the call to this function!
        """
        no_above_abs = int(no_above * self.num_docs)  # convert fractional threshold to absolute threshold

        # determine which tokens to keep
        if keep_tokens:
            keep_ids = [self.token2id[v] for v in keep_tokens if v in self.token2id]
            good_ids = (
                v for v in itervalues(self.token2id)
                if no_below <= self.dfs.get(v, 0) <= no_above_abs or v in keep_ids
            )
        else:
            good_ids = (
                v for v in itervalues(self.token2id)
                if no_below <= self.dfs.get(v, 0) <= no_above_abs
            )
        good_ids = sorted(good_ids, key=self.dfs.get, reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        bad_words = [(self[idx], self.dfs.get(idx, 0)) for idx in set(self).difference(good_ids)]
        logger.info("discarding %i tokens: %s...", len(self) - len(good_ids), bad_words[:10])
        logger.info(
            "keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents",
            len(good_ids), no_below, no_above_abs, 100.0 * no_above
        )

        # do the actual filtering, then rebuild dictionary to remove gaps in ids
        self.filter_tokens(good_ids=good_ids)
        logger.info("resulting dictionary: %s", self)

    def filter_n_most_frequent(self, remove_n):
        """
        Filter out the 'remove_n' most frequent tokens that appear in the documents.

        After the pruning, shrink resulting gaps in word ids.

        **Note**: Due to the gap shrinking, the same word may have a different
        word id before and after the call to this function!
        """
        # determine which tokens to keep
        most_frequent_ids = (v for v in itervalues(self.token2id))
        most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=True)
        most_frequent_ids = most_frequent_ids[:remove_n]
        # do the actual filtering, then rebuild dictionary to remove gaps in ids
        most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
        logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])

        self.filter_tokens(bad_ids=most_frequent_ids)
        logger.info("resulting dictionary: %s", self)

    def filter_tokens(self, bad_ids=None, good_ids=None):
        """
        Remove the selected `bad_ids` tokens from all dictionary mappings, or, keep
        selected `good_ids` in the mapping and remove the rest.

        `bad_ids` and `good_ids` are collections of word ids to be removed.
        """
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = {token: tokenid for token, tokenid in iteritems(self.token2id) if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in iteritems(self.dfs) if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = {token: tokenid for token, tokenid in iteritems(self.token2id) if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in iteritems(self.dfs) if tokenid in good_ids}
        self.compactify()

    def compactify(self):
        """
        Assign new word ids to all words.

        This is done to make the ids more compact, e.g. after some tokens have
        been removed via :func:`filter_tokens` and there are gaps in the id series.
        Calling this method will remove the gaps.
        """
        logger.debug("rebuilding dictionary, shrinking gaps")

        # build mapping from old id -> new id
        idmap = dict(izip(itervalues(self.token2id), xrange(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in iteritems(self.token2id)}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in iteritems(self.dfs)}

    def save_as_text(self, fname, sort_by_word=True):
        """
        Save this Dictionary to a text file, in format:
        `num_docs`
        `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`. Sorted by word,
        or by decreasing word frequency.

        Note: text format should be use for corpus inspection. Use `save`/`load`
        to store in binary format (pickle) for improved performance.
        """
        logger.info("saving dictionary mapping to %s", fname)
        with utils.smart_open(fname, 'wb') as fout:
            numdocs_line = "%d\n" % self.num_docs
            fout.write(utils.to_utf8(numdocs_line))
            if sort_by_word:
                for token, tokenid in sorted(iteritems(self.token2id)):
                    line = "%i\t%s\t%i\n" % (tokenid, token, self.dfs.get(tokenid, 0))
                    fout.write(utils.to_utf8(line))
            else:
                for tokenid, freq in sorted(iteritems(self.dfs), key=lambda item: -item[1]):
                    line = "%i\t%s\t%i\n" % (tokenid, self[tokenid], freq)
                    fout.write(utils.to_utf8(line))

    def merge_with(self, other):
        """
        Merge another dictionary into this dictionary, mapping same tokens to the
        same ids and new tokens to new ids. The purpose is to merge two corpora
        created using two different dictionaries, one from `self` and one from `other`.

        `other` can be any id=>word mapping (a dict, a Dictionary object, ...).

        Return a transformation object which, when accessed as `result[doc_from_other_corpus]`,
        will convert documents from a corpus built using the `other` dictionary
        into a document using the new, merged dictionary (see :class:`gensim.interfaces.TransformationABC`).

        Example:

        >>> dict1 = Dictionary(some_documents)
        >>> dict2 = Dictionary(other_documents)  # ids not compatible with dict1!
        >>> dict2_to_dict1 = dict1.merge_with(dict2)
        >>> # now we can merge corpora from the two incompatible dictionaries into one
        >>> merged_corpus = itertools.chain(some_corpus_from_dict1, dict2_to_dict1[some_corpus_from_dict2])

        """
        old2new = {}
        for other_id, other_token in iteritems(other):
            if other_token in self.token2id:
                new_id = self.token2id[other_token]
            else:
                new_id = len(self.token2id)
                self.token2id[other_token] = new_id
                self.dfs[new_id] = 0
            old2new[other_id] = new_id
            try:
                self.dfs[new_id] += other.dfs[other_id]
            except Exception:
                # `other` isn't a Dictionary (probably just a dict) => ignore dfs, keep going
                pass
        try:
            self.num_docs += other.num_docs
            self.num_nnz += other.num_nnz
            self.num_pos += other.num_pos
        except Exception:
            pass

        import gensim.models
        return gensim.models.VocabTransform(old2new)

    @staticmethod
    def load_from_text(fname):
        """
        Load a previously stored Dictionary from a text file.
        Mirror function to `save_as_text`.
        """
        result = Dictionary()
        with utils.smart_open(fname) as f:
            for lineno, line in enumerate(f):
                line = utils.to_unicode(line)
                if lineno == 0:
                    if line.strip().isdigit():
                        # Older versions of save_as_text may not write num_docs on first line.
                        result.num_docs = int(line.strip())
                        continue
                    else:
                        logging.warning("Text does not contain num_docs on the first line.")
                try:
                    wordid, word, docfreq = line[:-1].split('\t')
                except Exception:
                    raise ValueError("invalid line in dictionary file %s: %s"
                                     % (fname, line.strip()))
                wordid = int(wordid)
                if word in result.token2id:
                    raise KeyError('token %s is defined as ID %d and as ID %d' % (word, wordid, result.token2id[word]))
                result.token2id[word] = wordid
                result.dfs[wordid] = int(docfreq)
        return result

    @staticmethod
    def from_corpus(corpus, id2word=None):
        """
        Create Dictionary from an existing corpus. This can be useful if you only
        have a term-document BOW matrix (represented by `corpus`), but not the
        original text corpus.

        This will scan the term-document count matrix for all word ids that
        appear in it, then construct and return Dictionary which maps each
        `word_id -> id2word[word_id]`.

        `id2word` is an optional dictionary that maps the `word_id` to a token. In
        case `id2word` isn't specified the mapping `id2word[word_id] = str(word_id)`
        will be used.
        """

        result = Dictionary()
        max_id = -1
        for docno, document in enumerate(corpus):
            if docno % 10000 == 0:
                logger.info("adding document #%i to %s", docno, result)
            result.num_docs += 1
            result.num_nnz += len(document)
            for wordid, word_freq in document:
                max_id = max(wordid, max_id)
                result.num_pos += word_freq
                result.dfs[wordid] = result.dfs.get(wordid, 0) + 1

        if id2word is None:
            # make sure length(result) == get_max_id(corpus) + 1
            result.token2id = {unicode(i): i for i in xrange(max_id + 1)}
        else:
            # id=>word mapping given: simply copy it
            result.token2id = {utils.to_unicode(token): idx for idx, token in iteritems(id2word)}
        for idx in itervalues(result.token2id):
            # make sure all token ids have a valid `dfs` entry
            result.dfs[idx] = result.dfs.get(idx, 0)

        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            result, result.num_docs, result.num_pos
        )
        return result
