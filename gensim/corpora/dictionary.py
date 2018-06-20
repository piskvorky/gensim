#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module implements the concept of a Dictionary -- a mapping between words and their integer ids."""

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


logger = logging.getLogger(__name__)


class Dictionary(utils.SaveLoad, Mapping):
    """Dictionary encapsulates the mapping between normalized words and their integer ids.

    Notable instance attributes:

    Attributes
    ----------
    token2id : dict of (str, int)
        token -> tokenId.
    id2token : dict of (int, str)
        Reverse mapping for token2id, initialized in a lazy manner to save memory (not created until needed).
    dfs : dict of (int, int)
        Document frequencies: token_id -> how many documents contain this token.
    num_docs : int
        Number of documents processed.
    num_pos : int
        Total number of corpus positions (number of processed words).
    num_nnz : int
        Total number of non-zeroes in the BOW matrix (sum of the number of unique
        words per document over the entire corpus).

    """
    def __init__(self, documents=None, prune_at=2000000):
        """

        Parameters
        ----------
        documents : iterable of iterable of str, optional
            Documents to be used to initialize the mapping and collect corpus statistics.
        prune_at : int, optional
            Dictionary will keep no more than `prune_at` words in its mapping, to limit its RAM footprint.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> texts = [['human', 'interface', 'computer']]
        >>> dct = Dictionary(texts)  # initialize a Dictionary
        >>> dct.add_documents([["cat", "say", "meow"], ["dog"]])  # add more document (extend the vocabulary)
        >>> dct.doc2bow(["dog", "computer", "non_existent_word"])
        [(0, 1), (6, 1)]

        """
        self.token2id = {}
        self.id2token = {}
        self.dfs = {}

        self.num_docs = 0
        self.num_pos = 0
        self.num_nnz = 0

        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)

    def __getitem__(self, tokenid):
        """Get the string token that corresponds to `tokenid`.

        Parameters
        ----------
        tokenid : int
            Id of token.

        Returns
        -------
        str
            Token corresponding to `tokenid`.

        Raises
        ------
        KeyError
            If this Dictionary doesn't contain such `tokenid`.

        """
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = utils.revdict(self.token2id)
        return self.id2token[tokenid]  # will throw for non-existent ids

    def __iter__(self):
        """Iterate over all tokens."""
        return iter(self.keys())

    if PY3:
        # restore Py2-style dict API
        iterkeys = __iter__

        def iteritems(self):
            return self.items()

        def itervalues(self):
            return self.values()

    def keys(self):
        """Get all stored ids.

        Returns
        -------
        list of int
            List of all token ids.

        """
        return list(self.token2id.values())

    def __len__(self):
        """Get number of stored tokens.

        Returns
        -------
        int
            Number of stored tokens.

        """
        return len(self.token2id)

    def __str__(self):
        some_keys = list(itertools.islice(iterkeys(self.token2id), 5))
        return "Dictionary(%i unique tokens: %s%s)" % (len(self), some_keys, '...' if len(self) > 5 else '')

    @staticmethod
    def from_documents(documents):
        """Create :class:`~gensim.corpora.dictionary.Dictionary` from `documents`.

        Equivalent to `Dictionary(documents=documents)`.

        Parameters
        ----------
        documents : iterable of iterable of str
            Input corpus.

        Returns
        -------
        :class:`~gensim.corpora.dictionary.Dictionary`
            Dictionary initialized from `documents`.

        """
        return Dictionary(documents=documents)

    def add_documents(self, documents, prune_at=2000000):
        """Update dictionary from a collection of `documents`.

        Parameters
        ----------
        documents : iterable of iterable of str
            Input corpus. All tokens should be already **tokenized and normalized**.
        prune_at : int, optional
            Dictionary will keep no more than `prune_at` words in its mapping, to limit its RAM footprint.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = ["máma mele maso".split(), "ema má máma".split()]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.add_documents([["this", "is", "sparta"], ["just", "joking"]])
        >>> len(dct)
        10

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
        """Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples.

        Parameters
        ----------
        document : list of str
            Input document.
        allow_update : bool, optional
            Update self, by adding new tokens from `document` and updating internal corpus statistics.
        return_missing : bool, optional
            If True - return missing tokens (tokens present in `document` but not in self) with frequencies.

        Return
        ------
        list of (int, int)
            BoW representation of `document`.
        list of (int, int), dict of (str, int)
            If `return_missing` is True, return BoW representation of `document` + dictionary with missing
            tokens and their frequencies.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>> dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
        >>> dct.doc2bow(["this", "is", "máma"])
        [(2, 1)]
        >>> dct.doc2bow(["this", "is", "máma"], return_missing=True)
        ([(2, 1)], {u'this': 1, u'is': 1})

        """
        if isinstance(document, string_types):
            raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")

        # Construct (word, frequency) mapping.
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, unicode) else unicode(w, 'utf-8')] += 1

        token2id = self.token2id
        if allow_update or return_missing:
            missing = sorted(x for x in iteritems(counter) if x[0] not in token2id)
            if allow_update:
                for w, _ in missing:
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
            return result, dict(missing)
        else:
            return result

    def doc2idx(self, document, unknown_word_index=-1):
        """Convert `document` (a list of words) into a list of indexes = list of `token_id`.
        Replace all unknown words i.e, words not in the dictionary with the index as set via `unknown_word_index`.

        Parameters
        ----------
        document : list of str
            Input document
        unknown_word_index : int, optional
            Index to use for words not in the dictionary.

        Returns
        -------
        list of int
            Token ids for tokens in `document`, in the same order.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["a", "a", "b"], ["a", "c"]]
        >>> dct = Dictionary(corpus)
        >>> dct.doc2idx(["a", "a", "c", "not_in_dictionary", "c"])
        [0, 0, 2, -1, 2]

        """
        if isinstance(document, string_types):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")

        document = [word if isinstance(word, unicode) else unicode(word, 'utf-8') for word in document]
        return [self.token2id.get(word, unknown_word_index) for word in document]

    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        """Filter out tokens in the dictionary by their frequency.

        Parameters
        ----------
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents.
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
        keep_n : int, optional
            Keep only the first `keep_n` most frequent tokens.
        keep_tokens : iterable of str
            Iterable of tokens that **must** stay in dictionary after filtering.

        Notes
        -----
        This removes all tokens in the dictionary that are:

        #. Less frequent than `no_below` documents (absolute number, e.g. `5`) or \n
        #. More frequent than `no_above` documents (fraction of the total corpus size, e.g. `0.3`).
        #. After (1) and (2), keep only the first `keep_n` most frequent tokens (or keep all if `keep_n=None`).

        After the pruning, resulting gaps in word ids are shrunk.
        Due to this gap shrinking, **the same word may have a different word id before and after the call
        to this function!**

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.filter_extremes(no_below=1, no_above=0.5, keep_n=1)
        >>> len(dct)
        1

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
        """Filter out the 'remove_n' most frequent tokens that appear in the documents.

        Parameters
        ----------
        remove_n : int
            Number of the most frequent tokens that will be removed.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.filter_n_most_frequent(2)
        >>> len(dct)
        3

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
        """Remove the selected `bad_ids` tokens from :class:`~gensim.corpora.dictionary.Dictionary`.

        Alternatively, keep selected `good_ids` in :class:`~gensim.corpora.dictionary.Dictionary` and remove the rest.

        Parameters
        ----------
        bad_ids : iterable of int, optional
            Collection of word ids to be removed.
        good_ids : collection of int, optional
            Keep selected collection of word ids and remove the rest.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> 'ema' in dct.token2id
        True
        >>> dct.filter_tokens(bad_ids=[dct.token2id['ema']])
        >>> 'ema' in dct.token2id
        False
        >>> len(dct)
        4
        >>> dct.filter_tokens(good_ids=[dct.token2id['maso']])
        >>> len(dct)
        1

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
        """Assign new word ids to all words, shrinking any gaps."""
        logger.debug("rebuilding dictionary, shrinking gaps")

        # build mapping from old id -> new id
        idmap = dict(izip(sorted(itervalues(self.token2id)), xrange(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in iteritems(self.token2id)}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in iteritems(self.dfs)}

    def save_as_text(self, fname, sort_by_word=True):
        """Save :class:`~gensim.corpora.dictionary.Dictionary` to a text file.

        Parameters
        ----------
        fname : str
            Path to output file.
        sort_by_word : bool, optional
            Sort words in lexicographical order before writing them out.

        Notes
        -----
        Format::

            num_docs
            id_1[TAB]word_1[TAB]document_frequency_1[NEWLINE]
            id_2[TAB]word_2[TAB]document_frequency_2[NEWLINE]
            ....
            id_k[TAB]word_k[TAB]document_frequency_k[NEWLINE]

        This text format is great for corpus inspection and debugging. As plaintext, it's also easily portable
        to other tools and frameworks. For better performance and to store the entire object state,
        including collected corpus statistics, use :meth:`~gensim.corpora.dictionary.Dictionary.save` and
        :meth:`~gensim.corpora.dictionary.Dictionary.load` instead.

        See Also
        --------
        :meth:`~gensim.corpora.dictionary.Dictionary.load_from_text`
            Load :class:`~gensim.corpora.dictionary.Dictionary` from text file.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>> from gensim.test.utils import get_tmpfile
        >>>
        >>> tmp_fname = get_tmpfile("dictionary")
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>>
        >>> dct = Dictionary(corpus)
        >>> dct.save_as_text(tmp_fname)
        >>>
        >>> loaded_dct = Dictionary.load_from_text(tmp_fname)
        >>> assert dct.token2id == loaded_dct.token2id

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
        """Merge another dictionary into this dictionary, mapping the same tokens to the same ids
        and new tokens to new ids.

        Notes
        -----
        The purpose is to merge two corpora created using two different dictionaries: `self` and `other`.
        `other` can be any id=>word mapping (a dict, a Dictionary object, ...).

        Return a transformation object which, when accessed as `result[doc_from_other_corpus]`, will convert documents
        from a corpus built using the `other` dictionary into a document using the new, merged dictionary.

        Parameters
        ----------
        other : {dict, :class:`~gensim.corpora.dictionary.Dictionary`}
            Other dictionary.

        Return
        ------
        :class:`gensim.models.VocabTransform`
            Transformation object.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus_1, corpus_2 = [["a", "b", "c"]], [["a", "f", "f"]]
        >>> dct_1, dct_2 = Dictionary(corpus_1), Dictionary(corpus_2)
        >>> dct_1.doc2bow(corpus_2[0])
        [(0, 1)]
        >>> transformer = dct_1.merge_with(dct_2)
        >>> dct_1.doc2bow(corpus_2[0])
        [(0, 1), (3, 2)]

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
        """Load a previously stored :class:`~gensim.corpora.dictionary.Dictionary` from a text file.

        Mirror function to :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`.

        Parameters
        ----------
        fname: str
            Path to a file produced by :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`.

        See Also
        --------
        :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`
            Save :class:`~gensim.corpora.dictionary.Dictionary` to text file.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>> from gensim.test.utils import get_tmpfile
        >>>
        >>> tmp_fname = get_tmpfile("dictionary")
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>>
        >>> dct = Dictionary(corpus)
        >>> dct.save_as_text(tmp_fname)
        >>>
        >>> loaded_dct = Dictionary.load_from_text(tmp_fname)
        >>> assert dct.token2id == loaded_dct.token2id

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
        """Create :class:`~gensim.corpora.dictionary.Dictionary` from an existing corpus.

        Parameters
        ----------
        corpus : iterable of iterable of (int, number)
            Corpus in BoW format.
        id2word : dict of (int, object)
            Mapping id -> word. If None, the mapping `id2word[word_id] = str(word_id)` will be used.

        Notes
        -----
        This can be useful if you only have a term-document BOW matrix (represented by `corpus`), but not the original
        text corpus. This method will scan the term-document count matrix for all word ids that appear in it,
        then construct :class:`~gensim.corpora.dictionary.Dictionary` which maps each `word_id -> id2word[word_id]`.
        `id2word` is an optional dictionary that maps the `word_id` to a token.
        In case `id2word` isn't specified the mapping `id2word[word_id] = str(word_id)` will be used.

        Returns
        -------
        :class:`~gensim.corpora.dictionary.Dictionary`
            Inferred dictionary from corpus.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [[(1, 1.0)], [], [(0, 5.0), (2, 1.0)], []]
        >>> dct = Dictionary.from_corpus(corpus)
        >>> len(dct)
        3

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
