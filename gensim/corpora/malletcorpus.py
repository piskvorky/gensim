#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Corpus in `Mallet format <http://mallet.cs.umass.edu/import.php>`_."""

from __future__ import with_statement

import logging

from gensim import utils
from gensim.corpora import LowCorpus


logger = logging.getLogger(__name__)


class MalletCorpus(LowCorpus):
    """Corpus handles input in `Mallet format <http://mallet.cs.umass.edu/import.php>`_.

    **Format description**

    One file, one instance per line, assume the data is in the following format ::

        [URL] [language] [text of the page...]

    Or, more generally, ::

        [document #1 id] [label] [text of the document...]
        [document #2 id] [label] [text of the document...]
        ...
        [document #N id] [label] [text of the document...]

    Note that language/label is *not* considered in Gensim, used `__unknown__` as default value.

    Examples
    --------
    >>> from gensim.test.utils import datapath, get_tmpfile, common_texts
    >>> from gensim.corpora import MalletCorpus
    >>> from gensim.corpora import Dictionary
    >>>
    >>> # Prepare needed data
    >>> dictionary = Dictionary(common_texts)
    >>> corpus = [dictionary.doc2bow(doc) for doc in common_texts]
    >>>
    >>> # Write corpus in Mallet format to disk
    >>> output_fname = get_tmpfile("corpus.mallet")
    >>> MalletCorpus.serialize(output_fname, corpus, dictionary)
    >>>
    >>> # Read corpus
    >>> loaded_corpus = MalletCorpus(output_fname)

    """
    def __init__(self, fname, id2word=None, metadata=False):
        """

        Parameters
        ----------
        fname : str
            Path to file in Mallet format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `fname`.
        metadata : bool, optional
            If True, return additional information ("document id" and "lang" when you call
            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.line2doc`,
            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.__iter__` or
            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.docbyoffset`

       """
        self.metadata = metadata
        LowCorpus.__init__(self, fname, id2word)

    def _calculate_num_docs(self):
        """Get number of documents.

        Returns
        -------
        int
            Number of documents in file.

        """
        with utils.smart_open(self.fname) as fin:
            result = sum(1 for _ in fin)
        return result

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format (+"document_id" and "lang" if metadata=True).

        """
        with utils.smart_open(self.fname) as f:
            for line in f:
                yield self.line2doc(line)

    def line2doc(self, line):
        """Covert line into document in BoW format.

        Parameters
        ----------
        line : str
            Line from input file.

        Returns
        -------
        list of (int, int)
            Document in BoW format (+"document_id" and "lang" if metadata=True).

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.corpora import MalletCorpus
        >>>
        >>> corpus = MalletCorpus(datapath("testcorpus.mallet"))
        >>> corpus.line2doc("en computer human interface")
        [(3, 1), (4, 1)]

        """
        splited_line = [word for word in utils.to_unicode(line).strip().split(' ') if word]
        docid, doclang, words = splited_line[0], splited_line[1], splited_line[2:]

        doc = super(MalletCorpus, self).line2doc(' '.join(words))

        if self.metadata:
            return doc, (docid, doclang)
        else:
            return doc

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save a corpus in the Mallet format.

        Warnings
        --------
        This function is automatically called by :meth:`gensim.corpora.malletcorpus.MalletCorpus.serialize`,
        don't call it directly, call :meth:`gensim.corpora.lowcorpus.malletcorpus.MalletCorpus.serialize` instead.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format.
        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between word_ids (integers) and words (strings).
            If not provided, the mapping is constructed directly from `corpus`.
        metadata : bool, optional
            If True - ????

        Return
        ------
        list of int
            List of offsets in resulting file for each document (in bytes),
            can be used for :meth:`~gensim.corpora.malletcorpus.Malletcorpus.docbyoffset`.

        Notes
        -----
        The document id will be generated by enumerating the corpus.
        That is, it will range between 0 and number of documents in the corpus.

        Since Mallet has a language field in the format, this defaults to the string '__unknown__'.
        If the language needs to be saved, post-processing will be required.

        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)

        logger.info("storing corpus in Mallet format into %s", fname)

        truncated = 0
        offsets = []
        with utils.smart_open(fname, 'wb') as fout:
            for doc_id, doc in enumerate(corpus):
                if metadata:
                    doc_id, doc_lang = doc[1]
                    doc = doc[0]
                else:
                    doc_lang = '__unknown__'

                words = []
                for wordid, value in doc:
                    if abs(int(value) - value) > 1e-6:
                        truncated += 1
                    words.extend([utils.to_unicode(id2word[wordid])] * int(value))
                offsets.append(fout.tell())
                fout.write(utils.to_utf8('%s %s %s\n' % (doc_id, doc_lang, ' '.join(words))))

        if truncated:
            logger.warning(
                "Mallet format can only save vectors with integer elements; "
                "%i float entries were truncated to integer value", truncated
            )

        return offsets

    def docbyoffset(self, offset):
        """Get the document stored in file by `offset` position.

        Parameters
        ----------
        offset : int
            Offset (in bytes) to begin of document.

        Returns
        -------
        list of (int, int)
            Document in BoW format (+"document_id" and "lang" if metadata=True).

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.corpora import MalletCorpus
        >>>
        >>> data = MalletCorpus(datapath("testcorpus.mallet"))
        >>> data.docbyoffset(1)  # end of first line
        [(3, 1), (4, 1)]
        >>> data.docbyoffset(4)  # start of second line
        [(4, 1)]

        """
        with utils.smart_open(self.fname) as f:
            f.seek(offset)
            return self.line2doc(f.readline())
