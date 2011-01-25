#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a variation on wikicorpus that processes plain text, not xml dumps
# from wikipedia.  We needed this because our parsing that replicates
# Gabrilovitch (2009) requires some link analysis that is very difficult to do
# (maybe cannot be done) straight from the dump.


"""
TODO: Update this

clone of wikicorpus.py
"""

import logging
import sys
import bz2

from gensim import interfaces, matutils
from gensim.corpora.dictionary import Dictionary


logger = logging.getLogger('wikiExternParsingCorpus')
logger.setLevel(logging.DEBUG)


class WikiExternParsingCorpus(interfaces.CorpusABC):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a (read-only)
    corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    Just start (and study) the __main__ and you will get a demo.
    """

    def __init__(self, fname, noBelow=20, keep_words=200000, dictionary=None):
        """
        Initialize the corpus. This scans the corpus once, to determine its
        vocabulary (only the first `keep_words` most frequent words that
        appear in at least `noBelow` documents are kept).
        """

        self.fname = fname
        if dictionary is None:
            self.dictionary = Dictionary(self.getArticles())
            # TODO: make filtering optional with a parameter
            # self.dictionary.filterExtremes(noBelow=noBelow, noAbove=0.1,
            #        keepN=keep_words)
        else:
            self.dictionary = dictionary

    def __len__(self):
        return self.numDocs

    def __iter__(self):
        """
        The function that defines a corpus -- iterating over the corpus yields
        vectors, one for each document.
        """
        for docNo, text in enumerate(self.getArticles()):
            yield self.dictionary.doc2bow(text, allowUpdate=False)

    def saveDictionary(self, fname):
        """
        Store id->word mapping to a file, in format:
        `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.
        """

        logger.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for token, tokenId in sorted(self.dictionary.token2id.iteritems()):
            fout.write("%i\t%s\t%i\n" % (tokenId, token,
                self.dictionary.docFreq[tokenId]))
        fout.close()

    @staticmethod
    def loadDictionary(fname):
        """
        Load previously stored mapping between words and their ids.

        The result can be used as the `id2word` parameter for input to
        transformations.
        """
        result = {}
        for lineNo, line in enumerate(open(fname)):
            cols = line[:-1].split('\t')
            if len(cols) == 2:
                wordId, word = cols
            elif len(cols) == 3:
                wordId, word, docFreq = cols
            else:
                continue
            # docFreq not used
            result[int(wordId)] = word
        return result

    def saveAsText(self, fname):
        """
        Store the corpus to disk, in a human-readable text format.

        This actually saves two files:

        1. Document-term co-occurence frequency counts (bag-of-words), as
           a Matrix Market file `fname_bow.mm`.
        2. Token to integer mapping, as a text file `fname_wordids.txt`.

        """
        self.saveDictionary(fname + '_wordids.txt')
        matutils.MmWriter.writeCorpus(fname + '_bow.mm', self,
                progressCnt=10000)

    def getArticles(self):
        """
        Iterate over the dump, returning text version of each article.

        Only articles of sufficient length are returned (short articles
        & redirects etc are ignored).
        """
        articles, intext = 0, False

        for lineno, line in enumerate(bz2.BZ2File(self.fname)):
            articles += 1
            # split text into tokens
            yield line.split()
        # cache corpus length
        self.numDocs = articles


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    # the demo file is in the corpora folder
    module_path = os.path.dirname(__file__)
    corpusname = os.path.join(module_path, 'head500.noblanks.cor')

    source = corpusname + '.bz2'
    output = corpusname

    # build dictionary.
    logging.info("source: " + source)
    wiki = WikiExternParsingCorpus(source, keep_words=200000)

    # save dictionary and bag-of-words
    wiki.saveAsText(output)
    del wiki

    # initialize corpus reader and word->id mapping
    from gensim.corpora import MmCorpus
    id2token = WikiExternParsingCorpus.loadDictionary(output + '_wordids.txt')
    mm = MmCorpus(output + '_bow.mm')

    # build tfidf
    from gensim.models import TfidfModel
    tfidf = TfidfModel(mm, id2word=id2token, normalize=True)

    # save tfidf vectors in matrix market format
    MmCorpus.saveCorpus(output + '_tfidf.mm', tfidf[mm], progressCnt=10000)

    logging.info("finished running")
