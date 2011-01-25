#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a variation on wikicorpus that processes plain text, not xml dumps from wikipedia.
# We needed this because our parsing that replicates Gabrilovitch (2009) requires some link
# analysis that is very difficult to do (maybe cannot be done) straight from the dump.


"""USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a bz2-compressed \
dump of Wikipedia articles, in XML format.

This actually creates three files:
 * OUTPUT_PREFIX_wordids.txt: mapping between words and their integer ids
 * OUTPUT_PREFIX_bow.mm: bag-of-words (word counts) representation, in Matrix Matrix format
 * OUTPUT_PREFIX_tfidf.mm: TF-IDF representation

The output Matrix Market files can then be compressed (e.g., by bzip2) to save \
disk space; gensim's corpus iterators can work with compressed input, too.

Example: ./wikicorpus.py ~/gensim/results/enwiki-20100622-pages-articles.xml.bz2 ~/gensim/results/wiki_en
"""


import logging
import itertools
import sys
import os.path
import re
import bz2

import sys
base_path       = "/home/quesada/coding/gensim/"
import gensim

from gensim import interfaces, matutils, utils
from gensim.corpora.dictionary import Dictionary # for constructing word->id mappings



logger = logging.getLogger('wikiExternParsingCorpus')
logger.setLevel(logging.DEBUG)


ARTICLE_MIN_CHARS = 500



#def tokenize(content):
#    """
#    Tokenize a piece of text from wikipedia. The input string `content` is assumed
#    to be mark-up free (see `filterWiki()`).
#
#    Return tokens as utf8 bytestrings.
#    """
#    # todo add the preprocessing module to gensim
#    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
#    return [token.encode('utf8') for token in utils.tokenize(content, lower = True, errors = 'ignore')
#            if len(token) <= 15 and not token.startswith('_')]



class WikiExternParsingCorpus(interfaces.CorpusABC):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id, takes almost 7h
    >>> wiki.saveAsText('wiki_en_vocab200k') # another 7.5h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fname, noBelow = 20, keep_words = 200000, dictionary = None):
        """
        Initialize the corpus. This scans the corpus once, to determine its
        vocabulary (only the first `keep_words` most frequent words that
        appear in at least `noBelow` documents are kept).
        """

        # note: we get 297000 words. They cut at 2000000 Not sure what their criteria are (noBelow 20, noAbove .1)

        self.fname = fname
        if dictionary is None:
            self.dictionary = Dictionary(self.getArticles())
            # todo: make filtering optional with a parameter
            #self.dictionary.filterExtremes(noBelow = noBelow, noAbove = 0.1, keepN = keep_words)
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
            yield self.dictionary.doc2bow(text, allowUpdate = False)


    def saveDictionary(self, fname):
        """
        Store id->word mapping to a file, in format `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.
        """
        logger.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for token, tokenId in sorted(self.dictionary.token2id.iteritems()):
            fout.write("%i\t%s\t%i\n" % (tokenId, token, self.dictionary.docFreq[tokenId]))
        fout.close()

    #todo figure out why this function cannot be called from ouside even commeting out staticmethod
    #@staticmethod
    def loadDictionary(fname):
        """
        Load previously stored mapping between words and their ids.

        The result can be used as the `id2word` parameter for input to transformations.
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
            result[int(wordId)] = word # docFreq not used
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
        matutils.MmWriter.writeCorpus(fname + '_bow.mm', self, progressCnt = 10000)


    def getArticles(self):
        """
        Iterate over the dump, returning text version of each article.

        Only articles of sufficient length are returned
        """
        articles, intext = 0, False

        for lineno, line in enumerate(bz2.BZ2File(self.fname)):
            articles =+ 1
            yield line.split() # split text into tokens
        self.numDocs = articles # cache corpus length

#endclass WikiCorpus

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))
    # harcoded for testing
    basepath = '/home/quesada/coding/gensim/'
    corpuspath = '/data/corpora/wiki-mar2008/'
    corpusname = 'head500.noblanks.cor' #for testing
    #corpusname =  'stemmedAllCleaned-fq10-cd10.noblanks.cor'
    program = os.path.basename(sys.argv[0])

    # check and process input arguments
#    if len(sys.argv) < 3:
#        print globals()['__doc__'] % locals()
#        sys.exit(1)
#    input, output = sys.argv[1:3]
    input = basepath + corpuspath + corpusname + '.bz2'
    output = basepath + corpuspath + corpusname
    # build dictionary. only keep 200k most frequent words (out of total ~7m unique tokens)
    # takes about 8h on a macbook pro
    logging.info("input: " + input)
    wiki = gensim.corpora.WikiExternParsingCorpus(input,keep_words = 200000)

    # save dictionary and bag-of-words
    # another ~8h
    wiki.saveAsText(output)
    del wiki

    # initialize corpus reader and word->id mapping
    from gensim.corpora import MmCorpus
    id2token = WikiExternParsingCorpus.loadDictionary(output + '_wordids.txt')
    mm = MmCorpus(output + '_bow.mm')

    # build tfidf
    # ~20min
    from gensim.models import TfidfModel
    tfidf = TfidfModel(mm, id2word = id2token, normalize = True)

    # save tfidf vectors in matrix market format
    # ~1.5h; result file is 14GB! bzip2'ed down to 4.5GB
    MmCorpus.saveCorpus(output + '_tfidf.mm', tfidf[mm], progressCnt = 10000)

    logging.info("finished running %s" % program)

    # running lsi (chunks=20000, numTopics=400) on wiki_tfidf then takes about 14h.

