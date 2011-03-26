#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

    Convert articles from a Wikipedia dump to (sparse) vectors. The input is a bz2-compressed \
dump of Wikipedia articles, in XML format.

This actually creates three files:

* `OUTPUT_PREFIX_wordids.txt`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation, in Matrix Matrix format
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation

The output Matrix Market files can then be compressed (e.g., by bzip2) to save \
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing all tokens that appear in more than 10 percent documents). Defaults to 100,000.

Example: ./wikicorpus.py ~/gensim/results/enwiki-20100622-pages-articles.xml.bz2 ~/gensim/results/wiki_en
"""


import logging
import itertools
import sys
import os.path
import re
import bz2

from gensim import interfaces, matutils, utils
from gensim.corpora.dictionary import Dictionary # for constructing word->id mappings



logger = logging.getLogger('wikicorpus')
logger.setLevel(logging.INFO)


# Wiki is first scanned for all distinct word types (~7M). The types that appear
# in more than 10% of articles are removed and from the rest, the DEFAULT_DICT_SIZE
# most frequent types are kept (default 100K).
DEFAULT_DICT_SIZE = 100000

# Ignore articles shorter than ARTICLE_MIN_CHARS characters (after preprocessing).
ARTICLE_MIN_CHARS = 500


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE)
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE)
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE)
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE)
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE)
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE)
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE)


def filterWiki(raw):
    """
    Filter out wiki mark-up from utf8 string `raw`, leaving only text.
    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.decode_htmlentities(unicode(raw, 'utf8', 'ignore'))
    text = utils.decode_htmlentities(text) # '&amp;nbsp;' --> '\xa0'
    text = re.sub(RE_P2, "", text) # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, "", text) # remove comments
        text = re.sub(RE_P1, '', text) # remove footnotes
        text = re.sub(RE_P9, "", text) # remove outside links
        text = re.sub(RE_P10, "", text) # remove math content
        text = re.sub(RE_P11, "", text) # remove all remaining tags
        # remove templates (no recursion)
        text = re.sub(RE_P3, '', text)
        text = re.sub(RE_P4, '', text)
        text = re.sub(RE_P5, '\\3', text) # remove urls, keep description
        text = re.sub(RE_P7, '\n\\3', text) # simplify images, keep description only
        text = re.sub(RE_P8, '\n\\3', text) # simplify files, keep description only
        text = re.sub(RE_P6, '\\2', text) # simplify links, keep description only
        # remove table markup
        text = text.replace('||', '\n|') # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text) # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text) # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        if old == text or iters > 2: # stop if nothing changed between two iterations or after a fixed number of iterations
            break

    # the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
    # TODO is this really desirable?
    text = text.replace('[', '').replace(']', '') # promote all remaining markup to plain text
    return text


def tokenize(content):
    """
    Tokenize a piece of text from wikipedia. The input string `content` is assumed
    to be mark-up free (see `filterWiki()`).

    Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
    that 15 characters (not bytes!).
    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
            if 2 <= len(token) <= 15 and not token.startswith('_')]



class WikiCorpus(interfaces.CorpusABC):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> wiki.saveAsText('wiki_en_vocab200k') # another 8h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fname, noBelow=20, keep_words=DEFAULT_DICT_SIZE, dictionary=None):
        """
        Initialize the corpus. This scans the corpus once, to determine its
        vocabulary (only the first `keep_words` most frequent words that
        appear in at least `noBelow` documents are kept).
        """
        self.fname = fname
        if dictionary is None:
            self.dictionary = Dictionary(self.getArticles())
            self.dictionary.filterExtremes(noBelow=noBelow, noAbove=0.1, keepN=keep_words)
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
        Store id->word mapping to a file, in format `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.
        """
        logger.info("saving dictionary mapping to %s" % fname)
        fout = open(fname, 'w')
        for token, tokenId in sorted(self.dictionary.token2id.iteritems()):
            fout.write("%i\t%s\t%i\n" % (tokenId, token, self.dictionary.dfs[tokenId]))
        fout.close()


    @staticmethod
    def loadDictionary(fname, mapping_only=True):
        """
        Load previously stored mapping between words and their ids.

        The result can be used as the `id2word` parameter for input to transformations.
        """
        if mapping_only:
            result = {}
            for lineNo, line in enumerate(open(fname)):
                cols = line[:-1].split('\t')
                if len(cols) == 2:
                    wordId, word = cols
                elif len(cols) == 3:
                    wordId, word, dfs = cols
                else:
                    raise ValueError("invalid line in dictionary file %s: %s" % (fname, line.strip()))
                result[int(wordId)] = word # dfs not used
        else:
            result = Dictionary()
            for lineNo, line in enumerate(open(fname)):
                cols = line[:-1].split('\t')
                if len(cols) == 3:
                    wordId, word, dfs = cols
                else:
                    raise ValueError("invalid line in dictionary file %s: %s" % (fname, line.strip()))
                wordId = int(wordId)
                result.token2id[word] = wordId
                result.dfs[wordId] = int(dfs)

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
        matutils.MmWriter.writeCorpus(fname + '_bow.mm', self, progressCnt=10000)


    def getArticles(self, return_raw=False):
        """
        Iterate over the dump, returning text version of each article.

        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function::

        >>> for vec in wiki_corpus:
        >>>     print doc
        """
        articles, articles_all = 0, 0
        intext, positions = False, 0
        for lineno, line in enumerate(bz2.BZ2File(self.fname)):
            if line.startswith('      <text'):
                intext = True
                line = line[line.find('>') + 1 : ]
                lines = [line]
            elif intext:
                lines.append(line)
            pos = line.find('</text>') # can be on the same line as <text>
            if pos >= 0:
                articles_all += 1
                intext = False
                if not lines:
                    continue
                lines[-1] = line[:pos]
                text = filterWiki(''.join(lines))
                if len(text) > ARTICLE_MIN_CHARS: # article redirects are pruned here
                    articles += 1
                    if return_raw:
                        result = text
                    else:
                        result = tokenize(text) # text into tokens here
                        positions += len(result)
                    yield result

        logger.info("finished iterating over Wikipedia corpus of %i documents with %i positions"
                     " (total %i articles before pruning)" %
                     (articles, positions, articles_all))
        self.numDocs = articles # cache corpus length
#endclass WikiCorpus




class VocabTransform(interfaces.TransformationABC):
    """
    Convert a corpus to another, with different feature ids.

    Given a mapping between old ids and new ids (some old ids may be missing,
    i.e. the mapping need not be a bijection), this will wrap a
    corpus so that iterating over VocabTransform[corpus] returns the same vectors 
    but with the new ids.

    Old features that have no counterpart in the new ids are discarded. This
    can be used to filter vocabulary of a corpus::

    >>> old2new = dict((oldid, newid) for newid, oldid in enumerate(remaining_ids))
    >>> id2word = dict((newid, oldid2token[oldid]) for oldid, newid in old2new.iteritems())
    >>> vt = VocabTransform(old2new, id2token)

    """
    def __init__(self, old2new, id2token=None):
        self.old2new = old2new
        self.id2token = id2token


    def __getitem__(self, bow):
        """
        Return representation with the ids transformed.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        isCorpus, bow = utils.isCorpus(bow)
        if isCorpus:
            return self._apply(bow)

        return [(self.old2new[oldid], weight) for oldid, weight in bow if oldid in self.old2new]
#endclass VocabTransform



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    input, output = sys.argv[1:3]
    if len(sys.argv) > 3:
        keep_words = int(sys.argv[3])
    else:
        keep_words = DEFAULT_DICT_SIZE

    # build dictionary. only keep 100k most frequent words (out of total ~7m unique tokens)
    # takes about 8h on a macbook pro
    wiki = WikiCorpus(input, keep_words=keep_words)

    # save dictionary and bag-of-words (term-document frequency matrix)
    # another ~8h
    wiki.saveAsText(output)
    del wiki

    # initialize corpus reader and word->id mapping
    from gensim.corpora import MmCorpus
    id2token = WikiCorpus.loadDictionary(output + '_wordids.txt')
    mm = MmCorpus(output + '_bow.mm')

    # build tfidf
    # ~20min
    from gensim.models import TfidfModel
    tfidf = TfidfModel(mm, id2word=id2token, normalize=True)

    # save tfidf vectors in matrix market format
    # ~1.5h; result file is 14GB! bzip2'ed down to 4.5GB
    MmCorpus.saveCorpus(output + '_tfidf.mm', tfidf[mm], progressCnt=10000)

    logging.info("finished running %s" % program)
