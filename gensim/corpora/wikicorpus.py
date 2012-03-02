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
removing tokens that appear in more than 10%% of all documents). Defaults to 50,000.

If you have the `pattern` package installed, this script will use a fancy lemmatization
to get a lemma of each token (instead of plain alphabetic tokenizer).

Example: ./wikicorpus.py ~/gensim/results/enwiki-latest-pages-articles.xml.bz2 ~/gensim/results/wiki_en
"""


import logging
import itertools
import sys
import os.path
import re
import bz2

from gensim import interfaces, matutils, utils

# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.mmcorpus import MmCorpus

logger = logging.getLogger('gensim.corpora.wikicorpus')


# Wiki is first scanned for all distinct word types (~7M). The types that appear
# in more than 10% of articles are removed and from the rest, the DEFAULT_DICT_SIZE
# most frequent types are kept (default 100K).
DEFAULT_DICT_SIZE = 50000

# Ignore articles shorter than ARTICLE_MIN_CHARS characters (after preprocessing).
ARTICLE_MIN_CHARS = 500

# if 'pattern' package is installed, we can use a fancy shallow parsing to get
# token lemmas. otherwise, use simple regexp tokenization
LEMMATIZE = utils.HAS_PATTERN


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE) # comments
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE) # footnotes
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE) # links to languages
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE) # template
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE) # template
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE) # remove URL, keep description
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE) # simplify links, keep description
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE) # keep description of images
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE) # keep description of files
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE) # outside links
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE) # math content
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE) # all other tags
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE) # table formatting
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE) # table cell formatting
RE_P14 = re.compile('\[\[Category:[^][]*\]\]', re.UNICODE) # categories


def filter_wiki(raw):
    """
    Filter out wiki mark-up from `raw`, leaving only text. `raw` is either unicode
    or utf-8 encoded string.
    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.decode_htmlentities(utils.to_unicode(raw, 'utf8', errors='ignore'))
    text = utils.decode_htmlentities(text) # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text)


def remove_markup(text):
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
        text = re.sub(RE_P14, '', text) # remove categories
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
    to be mark-up free (see `filter_wiki()`).

    Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
    that 15 characters (not bytes!).
    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
            if 2 <= len(token) <= 15 and not token.startswith('_')]



class WikiCorpus(TextCorpus):
    """
    Treat a wikipedia articles dump (*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> wiki.saveAsText('wiki_en_vocab200k') # another 8h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fname, no_below=20, keep_words=DEFAULT_DICT_SIZE, dictionary=None):
        """
        Initialize the corpus. This scans the corpus once, to determine its
        vocabulary (only the first `keep_words` most frequent words that
        appear in at least `noBelow` documents are kept).
        """
        self.fname = fname
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
            self.dictionary.filter_extremes(no_below=no_below, no_above=0.1, keep_n=keep_words)
        else:
            self.dictionary = dictionary


    def get_texts(self, return_raw=False):
        """
        Iterate over the dump, returning text version of each article.

        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function::

        >>> for vec in wiki_corpus:
        >>>     print vec
        """
        articles, articles_all = 0, 0
        intext, positions = False, 0
        if LEMMATIZE:
            lemmatizer = utils.lemmatizer
            yielded = 0

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
                text = filter_wiki(''.join(lines))
                if len(text) > ARTICLE_MIN_CHARS: # article redirects are pruned here
                    articles += 1
                    if return_raw:
                        result = text
                        yield result
                    else:
                        if LEMMATIZE:
                            _ = lemmatizer.feed(text)
                            while lemmatizer.has_results():
                                _, result = lemmatizer.read() # not necessarily the same text as entered above!
                                positions += len(result)
                                yielded += 1
                                yield result
                        else:
                            result = tokenize(text) # text into tokens here
                            positions += len(result)
                            yield result

        if LEMMATIZE:
            logger.info("all %i articles read; waiting for lemmatizer to finish the %i remaining jobs" %
                        (articles, articles - yielded))
            while yielded < articles:
                _, result = lemmatizer.read()
                positions += len(result)
                yielded += 1
                yield result

        logger.info("finished iterating over Wikipedia corpus of %i documents with %i positions"
                     " (total %i articles before pruning)" %
                     (articles, positions, articles_all))
        self.length = articles # cache corpus length
#endclass WikiCorpus



class VocabTransform(interfaces.TransformationABC):
    """
    Remap feature ids to new values.

    Given a mapping between old ids and new ids (some old ids may be missing = these
    features are to be discarded), this will wrap a corpus so that iterating over
    `VocabTransform[corpus]` returns the same vectors but with the new ids.

    Old features that have no counterpart in the new ids are discarded. This
    can be used to filter vocabulary of a corpus "online"::

    >>> old2new = dict((oldid, newid) for newid, oldid in enumerate(ids_you_want_to_keep))
    >>> vt = VocabTransform(old2new)
    >>> for vec_with_new_ids in vt[corpus_with_old_ids]:
    >>>     ...

    """
    def __init__(self, old2new, id2token=None):
        # id2word = dict((newid, oldid2word[oldid]) for oldid, newid in old2new.iteritems())
        self.old2new = old2new
        self.id2token = id2token


    def __getitem__(self, bow):
        """
        Return representation with the ids transformed.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        return [(self.old2new[oldid], weight) for oldid, weight in bow if oldid in self.old2new]
#endclass VocabTransform



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

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

    # build dictionary. only keep 100k most frequent words (out of total ~8.2m unique tokens)
    # takes about 9h on a macbook pro, for 3.5m articles (june 2011 wiki dump)
    wiki = WikiCorpus(input, keep_words=keep_words)
    # save dictionary and bag-of-words (term-document frequency matrix)
    # another ~9h
    wiki.dictionary.save_as_text(output + '_wordids.txt')
    MmCorpus.serialize(output + '_bow.mm', wiki, progress_cnt=10000)
    del wiki

    # initialize corpus reader and word->id mapping
    id2token = Dictionary.load_from_text(output + '_wordids.txt')
    mm = MmCorpus(output + '_bow.mm')

    # build tfidf,
    # ~30min
    from gensim.models import TfidfModel
    tfidf = TfidfModel(mm, id2word=id2token, normalize=True)

    # save tfidf vectors in matrix market format
    # ~2h; result file is 15GB! bzip2'ed down to 4.5GB
    MmCorpus.serialize(output + '_tfidf.mm', tfidf[mm], progress_cnt=10000)

    logger.info("finished running %s" % program)
