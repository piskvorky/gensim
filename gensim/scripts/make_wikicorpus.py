#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This actually creates three files:

* `OUTPUT_PREFIX_wordids.txt`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation, in
  Matrix Matrix format
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
50,000.

If you have the `pattern` package installed, this script will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

Example: python -m gensim.scripts.make_wikicorpus ~/gensim/results/enwiki-latest-pages-articles.xml.bz2 ~/gensim/results/wiki_en
"""


import logging
import os.path
import sys

from gensim.corpora import Dictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel


# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 50000


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


# check and process input arguments
if len(sys.argv) < 3:
    print globals()['__doc__'] % locals()
    sys.exit(1)
inp, outp = sys.argv[1:3]
if len(sys.argv) > 3:
    keep_words = int(sys.argv[3])
else:
    keep_words = DEFAULT_DICT_SIZE

# build dictionary. only keep the most frequent words (out of total ~8.2m
# unique tokens) takes about 9h on a macbook pro, for 3.5m articles (june 2011)
wiki = WikiCorpus(inp, keep_words=keep_words)
# save dictionary and bag-of-words (term-document frequency matrix)
# another ~9h
wiki.dictionary.save_as_text(outp + '_wordids.txt')
MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000)
del wiki

# initialize corpus reader and word->id mapping
id2token = Dictionary.load_from_text(outp + '_wordids.txt')
mm = MmCorpus(outp + '_bow.mm')

# build tfidf,
# ~30min
tfidf = TfidfModel(mm, id2word=id2token, normalize=True)

# save tfidf vectors in matrix market format
# ~2h; result file is 15GB! bzip2'ed down to 4.5GB
MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)

logger.info("finished running %s" % program)
