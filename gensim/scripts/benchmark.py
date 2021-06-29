#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>

"""
Help script (template) for benchmarking. Run with:

  /usr/bin/time --format "%E elapsed\n%Mk peak RAM" python -m gensim.scripts.benchmark ~/gensim-data/text9/text9.txt

"""

import logging
import sys

from gensim.models.word2vec import Text8Corpus, LineSentence  # noqa: F401
from gensim.models import FastText, Word2Vec, Doc2Vec, Phrases  # noqa: F401
from gensim import __version__

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(processName)s/%(process)d] [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
        level=logging.INFO,
    )

    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    corpus = Text8Corpus(sys.argv[1])  # text8/text9 format from http://mattmahoney.net/dc/textdata.html
    cls = FastText
    cls(corpus, workers=12, epochs=1).save(f'/tmp/{cls.__name__}.gensim{__version__}')
