#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate some word embeddings using word2vec.
"""

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from random import choice
from time import time

import pdb

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    model = Word2Vec.load_word2vec_format('text9.model.bin', binary=True)
    stop_words = stopwords.words('english')
    vocab = model.vocab.keys()

    for n_words in [10, 100, 1000, 10000]:
        document1 = [choice(vocab) for i in range(n_words)]
        document2 = [choice(vocab) for i in range(n_words)]

        start = time()
        distance = model.wmdistance(document1, document2)
        logging.info('Elapsed time: %f minutes', (time() - start)/60.)
        logging.info('distance = %f', distance)



    pdb.set_trace()

