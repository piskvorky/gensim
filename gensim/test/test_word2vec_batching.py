#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test batching sentences in word2vec.
"""

from time import time

from nltk.corpus import brown
from nltk import word_tokenize

from gensim.models import Word2Vec

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SentenceGenerator(object):
    def __init__(self, num_sents):
        self.brown_sentences = brown.sents()
        self.num_sents = num_sents

    def __iter__(self):
        i = 0
        for sentence in self.brown_sentences:
            if i > self.num_sents:  # NOTE: slice instead.
                break
            i += 1
            tokens = word_tokenize(' '.join(sentence))
            words = [w.lower() for w in tokens if w.isalnum()]
            yield words

if __name__ == '__main__':

    num_sents = 10000
    sentences = SentenceGenerator(num_sents=num_sents)

    test_words = ['chance', 'strings', 'spiral']

    logging.info('Training model with batching.')
    start = time()
    model2 = Word2Vec(sentences, batch=True, seed=0)
    logging.info('------------------------------------------------------')
    logging.info('Done training model. Time elapsed: %f seconds.', time() - start)

    logging.info('Training model without batching.')
    start = time()
    model1 = Word2Vec(sentences, seed=0)
    logging.info('------------------------------------------------------')
    logging.info('Done training model. Time elapsed: %f seconds.', time() - start)

    diff = {}
    for test_word in test_words:
        diff[test_word] = model1[test_word] - model2[test_word]

    import pdb
    pdb.set_trace()
