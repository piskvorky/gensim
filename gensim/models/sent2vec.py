#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
from collections import defaultdict
from six import itervalues, string_types
import numpy as np
from numpy import dot
from gensim import utils, matutils

from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper
from random import randint

logger = logging.getLogger(__name__)

MAX_WORDS_IN_BATCH = 10000

class Sent2Vec(FastText):
    def __init__(
            self, sentences=None, sg=0, hs=0, size=100, alpha=0.2, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=2, loss='ns', sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000,
            trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, dropoutK=2):

        # sent2vec specific params
        #dropoutK is the number of ngrams dropped while training a sent2vec model
        self.dropoutK = dropoutK

        super(Sent2Vec, self).__init__(bucket=bucket, word_ngrams=word_ngrams, min_n=min_n, max_n=max_n,
            sentences=sentences, size=size, alpha=alpha, window=window, min_count=min_count,
            max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
            sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter, null_word=null_word,
            trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words)

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warning(
                        "Each 'sentences' item should be a list of words (usually unicode strings). "
                        "First item here is instead plain %s.",
                        type(sentence)
                    )
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                    sentence_no, sum(itervalues(vocab)) + total_words, len(vocab)
                )
            for word in sentence:
                num_discarded = 0
                line_size = len(sentence)
                discard = [False] * line_size
                while (num_discarded < self.dropoutK and line_size - num_discarded > 2):
                    token_to_discard = randint(0,line_size-1)
                    if discard[token_to_discard] == False:
                        discard[token_to_discard] = True
                        num_discarded += 1
                for i in range(line_size):
                    if discard[i]:
                        continue
                    for j in range(i + self.word_ngrams):
                        if j >= line_size or discard[j]:
                            break
                        vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_words += sum(itervalues(vocab))
        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(vocab), total_words, sentence_no + 1
        )
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        if update:
            if not len(self.wv.vocab):
                raise RuntimeError("You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            self.old_vocab_len = len(self.wv.vocab)
            self.old_hash2index_len = len(self.wv.hash2index)

        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        self.finalize_vocab(update=update)
        self.init_ngrams(update=update)

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0):
        super(Sent2Vec, self).train(sentences, total_examples=total_examples, epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)

    def __getitem__(self, word):
        return self.word_vec(word)

    def word_vec(self, word, use_norm=False):
        return FastTextKeyedVectors.word_vec(self.wv, word, use_norm=use_norm)

    def sent_vec(self, sentence):
        sent_vector = np.zeros(self.vector_size)
        for word in sentence.strip().split(' '):
            word_vector = self.__getitem__(word)
            sent_vector += np.array(word_vector)
        sent_vector *= (1.0 / len(sentence.split(' ')))
        return sent_vector

    def similarity(self, sent1, sent2):
        #cosine similarity between two sentences
        return dot(matutils.unitvec(self.sent_vec(sent1)), matutils.unitvec(self.sent_vec(sent2)))

    @classmethod
    def load_sent2vec_format(cls, *args, **kwargs):
        return Ft_Wrapper.load_fasttext_format(*args, **kwargs)

    def save(self, *args, **kwargs):
        super(Sent2Vec, self).save(*args, **kwargs)
