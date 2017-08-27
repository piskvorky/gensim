#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from types import GeneratorType
from copy import deepcopy
import numpy as np
from numpy import dot, zeros, ones, vstack, outer, random, sum as np_sum, empty, float32 as REAL
from scipy.special import expit

from gensim.utils import call_on_class_only
from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper, compute_ngrams, ft_hash

logger = logging.getLogger(__name__)

MAX_WORDS_IN_BATCH = 10000


def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

            word2_subwords = []
            subwords_indices = []

            for index in word2_indices:
                subwords_indices += [index]
                word2_subwords += model.wv.ngrams_word[model.wv.index2word[index]]

            for subword in word2_subwords:
                subwords_indices.append(model.wv.ngrams[subword])

            l1 = np_sum(model.wv.syn0_all[subwords_indices], axis=0)  # 1 x vector_size
            if subwords_indices and model.cbow_mean:
                l1 /= len(subwords_indices)

            train_cbow_pair(model, word, subwords_indices, l1, alpha, is_ft=True)  # train on the sliding window for target word
        result += len(word_vocabs)
    return result


def train_batch_sg(model, sentences, alpha, work=None):
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - model.window + reduced_window)

            subwords_indices = [word.index]
            word2_subwords = model.wv.ngrams_word[model.wv.index2word[word.index]]

            for subword in word2_subwords:
                subwords_indices.append(model.wv.ngrams[subword])

            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                if pos2 != pos:  # don't train on the `word` itself
                    train_sg_pair(model, model.wv.index2word[word2.index], subwords_indices, alpha, is_ft=True)

        result += len(word_vocabs)
    return result


class FastText(Word2Vec):
    def __init__(
            self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=1, loss='ns', sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000,
            trim_rule=None, batch_words=MAX_WORDS_IN_BATCH):

        # fastText specific functions and params
        self.initialize_ngram_vectors()
        self.bucket = bucket
        self.word_ngrams = word_ngrams
        self.min_n = min_n
        self.max_n = max_n
        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0
        self.wv.min_n = min_n
        self.wv.max_n = max_n
        self.wv.ngrams_word = {}

        super(FastText, self).__init__(sentences=sentences, size=size, alpha=alpha, window=window, min_count=min_count,
            max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
            sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter, null_word=null_word,
            trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words, init_wv=False)

    def initialize_ngram_vectors(self):
        self.wv = FastTextKeyedVectors()

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        if update:
            if not len(self.wv.vocab):
                raise RuntimeError("You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            self.old_vocab_len = len(self.wv.vocab)
            self.old_hash2index_len = len(self.wv.hash2index)

        super(FastText, self).build_vocab(sentences, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, progress_per=progress_per, update=update)
        self.init_ngrams(update=update)

    def init_ngrams(self, update=False):
        if not update:
            self.wv.ngrams = {}
            self.wv.syn0_all = empty((len(self.wv.vocab) + self.bucket, self.vector_size), dtype=REAL)
            self.syn0_all_lockf = ones((len(self.wv.vocab) + self.bucket, self.vector_size), dtype=REAL)

            all_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                all_ngrams += self.wv.ngrams_word[w]

            all_ngrams = list(set(all_ngrams))
            self.num_ngram_vectors = len(self.wv.vocab) + len(all_ngrams)
            logger.info("Total number of ngrams is %d", len(all_ngrams))

            self.wv.hash2index = {}
            ngram_indices = range(len(self.wv.vocab))  # keeping the first `len(self.wv.vocab)` rows intact
            for i, ngram in enumerate(all_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash in self.wv.hash2index:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                else:
                    ngram_indices.append(len(self.wv.vocab) + ngram_hash % self.bucket)
                    self.wv.hash2index[ngram_hash] = i + len(self.wv.vocab)
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]

            self.wv.syn0_all = self.wv.syn0_all.take(ngram_indices, axis=0)
            self.reset_ngram_weights()
        else:
            new_vocab_len = len(self.wv.vocab)
            for ngram, idx in self.wv.hash2index.items():
                self.wv.hash2index[ngram] = idx + new_vocab_len - self.old_vocab_len

            new_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                new_ngrams += [ng for ng in self.wv.ngrams_word[w] if ng not in self.wv.ngrams]

            new_ngrams = list(set(new_ngrams))
            logger.info("Number of new ngrams is %d", len(new_ngrams))
            new_count = 0
            for i, ngram in enumerate(new_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash not in self.wv.hash2index:
                    self.wv.hash2index[ngram_hash] = new_count + len(self.wv.vocab) + self.old_hash2index_len
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                else:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]

            old_vocab_rows = self.wv.syn0_all[0:self.old_vocab_len, ]
            old_ngram_rows = self.wv.syn0_all[self.old_vocab_len:, ]

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, (len(self.wv.vocab) - self.old_vocab_len, self.vector_size))
            new_ngram_rows = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, (len(self.wv.hash2index) - self.old_hash2index_len, self.vector_size))

            self.wv.syn0_all = vstack([old_vocab_rows, new_vocab_rows, old_ngram_rows, new_ngram_rows])

    def reset_ngram_weights(self):
        rand_obj = np.random
        rand_obj.seed(self.seed)
        for index in range(len(self.wv.vocab) + len(self.wv.ngrams)):
            self.wv.syn0_all[index] = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size)

    def _do_train_job(self, sentences, alpha, inits):
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0):
        self.neg_labels = []
        if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        Word2Vec.train(self, sentences, total_examples=self.corpus_count, epochs=self.iter,
            start_alpha=self.alpha, end_alpha=self.min_alpha)
        self.get_vocab_word_vecs()

    def __getitem__(self, word):
        return self.word_vec(word)

    def get_vocab_word_vecs(self):
        for w, v in self.wv.vocab.items():
            word_vec = self.wv.syn0_all[v.index]
            ngrams = self.wv.ngrams_word[w]
            ngram_weights = self.wv.syn0_all
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            word_vec /= (len(ngrams) + 1)
            self.wv.syn0[v.index] = word_vec

    def word_vec(self, word, use_norm=False):
        if word in self.wv.vocab:
            if use_norm:
                return self.wv.syn0norm[self.wv.vocab[word].index]
            else:
                return self.wv.syn0[self.wv.vocab[word].index]
        else:
            logger.info("Word is out of vocabulary")
            word_vec = np.zeros(self.wv.syn0_all.shape[1])
            ngrams = Ft_Wrapper.compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.wv.ngrams]
            if use_norm:
                ngram_weights = self.wv.syn0_all_norm
            else:
                ngram_weights = self.wv.syn0_all
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    @classmethod
    def load_fasttext_format(cls, *args, **kwargs):
        return Ft_Wrapper.load_fasttext_format(*args, **kwargs)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(FastText, self).save(*args, **kwargs)
