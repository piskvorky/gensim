#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
from numpy import zeros, ones, vstack, sum as np_sum, empty, float32 as REAL

from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper, compute_ngrams, ft_hash
from gensim import matutils

logger = logging.getLogger(__name__)

try:
    # TODO : log FAST_VERSION
    from gensim.models.fasttext_inner import train_batch_sg
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

except ImportError:
    # why falling back - log
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
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
                vocab_subwords_indices = []
                ngrams_subwords_indices = []

                for index in word2_indices:
                    vocab_subwords_indices += [index]
                    word2_subwords += model.wv.ngrams_word[model.wv.index2word[index]]

                for subword in word2_subwords:
                    ngrams_subwords_indices.append(model.wv.ngrams[subword])

                l1_vocab = np_sum(model.wv.syn0_vocab[vocab_subwords_indices], axis=0)  # 1 x vector_size
                l1_ngrams = np_sum(model.wv.syn0_ngrams[ngrams_subwords_indices], axis=0)  # 1 x vector_size

                l1 = np_sum([l1_vocab, l1_ngrams], axis=0)
                subwords_indices = [vocab_subwords_indices] + [ngrams_subwords_indices]
                if (subwords_indices[0] or subwords_indices[1]) and model.cbow_mean:
                    l1 /= (len(subwords_indices[0]) + len(subwords_indices[1]))

                train_cbow_pair(model, word, subwords_indices, l1, alpha, is_ft=True)  # train on the sliding window for target word
            result += len(word_vocabs)
        return result

    def train_batch_sg(model, sentences, alpha, work=None, neu1=None):
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

        # fastText specific params
        self.bucket = bucket
        self.word_ngrams = word_ngrams
        self.min_n = min_n
        self.max_n = max_n
        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0

        super(FastText, self).__init__(sentences=sentences, size=size, alpha=alpha, window=window, min_count=min_count,
            max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
            sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter, null_word=null_word,
            trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words)

    def initialize_word_vectors(self):
        self.wv = FastTextKeyedVectors()
        self.wv.min_n = self.min_n
        self.wv.max_n = self.max_n

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
            self.wv.syn0_vocab = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
            self.syn0_vocab_lockf = ones((len(self.wv.vocab), self.vector_size), dtype=REAL)

            self.wv.syn0_ngrams = empty((self.bucket, self.vector_size), dtype=REAL)
            self.syn0_ngrams_lockf = ones((self.bucket, self.vector_size), dtype=REAL)

            all_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                all_ngrams += self.wv.ngrams_word[w]

            all_ngrams = list(set(all_ngrams))
            self.num_ngram_vectors = len(all_ngrams)
            logger.info("Total number of ngrams is %d", len(all_ngrams))

            self.wv.hash2index = {}
            ngram_indices = []
            new_hash_count = 0
            for i, ngram in enumerate(all_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash in self.wv.hash2index:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                else:
                    ngram_indices.append(ngram_hash % self.bucket)
                    self.wv.hash2index[ngram_hash] = new_hash_count
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1

            self.wv.syn0_ngrams = self.wv.syn0_ngrams.take(ngram_indices, axis=0)
            self.syn0_ngrams_lockf = self.syn0_ngrams_lockf.take(ngram_indices, axis=0)
            self.reset_ngram_weights()
        else:
            new_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                new_ngrams += [ng for ng in self.wv.ngrams_word[w] if ng not in self.wv.ngrams]

            new_ngrams = list(set(new_ngrams))
            logger.info("Number of new ngrams is %d", len(new_ngrams))
            new_hash_count = 0
            for i, ngram in enumerate(new_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash not in self.wv.hash2index:
                    self.wv.hash2index[ngram_hash] = new_hash_count + self.old_hash2index_len
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1
                else:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, (len(self.wv.vocab) - self.old_vocab_len, self.vector_size))
            new_vocab_lockf_rows = ones((len(self.wv.vocab) - self.old_vocab_len, self.vector_size), dtype=REAL)
            new_ngram_rows = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, (len(self.wv.hash2index) - self.old_hash2index_len, self.vector_size))
            new_ngram_lockf_rows = ones((len(self.wv.hash2index) - self.old_hash2index_len, self.vector_size), dtype=REAL)

            self.wv.syn0_vocab = vstack([self.wv.syn0_vocab, new_vocab_rows])
            self.syn0_vocab_lockf = vstack([self.syn0_vocab_lockf, new_vocab_lockf_rows])
            self.wv.syn0_ngrams = vstack([self.wv.syn0_ngrams, new_ngram_rows])
            self.syn0_ngrams_lockf = vstack([self.syn0_ngrams_lockf, new_ngram_lockf_rows])

    def reset_ngram_weights(self):
        rand_obj = np.random
        rand_obj.seed(self.seed)
        for index in range(len(self.wv.vocab)):
            self.wv.syn0_vocab[index] = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size)
        for index in range(len(self.wv.hash2index)):
            self.wv.syn0_ngrams[index] = rand_obj.uniform(-1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size)

    def _do_train_job(self, sentences, alpha, inits):
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, neu1)
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
            word_vec = self.wv.syn0_vocab[v.index]
            ngrams = self.wv.ngrams_word[w]
            ngram_weights = self.wv.syn0_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            word_vec /= (len(ngrams) + 1)
            self.wv.syn0[v.index] = word_vec

    def word_vec(self, word, use_norm=False):
        return FastTextKeyedVectors.word_vec(self.wv, word, use_norm=use_norm)

    @classmethod
    def load_fasttext_format(cls, *args, **kwargs):
        return Ft_Wrapper.load_fasttext_format(*args, **kwargs)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_vocab_norm', 'syn0_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)