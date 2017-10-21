#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
import numpy as np
from numpy import dot
from gensim import matutils
from random import randint
import sys
import random
from gensim.utils import SaveLoad, tokenize
import time

logger = logging.getLogger(__name__)
# TODO: add logger statements instead of print statements
# TODO: add docstrings and tests


class Entry():
    def __init__(self, word=None, count=0, subwords=[]):
        self.word = word
        self.count = 0
        self.subwords = subwords


class Dictionary():
    def __init__(self, t, bucket, minn, maxn, max_vocab_size=30000000, max_line_size=1024):
        self.max_vocab_size = max_vocab_size
        self.max_line_size = max_line_size
        self.words = []
        self.word2int = [-1] * max_vocab_size
        self.pdiscard = []
        self.ntokens = 0
        self.size = 0
        self.t = t
        self.bucket = bucket
        self.maxn = maxn
        self.minn = minn

    def hash_(self, word):
        h = 2166136261
        for i in range(len(word)):
            h = h ^ ord(word[i])
            h = h * 16777619
        return h

    def find(self, word):
        h = self.hash_(word) % self.max_vocab_size
        while self.word2int[h] != -1 and self.words[self.word2int[h]].word != word:
            h = (h + 1) % self.max_vocab_size
        return h

    def add(self, word):
        h = self.find(word)
        self.ntokens += 1
        if self.word2int[h] == -1:
            e = Entry(word=word, count=1)
            self.words.append(e)
            self.word2int[h] = self.size
            self.size += 1
        else:
            self.words[self.word2int[h]].count += 1

    def read(self, sentences, min_count):
        minThreshold = 1
        for sentence in sentences:
            for word in sentence:
                self.add(word)
                if self.ntokens % 1000000 == 0:
                    print "Read " + str(self.ntokens / 1000000) + "M words"
                if self.size > 0.75 * self.max_vocab_size:
                    minThreshold += 1
                    self.threshold(minThreshold)

        self.threshold(min_count)
        self.initTableDiscard()
        self.initNgrams()
        print "Read " + str(self.ntokens / 1000000) + "M words"
        if(self.size == 0):
            print "Empty vocabulary. Try a smaller minCount value."
            sys.exit()

    def threshold(self, t):
        self.words = [entry for entry in self.words if entry.count > t]
        self.size = 0
        self.word2int = [-1] * self.max_vocab_size
        for entry in self.words:
            h = self.find(entry.word)
            self.word2int[h] = self.size
            self.size += 1

    def initTableDiscard(self):
        for i in range(self.size):
            f = self.words[i].count / self.ntokens
            self.pdiscard.append(((self.t / f)**(0.5)) + (self.t / f))

    def initNgrams(self):
        for i in range(self.size):
            self.words[i].subwords.append(i)
            word = self.words[i].word
            for j in range(len(word)):
                ngram = ""
                for k, n in zip(range(j, len(word)), range(1, self.maxn + 1)):
                    ngram += word[k]
                    k += 1
                    while k < len(word):
                        ngram += word[k]
                        k += 1
                    if n >= self.minn and ((n == 1 and (j == 0 or k == len(word))) is False):
                        h = self.hash_(ngram) % self.bucket
                        self.words[i].subwords.append(self.size + h)

    def add_ngrams_train(self, context, n, k):
        line = list(context)
        num_discarded = 0
        line_size = len(line)
        discard = [False] * line_size
        while num_discarded < k and line_size - num_discarded > 2:
            token_to_discard = randint(0, line_size - 1)
            if discard[token_to_discard] is False:
                discard[token_to_discard] = True
                num_discarded += 1
        for i in range(line_size):
            if discard[i] is True:
                continue
            h = line[i]
            for j in range(i + 1, line_size):
                if j >= i + n or discard[j] is True:
                    break
                h = h * 116049371 + line[j]
                line.append(self.size + (h % self.bucket))
        return line

    def add_ngrams(self, context, n):
        line = list(context)
        line_size = len(context)
        for i in range(line_size):
            h = line[i]
            for j in range(i + 1, line_size):
                if j >= i + n:
                    break
                h = h * 116049371 + line[j]
                line.append(self.size + (h % self.bucket))
        return line

    def get_line(self, sentence):
        hashes = []
        words = []
        ntokens = 0
        for word in sentence:
            h = self.find(word)
            wid = self.word2int[h]
            if wid < 0:
                hashes.append(self.hash_(word))
                continue
            ntokens += 1
            words.append(wid)
            hashes.append(self.hash_(word))
            if ntokens > self.max_line_size:
                break
        return ntokens, hashes, words


class Model():
    def __init__(self, dict_size, vector_size, neg, bucket):
        self.wi = np.random.uniform((-1 / vector_size), ((-1 / vector_size) + 1), (dict_size + bucket, vector_size))
        self.wo = np.zeros((dict_size, vector_size))
        self.negpos = 1
        self.loss = 0.0
        self.nexamples = 1
        self.osz = dict_size
        self.hsz = vector_size
        self.neg = neg
        self.negative_table_size = 10000000
        self.negatives = []
        self.vector_size = vector_size

    def negative_sampling(self, target, lr):
        loss = 0.0
        self.grad = np.zeros(self.vector_size)
        for i in range(self.neg + 1):
            if i == 0:
                loss += self.binary_logistic(target, True, lr)
            else:
                loss += self.binary_logistic(self.get_negative(target), False, lr)
        return loss

    def sigmoid(self, val):
        return 1.0 / (1.0 + np.exp(-val))

    def binary_logistic(self, target, label, lr):
        score = self.sigmoid(np.dot(self.wo[target], self.hidden))
        alpha = lr * (float(label) - score)
        self.grad += self.wo[target] * alpha
        self.wo[target] += self.hidden * alpha
        if label is True:
            return -np.log(score)
        else:
            return -np.log(1.0 - score)

    def init_table_negatives(self, counts):
        z = 0.0
        for i in range(len(counts)):
            z += counts[i] ** 0.5
        for i in range(len(counts)):
            c = counts[i] ** 0.5
            for j in range(int(c * self.negative_table_size / z) + 1):
                self.negatives.append(i)
        random.shuffle(self.negatives)

    def get_negative(self, target):
        while True:
            negative = self.negatives[self.negpos]
            self.negpos = (self.negpos + 1) % len(self.negatives)
            if target != negative:
                break
        return negative

    def update(self, input_, target, lr):
        assert(target >= 0)
        assert(target < self.osz)
        if len(input_) == 0:
            return
        self.hidden = np.zeros(self.vector_size)
        for i in input_:
            self.hidden += self.wi[i]
        self.hidden *= (1.0 / len(input_))
        self.loss += self.negative_sampling(target, lr)
        self.nexamples += 1
        self.grad *= (1.0 / len(input_))
        for i in input_:
            self.wi[i] += self.grad


class Sent2Vec(SaveLoad):
    def __init__(self, vector_size=100, lr=0.2, lr_update_rate=100, epochs=5,
            min_count=5, neg=10, word_ngrams=2, loss='ns', bucket=2000000, t=0.0001,
            minn=3, maxn=6, dropoutk=2):

        self.vector_size = vector_size
        self.lr = lr
        self.lr_update_rate = lr_update_rate
        self.epochs = epochs
        self.min_count = min_count
        self.neg = neg
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.bucket = bucket
        self.t = t
        self.minn = minn
        self.maxn = maxn
        self.dropoutk = dropoutk

    def train(self, sentences):
        print "Creating dictionary..."
        self.dict = Dictionary(t=self.t, bucket=self.bucket, maxn=self.maxn, minn=self.minn)
        self.dict.read(sentences=sentences, min_count=self.min_count)
        print "Dictionary created, dictionary size:", self.dict.size, ",tokens read:", self.dict.ntokens
        self.token_count = 0
        counts = [entry.count for entry in self.dict.words]
        print "Initializing model..."
        self.model = Model(vector_size=self.vector_size, dict_size=self.dict.size, neg=self.neg, bucket=self.bucket)
        self.model.init_table_negatives(counts=counts)
        ntokens = self.dict.ntokens
        local_token_count = 0
        print "Training..."
        progress = 0
        start_time = time.time()
        for i in range(self.epochs):
            print "Begin epoch", i, ":"
            for sentence in sentences:
                progress = self.token_count / (self.epochs * ntokens)
                if progress >= 1:
                    break
                lr = self.lr * (1.0 - progress)
                ntokens_temp, hashes, words = self.dict.get_line(sentence)
                local_token_count += ntokens_temp
                if len(words) > 1:
                    for i in range(len(words)):
                        if random.uniform(0, 1) > self.dict.pdiscard[words[i]]:
                            continue
                        context = list(words)
                        context[i] = 0
                        context = self.dict.add_ngrams_train(context=context, n=self.word_ngrams, k=self.dropoutk)
                        self.model.update(input_=context, target=words[i], lr=lr)
                if local_token_count > self.lr_update_rate:
                    self.token_count += local_token_count
                    local_token_count = 0
                if self.token_count >= self.epochs * ntokens:
                    break
            print "Progress: ", progress * 100, "% lr: ", lr, " loss: ", self.model.loss / self.model.nexamples
        print "\n\nTotal training time: %s seconds" % (time.time() - start_time)

    def sentence_vectors(self, sentence_string):
        sentence = tokenize(sentence_string)
        ntokens_temp, hashes, words = self.dict.get_line(sentence)
        sent_vec = np.zeros(self.vector_size)
        line = self.dict.add_ngrams(context=words, n=self.word_ngrams)
        for word_vec in line:
            sent_vec += self.model.wi[word_vec]
        sent_vec *= (1.0 / len(line))
        return sent_vec

    def similarity(self, sent1, sent2):
        # cosine similarity between two sentences
        return dot(matutils.unitvec(self.sentence_vectors(sent1)), matutils.unitvec(self.sentence_vectors(sent2)))
