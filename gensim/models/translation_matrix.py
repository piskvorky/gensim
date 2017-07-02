#!/usr/bin/env python
# encoding: utf-8
import numpy as np


class Space(object):
    def __init__(self, matrix, id2row):
        self.mat = matrix
        self.id2row = id2row
        self.build_row2id()

    def build_row2id(self):
        self.row2id = {}
        for idx, word in enumerate(self.id2row):
            if word in self.row2id:
                raise ValueError("duplicate word: %s" % word)
            self.row2id[word] = idx

    @classmethod
    def build(cls, lang_vec, lexicon):
        id2row = []
        mat = []
        if lexicon is not None:
            for item in lexicon:
                id2row.append(item)
                mat.append(lang_vec.syn0[lang_vec.vocab[item].index])

        else:
            for item in lang_vec.vocab.keys():
                id2row.append(item)
                mat.append(lang_vec.syn0[lang_vec.vocab[item].index])

        return Space(mat, id2row)

    def normalize(self):
        self.mat = self.mat / np.sqrt(np.sum(np.multiply(self.mat, self.mat), axis=1, keepdims=True))


class TranslationMatrix(object):
    def __init__(self, word_pair, source_lang_vec, target_lang_vec):
        self.source_word, self.target_word = zip(*word_pair)
        self.source_space_vec = source_lang_vec
        self.target_space_vec = target_lang_vec

        self.translation_matrix = None

        self.source_space = self.build_space(source_lang_vec, set(self.source_word))
        self.target_space = self.build_space(target_lang_vec, set(self.target_word))

        self.translation_matrix = self.train(self.source_space, self.target_space)

    def build_space(self, words, lang_vec):
        return Space.build(words, lang_vec)

    def train(self, source_space, target_space):
        source_space.normalize()
        target_space.normalize()

        m1 = source_space.mat[[source_space.row2id[item] for item in self.source_word], :]
        m2 = target_space.mat[[target_space.row2id[item] for item in self.target_word], :]

        return np.linalg.lstsq(m1, m2, -1)[0]

    def translate(self, source_words, topn=5):
        translated_word = {}
        for idx, word in enumerate(source_words):
            if word in self.source_space.row2id:
                source_word_vec = self.source_space.mat[self.source_space.row2id[word], :]
                predicted_word_vec = np.dot(source_word_vec, self.translation_matrix)
                candidates = [i[0] for i in self.target_space_vec.most_similar(positive=[predicted_word_vec], topn=topn)]
            else:
                candidates = []
            translated_word[word] = candidates
        return translated_word

