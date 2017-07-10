#!/usr/bin/env python
# encoding: utf-8
import numpy as np

from gensim import utils


class Space(object):
    """
    An auxiliary class for store the the words space
    """
    def __init__(self, matrix, id2row):
        """
        matrix: N * length_of_word_vec, which store the word's word vector
        id2row: a list of words
        """
        self.mat = matrix
        self.id2row = id2row
        self.build_row2id()

    def build_row2id(self):
        """
        build a dict to map word to index
        """
        self.row2id = {}
        for idx, word in enumerate(self.id2row):
            if word in self.row2id:
                raise ValueError("duplicate word: %s" % word)
            self.row2id[word] = idx

    @classmethod
    def build(cls, lang_vec, lexicon=None):
        """
        construct a space class for the lexicon, if it's provided.
        Args:
            lang_vec: word2vec model that extract word vector for lexicon
            lexicon: the default is None, if it is not provided, the lexicon
                    is the word that lang_vec's word
        Returns:
            space object for the lexicon
        """
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
        """
        normalized the word vector's matrix
        """
        self.mat = self.mat / np.sqrt(np.sum(np.multiply(self.mat, self.mat), axis=1, keepdims=True))


class TranslationMatrix(utils.SaveLoad):
    """
    Objects of this class realize the translation matrix which map the source language
    to the target language.
     The main methods are:

    1. constructor, which build a translation matrix
    2. the translate method, which given new word and its vector representation,
    we map it to the other language space by computing z = Wx, then return the
    word whose representation is close to z.

    the details use seen the notebook (translation_matrix.ipynb)

    >>> transmat = TranslationMatrix(word_pair, source_lang_vec, target_lang_vec)
    >>> translated_word = transmat.translate(words, topn=3)

    """
    def __init__(self, word_pair, source_lang_vec, target_lang_vec):
        self.source_word, self.target_word = zip(*word_pair)
        self.source_space_vec = source_lang_vec
        self.target_space_vec = target_lang_vec

        self.translation_matrix = None

        self.source_space = self.build_space(source_lang_vec, set(self.source_word))
        self.target_space = self.build_space(target_lang_vec, set(self.target_word))

        self.translation_matrix = self.train(self.source_space, self.target_space)

    def build_space(self, lang_vec, words):
        return Space.build(lang_vec, words)

    def train(self, source_space, target_space):
        """
        build the translation matrix that mapping from source space to target space.
        """
        source_space.normalize()
        target_space.normalize()

        m1 = source_space.mat[[source_space.row2id[item] for item in self.source_word], :]
        m2 = target_space.mat[[target_space.row2id[item] for item in self.target_word], :]

        return np.linalg.lstsq(m1, m2, -1)[0]

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['source_space_vec', 'source_space_vec'])

        super(TranslationMatrix, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(TranslationMatrix, cls).load(*args, **kwargs)
        return model

    def translate(self, source_words=None, topn=5, additional=None, source_lang_vec=None, target_lang_vec=None):
        """
        translate the word from the source language to the target language, and return the topn
        most similar words.
        """
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
