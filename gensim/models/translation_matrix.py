#!/usr/bin/env python
# encoding: utf-8
import warnings
import numpy as np

from collections import OrderedDict
from gensim import utils
from six import string_types


"""
Produce translation matrix to translate the word from one language to another language, using either
standard nearest neighbour method or globally corrected neighbour retrieval method [1].

This method can be used to augment the existing phrase tables with more candidate translations, or
filter out errors from the translation tables and known dictionaries [2].

Initialize a model with e.g.::

    >>> transmat = TranslationMatrix(word_pair, source_word_vec, target_word_vec)

Train a model with e.g.::

    >>> transmat.train(word_pair)

Persist a model to disk with::

    >>> transmat.save(fname)
    >>> transmat  = TranslationMatrix.load(fname)

Translate the source words to target words, for example

    >>> transmat.translate(["one", "two", "three"], topn=3)

.. [1] Dinu, Georgiana, Angeliki Lazaridou, and Marco Baroni. "Improving zero-shot learning by mitigating the
        hubness problem." arXiv preprint arXiv:1412.6568 (2014).
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
"""


class Space(object):
    """
    An auxiliary class for storing the the words space

    Attributes:
        `mat` (ndarray): each row is the word vector of the lexicon
        `index2word` (list): a list of lexicon
        `word2index` (dict): map the word to index
    """
    def __init__(self, matrix, index2word):
        """
        `matrix`: N * length_of_word_vec, which store the word's vector
        `index2word`: a list of words in the `Space` object
        `word2index`: a dict which for word indexing
        """
        self.mat = matrix
        self.index2word = index2word

        # build a dict to map word to index
        self.word2index = {}
        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx

    @classmethod
    def build(cls, lang_vec, lexicon=None):
        """
        construct a space class for the lexicon, if it's provided.
        Args:
            `lang_vec`: word2vec model that extract word vector for lexicon
            `lexicon`: the default is None, if it is not provided, the lexicon
                    is all the lang_vec's word, i.e. lang_vec.vocab.keys()
        Returns:
            `Space` object for the lexicon
        """
        # `words` to store all the word that
        # `mat` to store all the word vector for the word in 'words' list
        words = []
        mat = []
        if lexicon is not None:
            # if the lexicon is not provided, using the all the Keyedvectors's words as default
            for item in lexicon:
                words.append(item)
                mat.append(lang_vec.syn0[lang_vec.vocab[item].index])

        else:
            for item in lang_vec.vocab.keys():
                words.append(item)
                mat.append(lang_vec.syn0[lang_vec.vocab[item].index])

        return Space(mat, words)

    def normalize(self):
        """ normalized the word vector's matrix """
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
    >>> transmat.train(word_pair)
    >>> translated_word = transmat.translate(words, topn=3)

    """
    def __init__(self, word_pair, source_lang_vec, target_lang_vec, random_state=None):
        """
        Initialize the model from a list pair of `word_pair`. Each word_pair is tupe
         with source language word and target language word.

        Examples: [("one", "uno"), ("two", "due")]

        Args:
            `word_pair` (list): a list pair of `word_pair`
            `source_lang_vec` (KeyedVectors): a set of word vector of source language
            `target_lang_vec` (KeyedVectors): a set of word vector of target language
        """

        if len(word_pair[0]) != 2:
            raise ValueError("Each training data item must contain two different language words.")

        self.source_word, self.target_word = zip(*word_pair)
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec

        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None
        self.source_space = None
        self.target_space = None

    def train(self, word_pair):
        """
        build the translation matrix that mapping from source space to target space.

        Args:
            `source_space` (Space object): source language space
            `target_space` (Space object): target language space

        Returns:
            `translation matrix` that mapping from the source language to target language
        """
        self.source_space = Space.build(self.source_lang_vec, set(self.source_word))
        self.target_space = Space.build(self.target_lang_vec, set(self.target_word))

        self.source_space.normalize()
        self.target_space.normalize()

        m1 = self.source_space.mat[[self.source_space.word2index[item] for item in self.source_word], :]
        m2 = self.target_space.mat[[self.target_space.word2index[item] for item in self.target_word], :]

        self.translation_matrix = np.linalg.lstsq(m1, m2, -1)[0]
        return self.translation_matrix

    def save(self, *args, **kwargs):
        """
        Save the model to file but ignoring the souce_space and target_space
        """
        kwargs['ignore'] = kwargs.get('ignore', ['source_space', 'target_space'])

        super(TranslationMatrix, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """ load the pre-trained translation matrix model"""
        model = super(TranslationMatrix, cls).load(*args, **kwargs)
        return model

    def apply_transmat(self, words_space):
        """
        mapping the source word vector to the target word vector using translation matrix
        Args:
            `words_space`: the `Space` object that constructed for those words to be translate

        Returns:
            A `Space` object that constructed for those mapped words
        """
        return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)

    def translate(self, source_words, topn=5, gc=0, sample_num=None, source_lang_vec=None, target_lang_vec=None):
        """
        translate the word from the source language to the target language, and return the topn
        most similar words.
         Args:
            `source_words`(str/list): single word or a list of words to be translated
            `topn`: return the top N similar words. By default (`topn=5`)
            `gc`: defines the training algorithm. By default (`gc=0`), use standard NN retrieval.
            Otherwise use globally corrected neighbour retrieval method(as described in[1]).
            `sample_num`:  an int parameter that specify the number of word to sample from the source lexicon.
            if `gc=1`, then `sample_num` must be provided.
            `source_lang_vec`: you can specify the source language vector for translation, the default is to use
            the model's source language vector.
            `target_lang_vec`: you can specify the target language vector for retrieving the most similar word,
            the default is to use the model's target language vector.
        Returns:
            A OrderedDict object, each item is (word : `topn` translated words)

        [1] Dinu, Georgiana, Angeliki Lazaridou, and Marco Baroni. "Improving zero-shot learning by mitigating the
        hubness problem." arXiv preprint arXiv:1412.6568 (2014).
        """

        if isinstance(source_words, string_types):
            # pass only one word to translate
            source_words = [source_words]

        # if the language word vector not provided by user, use the model's
        # language word vector as default
        if source_lang_vec is None:
            warnings.warn("The parameter source_lang_vec isn't specified, use the model's source language word vector as default")
            source_lang_vec = self.source_lang_vec

        if target_lang_vec is None:
            warnings.warn("The parameter target_lang_vec isn't specified, use the model's target language word vector as default")
            target_lang_vec = self.target_lang_vec

        # if additional is provided, bootstrapping vocabulary from the source language word vector model.
        if gc:
            if sample_num is None:
                raise RuntimeError("When using the globally corrected neighbour retrieval method, the `sample_num` parameter(i.e. the number of words sampled from source space) must be provided.")
            lexicon = set(source_lang_vec.index2word)
            addition = min(sample_num, len(lexicon) - len(source_words))
            lexicon = self.random_state.choice(list(lexicon.difference(source_words)), addition)
            source_space = Space.build(source_lang_vec, set(source_words).union(set(lexicon)))
        else:
            source_space = Space.build(source_lang_vec, source_words)
        target_space = Space.build(target_lang_vec, )

        # normalize the source vector and target vector
        source_space.normalize()
        target_space.normalize()

        # map the source language to the target language
        mapped_source_space = self.apply_transmat(source_space)

        # use the cosine similarity metric
        sim_matrix = -np.dot(target_space.mat, mapped_source_space.mat.T)

        # if `gc=1`, using corrected retrieval method
        if gc:
            srtd_idx = np.argsort(np.argsort(sim_matrix, axis=1), axis=1)
            sim_matrix_idx = np.argsort(srtd_idx + sim_matrix, axis=0)
        else:
            sim_matrix_idx = np.argsort(sim_matrix, axis=0)

        # translate the words and for each word return the `topn` similar words
        translated_word = OrderedDict()
        for idx, word in enumerate(source_words):
            translated_target_word = []
            # search the most `topn` similar words
            for j in range(topn):
                map_space_id = sim_matrix_idx[j, source_space.word2index[word]]
                translated_target_word.append(target_space.index2word[map_space_id])
            translated_word[word] = translated_target_word
        return translated_word
