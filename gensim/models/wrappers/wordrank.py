#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python wrapper around word representation learning from FastText, a library for efficient learning
of word representations and sentence classification [1].
This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.
The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.
Example:
>>> model = gensim.models.wrappers.LdaMallet('/Users/kofola/fastText/fasttext', corpus_file='text8')
>>> print model[word]  # prints vector for given words
.. [1] https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information
"""


import logging
import tempfile
import os
import struct
import copy
import multiprocessing

import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from six import string_types
from collections import Counter
from smart_open import smart_open

if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess

logger = logging.getLogger(__name__)


class Wordrank(Word2Vec):
    """
    Class for word vector training using Wordrank. Communication between FastText and Python
    takes place by working with data files on disk and calling the FastText binary with
    subprocess.call().
    Implements functionality similar to [fasttext.py](https://github.com/salestock/fastText.py),
    improving speed and scope of functionality like `most_similar`, `accuracy` by extracting vectors
    into numpy matrix.
    """
    
    @classmethod
    def train(cls, wr_path, corpus_file, out_path=None, size=100, window=5, min_count=5, 
                 max_vocab_size=10000000, sgd_num=100, lrate=0.025, period=10, iter=11, epsilon=0.75, 
                 dump_period=10, reg=0, alpha=100, beta=99, loss='hinge', memory=4.0, sorted_vocab=1, ensemble=0):
        """
        `ft_path` is the path to the Wordrank directory, e.g. `/home/kofola/fastText/fasttext`.
        `corpus_file` is the filename of the text file to be used for training the FastText model.
        Expects file to contain space-separated tokens in a single line
        `model` defines the training algorithm. By default, cbow is used. Accepted values are
        cbow, skipgram.
        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).
        `min_count` = ignore all words with total frequency lower than this.
        `loss` = defines training objective. Allowed values are `hs` (hierarchical softmax),
        `ns` (negative sampling) and `softmax`. Defaults to `ns`
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).
        `negative` = the value for negative specifies how many "noise words" should be drawn
        (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        Only relevant when `loss` is set to `ns`
        `iter` = number of iterations (epochs) over the corpus. Default is 5.
        `min_n` = min length of char ngrams to be used for training word representations. Default is 1.
        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        greater than `min_n` to avoid char ngrams being used. Default is 5.
        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.
        """
        # self.path = out_path
        # self.sgd_num = sgd_num
        # self.lrate = lrate
        # self.iter = iter
        # self.epsilon = epsilon
        # self.dim = size
        # self.reg = reg
        # self.alpha = alpha
        # self.beta = beta
        # self.loss = loss
        # self.min_count = min_count
        # self.max_vocab_size = max_vocab_size
        # self.memory = memory
        # self.window = window
        # self.sorted_vocab = sorted_vocab
        # self.ensemble = ensemble

        # wr_path = 'Users/parul/Desktop/'+wr_path

        meta_data_path = 'matrix.meta'
        vocab_file = 'vocab.txt'
        temp_vocab_file = 'tempvocab.txt'
        cooccurrence_file = 'cooccurrence'
        cooccurrence_shuf_file = 'wiki.toy'
        meta_file = 'meta'

        cmd0 = [wr_path+'/glove/vocab_count', '-min-count', str(min_count), '-max-vocab', str(max_vocab_size)]
        cmd1 = [wr_path+'/glove/cooccur', '-memory', str(memory), '-vocab-file', temp_vocab_file, '-window-size', str(window)]
        cmd2 = [wr_path+'/glove/shuffle', '-memory', str(memory)]

        
        # utils.check_output(args=['rm', '-rf', out_path+';', 'mkdir', out_path+';', 'cd', out_path])
        os.makedirs(out_path+'/'+meta_data_path)
        os.rename(corpus_file, out_path+'/'+meta_data_path+'/'+corpus_file)
        cmds = [cmd0, cmd1, cmd2]
        ins = [corpus_file, corpus_file, cooccurrence_file]
        outs = [temp_vocab_file, cooccurrence_file, cooccurrence_shuf_file]
        
        os.chdir(out_path+'/'+meta_data_path)
        e = [excecute_command(cmd, inp, out) for cmd, inp, out in zip(cmds, ins, outs)]

        outp = open(vocab_file, 'w')
        process = subprocess.Popen(args=['cut', '-d', " ", '-f', '1', temp_vocab_file], stdout=outp)
        output, err = process.communicate()
        outp.flush()

        with smart_open(vocab_file) as f:
            numwords = sum(1 for line in f)
        with smart_open(cooccurrence_shuf_file) as f:
            numlines = sum(1 for line in f)

        meta_content = "{} {}\n{} {}\n{} {}".format(numwords, numwords, numlines, cooccurrence_shuf_file, numwords, vocab_file)
        with open(meta_file, 'w') as fin:
            fin.write(meta_content)

        wr_args = {
            'path': meta_data_path,
            'nthread': multiprocessing.cpu_count(),
            'sgd_num': sgd_num,
            'lrate': lrate,
            'period': period,
            'iter': iter,
            'epsilon': epsilon,
            'dump_prefix': 'model',
            'dump_period': dump_period,
            'dim': size,
            'reg': reg,
            'alpha': alpha,
            'beta': beta,
            'loss': loss
        }

        os.chdir('..')
        cmd = ['mpirun', '-np', '1', '../'+wr_path+'/wordrank']
        for option, value in wr_args.items():
            cmd.append("--%s" % option)
            cmd.append(str(value))

        output = utils.check_output(args=cmd)
        p = utils.check_output(args=['cp', '$model_word_$1.txt', '$wordrank.words'])
        p = utils.check_output(args=['cp', '$model_context_$1.txt', '$wordrank.contexts'])

        # model = cls.load_wordrank_model(out_path+'/wordrank.words', out_path+'/'+vocab_file, out_path+'/wordrank.contexts', sorted_vocab, ensemble)
        model = cls.load_wordrank_model('wordrank.words', vocab_file, 'wordrank.contexts', sorted_vocab, ensemble)
        return model

    @classmethod
    def load_wordrank_model(cls, model_file, vocab_file=None, context_file=None, sort=1, ensemble=1):
        glove2word2vec(model_file, model_file+'.w2vformat')
        model = cls.load_word2vec_format('%s.w2vformat' % model_file)
        if ensemble and context_file:
            model.ensemble_embedding(model_file, context_file)
        if sort and vocab_file:
            model.sort_embeddings(vocab_file)
        return model

    def sort_embeddings(self, vocab_file):
        counts = {}
        vocab_size = len(self.wv.vocab)
        prev_syn0 = copy.deepcopy(self.wv.syn0)
        prev_vocab = copy.deepcopy(self.wv.vocab)
        self.wv.index2word = []

        with utils.smart_open(vocab_file) as fin:
            for index, line in enumerate(fin):
                word, count = utils.to_unicode(line).strip(), vocab_size - index
                counts[word] = int(count)
                self.wv.index2word.append(word)
        assert len(self.wv.index2word) == vocab_size, 'mismatch between vocab sizes'

        for word_id, word in enumerate(self.wv.index2word):
            self.wv.syn0[word_id] = prev_syn0[prev_vocab[word].index]
            self.wv.vocab[word].index = word_id
            self.wv.vocab[word].count = counts[word]

    def ensemble_embedding(self, word_embedding, context_embedding):
        glove2word2vec(word_embedding, word_embedding+'.w2vformat')
        glove2word2vec(context_embedding, context_embedding+'.w2vformat')
        w_emb = self.load_word2vec_format('%s.w2vformat' % word_embedding)
        c_emb = self.load_word2vec_format('%s.w2vformat' % context_embedding)
        assert Counter(w_emb.index2word) == Counter(c_emb.index2word), 'Vocabs are not same for both embeddings'

        prev_c_emb = copy.deepcopy(c_emb.wv.syn0)
        for word_id, word in enumerate(w_emb.wv.index2word):
            c_emb.wv.syn0[word_id] = prev_c_emb[c_emb.wv.vocab[word].index]
        new_emb = w_emb.wv.syn0 + c_emb.wv.syn0
        self.wv.syn0 = new_emb


def excecute_command(cmd, stdin, stdout):
    inp = open(stdin)
    out = open(stdout, 'w')
    process = subprocess.Popen(cmd, stdin=inp, stdout=out)
    output, err = process.communicate()
    out.flush()

