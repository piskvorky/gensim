# Copyright (C) 2017 Parul Sethi <parul1sethi@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Python wrapper around word representation learning from Wordrank.
The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:
>>> model = gensim.models.wrappers.Wordrank('/Users/dummy/wordrank', corpus_file='text8', out_name='wr_model')
>>> print model[word]  # prints vector for given words

.. [1] https://bitbucket.org/shihaoji/wordrank/
.. [2] https://arxiv.org/pdf/1506.02761v3.pdf
"""

from __future__ import division

import logging
import os
import sys
import copy
import multiprocessing

import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from six import string_types
from smart_open import smart_open
from shutil import copyfile, rmtree


logger = logging.getLogger(__name__)


class Wordrank(KeyedVectors):
    """
    Class for word vector training using Wordrank. Communication between Wordrank and Python
    takes place by working with data files on disk and calling the Wordrank binary and glove's
    helper binaries (for preparing training data) with subprocess module.
    """
    
    @classmethod
    def train(cls, wr_path, corpus_file, out_name, size=100, window=15, symmetric=1, min_count=5, max_vocab_size=0,
              sgd_num=100, lrate=0.001, period=10, iter=90, epsilon=0.75, dump_period=10, reg=0, alpha=100,
              beta=99, loss='hinge', memory=4.0, cleanup_files=True, sorted_vocab=1, ensemble=0):
        """
        `wr_path` is the path to the Wordrank directory.
        `corpus_file` is the filename of the text file to be used for training the Wordrank model.
        Expects file to contain space-separated tokens in a single line
        `out_name` is name of the directory which will be created (in wordrank folder) to save embeddings and training data.
        `size` is the dimensionality of the feature vectors.
        `window` is the number of context words to the left (and to the right, if symmetric = 1).
        `symmetric` if 0, only use left context words, else use left and right both.
        `min_count` = ignore all words with total frequency lower than this.
        `max_vocab_size` upper bound on vocabulary size, i.e. keep the <int> most frequent words. Default is 0 for no limit.
        `sgd_num` number of SGD taken for each data point.
        `lrate` is the learning rate (too high diverges, give Nan).
        `period` is the period of xi variable updates
        `iter` = number of iterations (epochs) over the corpus.
        `epsilon` is the power scaling value for weighting function.
        `dump_period` is the period after which embeddings should be dumped.
        `reg` is the value of regularization parameter.
        `alpha` is the alpha parameter of gamma distribution.
        `beta` is the beta parameter of gamma distribution.
        `loss` = name of the loss (logistic, hinge).
        `memory` = soft limit for memory consumption, in GB.
        `cleanup_files` if True, delete directory and files used by this wrapper, setting to False can be useful for debugging
        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before assigning word indexes.
        `ensemble` = 0 (default), use ensemble of word and context vectors
        """

        meta_data_path = 'matrix.meta'
        vocab_file = 'vocab.txt'
        temp_vocab_file = 'tempvocab.txt'
        cooccurrence_file = 'cooccurrence'
        cooccurrence_shuf_file = 'wiki.toy'
        meta_file = 'meta'

        # prepare training data (cooccurrence matrix and vocab)
        model_dir = os.path.join(wr_path, out_name)
        meta_dir = os.path.join(model_dir, 'meta')
        os.makedirs(meta_dir)
        logger.info("Dumped data will be stored in '%s'", model_dir)
        copyfile(corpus_file, os.path.join(meta_dir, corpus_file.split('/')[-1]))
        os.chdir(meta_dir)

        cmd_vocab_count = ['../../glove/vocab_count', '-min-count', str(min_count), '-max-vocab', str(max_vocab_size)]
        cmd_cooccurence_count = ['../../glove/cooccur', '-memory', str(memory), '-vocab-file', temp_vocab_file, '-window-size', str(window), '-symmetric', str(symmetric)]
        cmd_shuffle_cooccurences = ['../../glove/shuffle', '-memory', str(memory)]
        cmd_del_vocab_freq = ['cut', '-d', " ", '-f', '1', temp_vocab_file]

        commands = [cmd_vocab_count, cmd_cooccurence_count, cmd_shuffle_cooccurences]
        input_fnames = [corpus_file.split('/')[-1], corpus_file.split('/')[-1], cooccurrence_file]
        output_fnames = [temp_vocab_file, cooccurrence_file, cooccurrence_shuf_file]

        logger.info("Prepare training data (%s) using glove code", ", ".join(input_fnames))
        for command, input_fname, output_fname in zip(commands, input_fnames, output_fnames):
            with smart_open(input_fname, 'rb') as r:
                with smart_open(output_fname, 'wb') as w:
                    utils.check_output(w, args=command, stdin=r)

        logger.info("Deleting frequencies from vocab file")
        with smart_open(vocab_file, 'wb') as w:
            utils.check_output(w, args=cmd_del_vocab_freq)

        with smart_open(vocab_file, 'rb') as f:
            numwords = sum(1 for line in f)
        with smart_open(cooccurrence_shuf_file, 'rb') as f:
            numlines = sum(1 for line in f)
        with smart_open(meta_file, 'wb') as f:
            meta_info = "{0} {1}\n{2} {3}\n{4} {5}".format(numwords, numwords, numlines, cooccurrence_shuf_file, numwords, vocab_file)
            f.write(meta_info.encode('utf-8'))
            
        if iter % dump_period == 0:
            iter += 1
        else:
            logger.warning(
                'Resultant embedding will be from %d iterations rather than the input %d iterations, '
                'as wordrank dumps the embedding only at dump_period intervals. '
                'Input an appropriate combination of parameters (iter, dump_period) such that '
                '"iter mod dump_period" is zero.', iter - (iter % dump_period), iter
                )

        wr_args = {
            'path': 'meta',
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
        # run wordrank executable with wr_args
        cmd = ['mpirun', '-np', '1', '../wordrank']
        for option, value in wr_args.items():
            cmd.append('--%s' % option)
            cmd.append(str(value))
        logger.info("Running wordrank binary")
        output = utils.check_output(args=cmd)

        # use embeddings from max. iteration's dump
        max_iter_dump = iter - (iter % dump_period)
        copyfile('model_word_%d.txt' % max_iter_dump, 'wordrank.words')
        copyfile('model_context_%d.txt' % max_iter_dump, 'wordrank.contexts')
        model = cls.load_wordrank_model('wordrank.words', os.path.join('meta', vocab_file), 'wordrank.contexts', sorted_vocab, ensemble)
        os.chdir('../..')

        if cleanup_files:
            rmtree(model_dir)
        return model

    @classmethod
    def load_wordrank_model(cls, model_file, vocab_file=None, context_file=None, sorted_vocab=1, ensemble=1):
        glove2word2vec(model_file, model_file+'.w2vformat')
        model = cls.load_word2vec_format('%s.w2vformat' % model_file)
        if ensemble and context_file:
            model.ensemble_embedding(model_file, context_file)
        if sorted_vocab and vocab_file:
            model.sort_embeddings(vocab_file)
        return model

    def sort_embeddings(self, vocab_file):
        """Sort embeddings according to word frequency."""
        counts = {}
        vocab_size = len(self.vocab)
        prev_syn0 = copy.deepcopy(self.syn0)
        prev_vocab = copy.deepcopy(self.vocab)
        self.index2word = []

        # sort embeddings using frequency sorted vocab file in wordrank
        with utils.smart_open(vocab_file) as fin:
            for index, line in enumerate(fin):
                word, count = utils.to_unicode(line).strip(), vocab_size - index
                # store word with it's count in a dict
                counts[word] = int(count)
                # build new index2word with frequency sorted words
                self.index2word.append(word)
        assert len(self.index2word) == vocab_size, 'mismatch between vocab sizes'

        for word_id, word in enumerate(self.index2word):
            self.syn0[word_id] = prev_syn0[prev_vocab[word].index]
            self.vocab[word].index = word_id
            self.vocab[word].count = counts[word]

    def ensemble_embedding(self, word_embedding, context_embedding):
        """Replace syn0 with the sum of context and word embeddings."""
        glove2word2vec(context_embedding, context_embedding+'.w2vformat')
        w_emb = KeyedVectors.load_word2vec_format('%s.w2vformat' % word_embedding)
        c_emb = KeyedVectors.load_word2vec_format('%s.w2vformat' % context_embedding)
        # compare vocab words using keys of dict vocab
        assert set(w_emb.vocab) == set(c_emb.vocab), 'Vocabs are not same for both embeddings'

        # sort context embedding to have words in same order as word embedding
        prev_c_emb = copy.deepcopy(c_emb.syn0)
        for word_id, word in enumerate(w_emb.index2word):
            c_emb.syn0[word_id] = prev_c_emb[c_emb.vocab[word].index]
        # add vectors of the two embeddings
        new_emb = w_emb.syn0 + c_emb.syn0
        self.syn0 = new_emb
        return new_emb

