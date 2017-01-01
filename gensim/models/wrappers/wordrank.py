"""
Python wrapper around word representation learning from Wordrank.
The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:
>>> model = gensim.models.wrappers.Wordrank('/Users/dummy/wordrank', corpus_file='text8', out_path='wr_model')
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
from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from six import string_types
from smart_open import smart_open
from shutil import copyfile, rmtree

if sys.version_info[:2] == (2, 6):
    from backport_collections import Counter
else:
    from collections import Counter

logger = logging.getLogger(__name__)


class Wordrank(Word2Vec):
    """
    Class for word vector training using Wordrank. Communication between Wordrank and Python
    takes place by working with data files on disk and calling the Wordrank binary and glove's
    helper binaries (for preparing training data) with subprocess module.
    """
    
    @classmethod
    def train(cls, wr_path, corpus_file, out_path, size=100, window=15, symmetric=1, min_count=5, max_vocab_size=0,
              sgd_num=100, lrate=0.001, period=10, iter=90, epsilon=0.75, dump_period=10, reg=0, alpha=100,
              beta=99, loss='hinge', memory=4.0, cleanup_files=True, sorted_vocab=1, ensemble=1):
        """
        `wr_path` is the path to the Wordrank directory.
        `corpus_file` is the filename of the text file to be used for training the Wordrank model.
        Expects file to contain space-separated tokens in a single line
        `out_path` is the path to directory which will be created to save embeddings and training data.
        `size` is the dimensionality of the feature vectors.
        `window` is the number of context words to the left (and to the right, if symmetric = 1).
        symmetric` if 0, only use left context words, else use left and right both.
        `min_count` = ignore all words with total frequency lower than this.
        `max_vocab_size` upper bound on vocabulary size, i.e. keep the <int> most frequent words. Default is 0 for no limit.
        `sgd_num` number of SGD taken for each data point.
        `lrate` is the learning rate (too high diverges, give Nan).
        `period` is the period of xi variable updates
        `iter` = number of iterations (epochs) over the corpus.
        `epsilon` is the power scaling value for weighting function.
        `dump_period` is the period after which parameters should be dumped.
        `reg` is the value of regularization parameter.
        `alpha` is the alpha parameter of gamma distribution.
        `beta` is the beta parameter of gamma distribution.
        `loss` = name of the loss (logistic, hinge).
        `memory` = soft limit for memory consumption, in GB.
        `cleanup_files` if to delete directory and files used by this wrapper, setting to False can be useful for debugging
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
        model_dir = os.path.join(wr_path, out_path)
        meta_dir = os.path.join(model_dir, 'meta')
        os.makedirs(meta_dir)
        logger.info("Dumped data will be stored in '%s'", model_dir)
        copyfile(corpus_file, os.path.join(meta_dir, corpus_file.split('/')[-1]))
        os.chdir(meta_dir)

        cmd0 = ['../../glove/vocab_count', '-min-count', str(min_count), '-max-vocab', str(max_vocab_size)]
        cmd1 = ['../../glove/cooccur', '-memory', str(memory), '-vocab-file', temp_vocab_file, '-window-size', str(window), '-symmetric', str(symmetric)]
        cmd2 = ['../../glove/shuffle', '-memory', str(memory)]
        cmd3 = ['cut', '-d', " ", '-f', '1', temp_vocab_file]
        cmds = [cmd0, cmd1, cmd2, cmd3]
        logger.info("Preparing training data using glove code '%s'", cmds)
        o0 = smart_open(temp_vocab_file, 'w')
        o1 = smart_open(cooccurrence_file, 'w')
        o2 = smart_open(cooccurrence_shuf_file, 'w')
        o3 = smart_open(vocab_file, 'w')
        i0 = smart_open(corpus_file.split('/')[-1])
        i1 = smart_open(corpus_file.split('/')[-1])
        i2 = smart_open(cooccurrence_file)
        i3 = None
        outputs = [o0, o1, o2, o3]
        inputs = [i0, i1, i2, i3]
        prepare_train_data = [utils.check_output(cmd, stdin=inp, stdout=out) for cmd, inp, out in zip(cmds, inputs, outputs)]
        o0.close()
        o1.close()
        o2.close()
        o3.close()
        i0.close()
        i1.close()
        i2.close()

        with smart_open(vocab_file) as f:
            numwords = sum(1 for line in f)
        with smart_open(cooccurrence_shuf_file) as f:
            numlines = sum(1 for line in f)
        with smart_open(meta_file, 'w') as f:
            f.write("{0} {1}\n{2} {3}\n{4} {5}".format(numwords, numwords, numlines, cooccurrence_shuf_file, numwords, vocab_file))

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
            cmd.append("--%s" % option)
            cmd.append(str(value))
        logger.info("Running wordrank binary '%s'", cmd)
        output = utils.check_output(args=cmd)

        max_iter_dump = iter / dump_period * dump_period - 1
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
        """Addition of two embeddings."""
        glove2word2vec(word_embedding, word_embedding+'.w2vformat')
        glove2word2vec(context_embedding, context_embedding+'.w2vformat')
        w_emb = Word2Vec.load_word2vec_format('%s.w2vformat' % word_embedding)
        c_emb = Word2Vec.load_word2vec_format('%s.w2vformat' % context_embedding)
        assert Counter(w_emb.index2word) == Counter(c_emb.index2word), 'Vocabs are not same for both embeddings'

        prev_c_emb = copy.deepcopy(c_emb.wv.syn0)
        for word_id, word in enumerate(w_emb.wv.index2word):
            c_emb.wv.syn0[word_id] = prev_c_emb[c_emb.wv.vocab[word].index]
        new_emb = w_emb.wv.syn0 + c_emb.wv.syn0
        self.wv.syn0 = new_emb
        return new_emb

