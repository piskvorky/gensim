# Copyright (C) 2017 Parul Sethi <parul1sethi@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Python wrapper around `Wordrank <https://bitbucket.org/shihaoji/wordrank/>`_.
Original paper: `"WordRank: Learning Word Embeddings via Robust Ranking " <https://arxiv.org/pdf/1506.02761v3.pdf>`_.

Installation
------------
Use `official guide <https://github.com/shihaoji/wordrank>`_ or this one

* On Linux ::

    sudo yum install boost-devel #(on RedHat/Centos)
    sudo apt-get install libboost-all-dev #(on Ubuntu)

    git clone https://bitbucket.org/shihaoji/wordrank
    cd wordrank/
    # replace icc to gcc in install.sh
    ./install.sh

* On MacOS ::

    brew install cmake
    brew install wget
    brew install boost
    brew install mercurial

    git clone https://bitbucket.org/shihaoji/wordrank
    cd wordrank/
    # replace icc to gcc in install.sh
    ./install.sh

Examples
--------
.. sourcecode:: pycon

    >>> from gensim.models.wrappers import Wordrank
    >>>
    >>> path_to_wordrank_binary = '/path/to/wordrank/binary'
    >>> model = Wordrank.train(path_to_wordrank_binary, corpus_file='text8', out_name='wr_model')
    >>>
    >>> print(model["hello"])  # prints vector for given words

Warnings
--------
Note that the wrapper might not work in a docker container for large datasets due to memory limits (caused by MPI).

"""

from __future__ import division

import logging
import os
import copy
import multiprocessing

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from smart_open import smart_open
from shutil import copyfile, rmtree


logger = logging.getLogger(__name__)


class Wordrank(KeyedVectors):
    """Python wrapper using `Wordrank implementation <https://bitbucket.org/shihaoji/wordrank/>`_

    Communication between Wordrank and Python takes place by working with data
    files on disk and calling the Wordrank binary and glove's helper binaries
    (for preparing training data) with subprocess module.

    Warnings
    --------
    This is **only** python wrapper for `Wordrank implementation <https://bitbucket.org/shihaoji/wordrank/>`_,
    you need to install original implementation first and pass the path to wordrank dir to ``wr_path``.

    """
    @classmethod
    def train(cls, wr_path, corpus_file, out_name, size=100, window=15, symmetric=1, min_count=5, max_vocab_size=0,
              sgd_num=100, lrate=0.001, period=10, iter=90, epsilon=0.75, dump_period=10, reg=0, alpha=100,
              beta=99, loss='hinge', memory=4.0, np=1, cleanup_files=False, sorted_vocab=1, ensemble=0):
        """Train model.

        Parameters
        ----------
        wr_path : str
            Absolute path to the Wordrank directory.
        corpus_file : str
            Path to corpus file, expected space-separated tokens in a each line format.
        out_name : str
            Name of the directory which will be created (in wordrank folder) to save embeddings and training data:
                * ``model_word_current_<iter>.txt`` - Word Embeddings saved after every dump_period.
                * ``model_context_current_<iter>.txt`` - Context Embeddings saved after every dump_period.
                * ``meta/vocab.txt`` - vocab file.
                * ``meta/wiki.toy`` - word-word concurrence values.
        size : int, optional
            Dimensionality of the feature vectors.
        window : int, optional
            Number of context words to the left (and to the right, if `symmetric = 1`).
        symmetric : {0, 1}, optional
            If 1 - using symmetric windows, if 0 - will use only left context words.
        min_count : int, optional
            Ignore all words with total frequency lower than `min_count`.
        max_vocab_size : int, optional
            Upper bound on vocabulary size, i.e. keep the <int> most frequent words. If 0 - no limit.
        sgd_num : int, optional
            Number of SGD taken for each data point.
        lrate : float, optional
            Learning rate (attention: too high diverges, give Nan).
        period : int, optional
            Period of xi variable updates.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        epsilon : float, optional
            Power scaling value for weighting function.
        dump_period : int, optional
            Period after which embeddings should be dumped.
        reg : int, optional
            Value of regularization parameter.
        alpha : int, optional
            Alpha parameter of gamma distribution.
        beta : int, optional
            Beta parameter of gamma distribution.
        loss : {"logistic", "hinge"}, optional
            Name of the loss function.
        memory : float, optional
            Soft limit for memory consumption, in GB.
        np : int, optional
            Number of process to execute (mpirun option).
        cleanup_files : bool, optional
            If True, delete directory and files used by this wrapper.
        sorted_vocab : {0, 1}, optional
            If 1 - sort the vocabulary by descending frequency before assigning word indexes, otherwise - do nothing.
        ensemble : {0, 1}, optional
            If 1 - use ensemble of word and context vectors.

        """

        # prepare training data (cooccurrence matrix and vocab)
        model_dir = os.path.join(wr_path, out_name)
        meta_dir = os.path.join(model_dir, 'meta')
        os.makedirs(meta_dir)
        logger.info("Dumped data will be stored in '%s'", model_dir)
        copyfile(corpus_file, os.path.join(meta_dir, corpus_file.split('/')[-1]))

        vocab_file = os.path.join(meta_dir, 'vocab.txt')
        temp_vocab_file = os.path.join(meta_dir, 'tempvocab.txt')
        cooccurrence_file = os.path.join(meta_dir, 'cooccurrence')
        cooccurrence_shuf_file = os.path.join(meta_dir, 'wiki.toy')
        meta_file = os.path.join(meta_dir, 'meta')

        cmd_vocab_count = [
            os.path.join(wr_path, 'glove', 'vocab_count'),
            '-min-count', str(min_count), '-max-vocab', str(max_vocab_size)
        ]
        cmd_cooccurence_count = [
            os.path.join(wr_path, 'glove', 'cooccur'), '-memory', str(memory),
            '-vocab-file', temp_vocab_file, '-window-size', str(window), '-symmetric', str(symmetric)
        ]
        cmd_shuffle_cooccurences = [os.path.join(wr_path, 'glove', 'shuffle'), '-memory', str(memory)]
        cmd_del_vocab_freq = ['cut', '-d', " ", '-f', '1', temp_vocab_file]

        commands = [cmd_vocab_count, cmd_cooccurence_count, cmd_shuffle_cooccurences]
        input_fnames = [
            os.path.join(meta_dir, os.path.split(corpus_file)[-1]),
            os.path.join(meta_dir, os.path.split(corpus_file)[-1]),
            cooccurrence_file
        ]
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
            numwords = sum(1 for _ in f)
        with smart_open(cooccurrence_shuf_file, 'rb') as f:
            numlines = sum(1 for _ in f)
        with smart_open(meta_file, 'wb') as f:
            meta_info = "{0} {1}\n{2} {3}\n{4} {5}".format(
                numwords, numwords, numlines, cooccurrence_shuf_file.split('/')[-1],
                numwords, vocab_file.split('/')[-1]
            )
            f.write(meta_info.encode('utf-8'))

        if iter % dump_period == 0:
            iter += 1
        else:
            logger.warning(
                "Resultant embedding will be from %d iterations rather than the input %d iterations, "
                "as wordrank dumps the embedding only at dump_period intervals. "
                "Input an appropriate combination of parameters (iter, dump_period) "
                "such that \"iter mod dump_period\" is zero.",
                iter - (iter % dump_period), iter
            )

        wr_args = {
            'path': meta_dir,
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

        # run wordrank executable with wr_args
        cmd = ['mpirun', '-np', str(np), os.path.join(wr_path, 'wordrank')]
        for option, value in wr_args.items():
            cmd.append('--%s' % option)
            cmd.append(str(value))
        logger.info("Running wordrank binary")
        utils.check_output(args=cmd)

        # use embeddings from max. iteration's dump
        max_iter_dump = iter - (iter % dump_period)
        os.rename('model_word_%d.txt' % max_iter_dump, os.path.join(model_dir, 'wordrank.words'))
        os.rename('model_context_%d.txt' % max_iter_dump, os.path.join(model_dir, 'wordrank.contexts'))
        model = cls.load_wordrank_model(
            os.path.join(model_dir, 'wordrank.words'), vocab_file,
            os.path.join(model_dir, 'wordrank.contexts'), sorted_vocab, ensemble
        )

        if cleanup_files:
            rmtree(model_dir)
        return model

    @classmethod
    def load_wordrank_model(cls, model_file, vocab_file=None, context_file=None, sorted_vocab=1, ensemble=1):
        """Load model from `model_file`.

        Parameters
        ----------
        model_file : str
            Path to model in GloVe format.
        vocab_file : str, optional
            Path to file with vocabulary.
        context_file : str, optional
            Path to file with context-embedding in word2vec_format.
        sorted_vocab : {0, 1}, optional
            If 1 - sort the vocabulary by descending frequency before assigning word indexes, otherwise - do nothing.
        ensemble : {0, 1}, optional
            If 1 - use ensemble of word and context vectors.

        """
        glove2word2vec(model_file, model_file + '.w2vformat')
        model = cls.load_word2vec_format('%s.w2vformat' % model_file)
        if ensemble and context_file:
            model.ensemble_embedding(model_file, context_file)
        if sorted_vocab and vocab_file:
            model.sort_embeddings(vocab_file)
        return model

    def sort_embeddings(self, vocab_file):
        """Sort embeddings according to word frequency.

        Parameters
        ----------
        vocab_file : str
            Path to file with vocabulary.

        """
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
        """Replace current syn0 with the sum of context and word embeddings.

        Parameters
        ----------
        word_embedding : str
            Path to word embeddings in GloVe format.
        context_embedding : str
            Path to context embeddings in word2vec_format.

        Returns
        -------
        numpy.ndarray
            Matrix with new embeddings.

        """
        glove2word2vec(context_embedding, context_embedding + '.w2vformat')
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
