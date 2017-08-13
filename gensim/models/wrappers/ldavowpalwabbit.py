#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Dave Challis <dave@suicas.net>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Python wrapper around Vowpal Wabbit's Latent Dirichlet Allocation (LDA)
implementation [1]_.

This uses Matt Hoffman's online algorithm, for LDA [2]_, i.e. the same
algorithm that Gensim's LdaModel is based on.

Note: Currently working and tested with Vowpal Wabbit versions 7.10 to 8.1.1.
Vowpal Wabbit's API isn't currently stable, so this may or may not work with
older/newer versions. The aim will be to ensure this wrapper always works with
the latest release of Vowpal Wabbit.

Tested with python 2.6, 2.7, and 3.4.

Example:

    >>> # train model
    >>> lda = gensim.models.wrappers.LdaVowpalWabbit('/usr/local/bin/vw',
                                                     corpus=corpus,
                                                     num_topics=20,
                                                     id2word=dictionary)

    >>> # update an existing model
    >>> lda.update(another_corpus)

    >>> # get topic probability distributions for a document
    >>> print(lda[doc_bow])

    >>> # print 10 topics
    >>> print(lda.print_topics())

    >>> # save/load the trained model:

    >>> lda.save('vw_lda.model')
    >>> lda = gensim.models.wrappers.LdaVowpalWabbit.load('vw_lda.model')

    >>> # get bound on log perplexity for given test set
    >>> print(lda.log_perpexity(test_corpus))

Vowpal Wabbit works on files, so this wrapper maintains a temporary directory
while it's around, reading/writing there as necessary.

Output from Vowpal Wabbit is logged at either INFO or DEBUG levels, enable
logging to view this.

.. [1] https://github.com/JohnLangford/vowpal_wabbit/wiki
.. [2] http://www.cs.princeton.edu/~mdhoffma/
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import shutil
import subprocess
import tempfile

import numpy

from gensim import utils, matutils
from gensim.models.ldamodel import LdaModel

LOG = logging.getLogger(__name__)


class LdaVowpalWabbit(utils.SaveLoad):
    """Class for LDA training using Vowpal Wabbit's online LDA. Communication
    between Vowpal Wabbit and Python takes place by passing around data files
    on disk and calling the 'vw' binary with the subprocess module.
    """
    def __init__(self, vw_path, corpus=None, num_topics=100, id2word=None,
                 chunksize=256, passes=1, alpha=0.1, eta=0.1, decay=0.5,
                 offset=1, gamma_threshold=0.001, random_seed=None,
                 cleanup_files=True, tmp_prefix='tmp'):
        """`vw_path` is the path to Vowpal Wabbit's 'vw' executable.

        `corpus` is an iterable training corpus. If given, training will
        start immediately, otherwise the model is left untrained (presumably
        because you want to call `update()` manually).

        `num_topics` is the number of requested latent topics to be extracted
        from the training corpus.
        Corresponds to VW's '--lda <num_topics>' argument.

        `id2word` is a mapping from word ids (integers) to words (strings).
        It is used to determine the vocabulary size, as well as for debugging
        and topic printing.

        `chunksize` is the number of documents examined in each batch.
        Corresponds to VW's '--minibatch <batch_size>' argument.

        `passes` is the number of passes over the dataset to use.
        Corresponds to VW's '--passes <passes>' argument.

        `alpha` is a float effecting sparsity of per-document topic weights.
        This is applied symmetrically, and should be set higher to when
        documents are thought to look more similar.
        Corresponds to VW's '--lda_alpha <alpha>' argument.

        `eta` is a float which affects the sparsity of topic distributions.
        This is applied symmetrically, and should be set higher when topics
        are thought to look more similar.
        Corresponds to VW's '--lda_rho <rho>' argument.

        `decay` learning rate decay, affects how quickly learnt values
        are forgotten. Should be set to a value between 0.5 and 1.0 to
        guarantee convergence.
        Corresponds to VW's '--power_t <tau>' argument.

        `offset` integer learning offset, set to higher values to slow down
        learning on early iterations of the algorithm.
        Corresponds to VW's '--initial_t <tau>' argument.

        `gamma_threshold` affects when learning loop will be broken out of,
        higher values will result in earlier loop completion.
        Corresponds to VW's '--epsilon <eps>' argument.

        `random_seed` sets Vowpal Wabbit's random seed when learning.
        Corresponds to VW's '--random_seed <seed>' argument.

        `cleanup_files` whether or not to delete temporary directory and files
        used by this wrapper. Setting to False can be useful for debugging,
        or for re-using Vowpal Wabbit files elsewhere.

        `tmp_prefix` used to prefix temporary working directory name.
        """
        # default parameters are taken from Vowpal Wabbit's defaults, and
        # parameter names changed to match Gensim's LdaModel where possible
        self.vw_path = vw_path
        self.id2word = id2word

        if self.id2word is None:
            if corpus is None:
                raise ValueError('at least one of corpus/id2word must be '
                                 'specified, to establish input space '
                                 'dimensionality')
            LOG.warning('no word id mapping provided; initializing from '
                        'corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError('cannot compute LDA over an empty collection '
                             '(no terms)')

        # LDA parameters
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.alpha = alpha
        self.eta = eta
        self.gamma_threshold = gamma_threshold
        self.offset = offset
        self.decay = decay
        self.random_seed = random_seed
        self._initial_offset = offset

        # temporary files used for Vowpal Wabbit input/output
        self.tmp_dir = None
        self.tmp_prefix = tmp_prefix
        self.cleanup_files = cleanup_files
        self._init_temp_dir(tmp_prefix)

        # used for saving/loading this model's state
        self._model_data = None
        self._topics_data = None

        # cache loaded topics as numpy array
        self._topics = None

        if corpus is not None:
            self.train(corpus)

    def train(self, corpus):
        """Clear any existing model state, and train on given corpus."""
        LOG.debug('Training new model from corpus')

        # reset any existing offset, model, or topics generated
        self.offset = self._initial_offset
        self._topics = None

        corpus_size = write_corpus_as_vw(corpus, self._corpus_filename)

        cmd = self._get_vw_train_command(corpus_size)

        _run_vw_command(cmd)

        # ensure that future updates of this model use correct offset
        self.offset += corpus_size

    def update(self, corpus):
        """Update existing model (if any) on corpus."""
        if not os.path.exists(self._model_filename):
            return self.train(corpus)

        LOG.debug('Updating exiting model from corpus')

        # reset any existing topics generated
        self._topics = None

        corpus_size = write_corpus_as_vw(corpus, self._corpus_filename)

        cmd = self._get_vw_update_command(corpus_size)

        _run_vw_command(cmd)

        # ensure that future updates of this model use correct offset
        self.offset += corpus_size

    def log_perplexity(self, chunk):
        """Return per-word lower bound on log perplexity.

        Also logs this and perplexity at INFO level.
        """
        vw_data = self._predict(chunk)[1]
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        bound = -vw_data['average_loss']
        LOG.info("%.3f per-word bound, %.1f perplexity estimate based on a "
                 "held-out corpus of %i documents with %i words",
                 bound,
                 numpy.exp2(-bound),
                 vw_data['corpus_size'],
                 corpus_words)
        return bound

    def get_topics(self):
        """
        Returns:
            np.ndarray: `num_topics` x `vocabulary_size` array of floats which represents
            the term topic matrix learned during inference.
        """
        topics = self._get_topics()
        return topics / topics.sum(axis=1)[:, None]

    def print_topics(self, num_topics=10, num_words=10):
        return self.show_topics(num_topics, num_words, log=True)

    def show_topics(self, num_topics=10, num_words=10,
                    log=False, formatted=True):
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
        else:
            num_topics = min(num_topics, self.num_topics)

        chosen_topics = range(num_topics)
        shown = []

        for i in chosen_topics:
            if formatted:
                topic = self.print_topic(i, topn=num_words)
            else:
                topic = self.show_topic(i, topn=num_words)

            shown.append(topic)

            if log:
                LOG.info("topic #%i (%.3f): %s", i, self.alpha, topic)

        return shown

    def print_topic(self, topicid, topn=10):
        return ' + '.join(['{0:.3f}*{1}'.format(v[0], v[1])
                           for v in self.show_topic(topicid, topn)])

    def show_topic(self, topicid, topn=10):
        topics = self._get_topics()
        topic = topics[topicid]
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(topic[t_id], self.id2word[t_id]) for t_id in bestn]

    def save(self, fname, *args, **kwargs):
        """Serialise this model to file with given name."""
        if os.path.exists(self._model_filename):
            # Vowpal Wabbit uses its own binary model file, read this into
            # variable before serialising this object - keeps all data
            # self contained within a single serialised file
            LOG.debug("Reading model bytes from '%s'", self._model_filename)
            with utils.smart_open(self._model_filename, 'rb') as fhandle:
                self._model_data = fhandle.read()

        if os.path.exists(self._topics_filename):
            LOG.debug("Reading topic bytes from '%s'", self._topics_filename)
            with utils.smart_open(self._topics_filename, 'rb') as fhandle:
                self._topics_data = fhandle.read()

        if 'ignore' not in kwargs:
            kwargs['ignore'] = frozenset(['_topics', 'tmp_dir'])

        super(LdaVowpalWabbit, self).save(fname, *args, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load LDA model from file with given name."""
        lda_vw = super(LdaVowpalWabbit, cls).load(fname, *args, **kwargs)
        lda_vw._init_temp_dir(prefix=lda_vw.tmp_prefix)

        if lda_vw._model_data:
            # Vowpal Wabbit operates on its own binary model file - deserialise
            # to file at load time, making it immediately ready for use
            LOG.debug("Writing model bytes to '%s'", lda_vw._model_filename)
            with utils.smart_open(lda_vw._model_filename, 'wb') as fhandle:
                fhandle.write(lda_vw._model_data)
            lda_vw._model_data = None # no need to keep in memory after this

        if lda_vw._topics_data:
            LOG.debug("Writing topic bytes to '%s'", lda_vw._topics_filename)
            with utils.smart_open(lda_vw._topics_filename, 'wb') as fhandle:
                fhandle.write(lda_vw._topics_data)
            lda_vw._topics_data = None

        return lda_vw

    def __del__(self):
        """Cleanup the temporary directory used by this wrapper."""
        if self.cleanup_files and self.tmp_dir:
            LOG.debug("Recursively deleting: %s", self.tmp_dir)
            shutil.rmtree(self.tmp_dir)

    def _init_temp_dir(self, prefix='tmp'):
        """Create a working temporary directory with given prefix."""
        self.tmp_dir = tempfile.mkdtemp(prefix=prefix)
        LOG.info('using %s as temp dir', self.tmp_dir)

    def _get_vw_predict_command(self, corpus_size):
        """Get list of command line arguments for running prediction."""
        cmd = [self.vw_path,
               '--testonly', # don't update model with this data
               '--lda_D', str(corpus_size),
               '-i', self._model_filename, # load existing binary model
               '-d', self._corpus_filename,
               '--learning_rate', '0', # possibly not needed, but harmless
               '-p', self._predict_filename]

        if self.random_seed is not None:
            cmd.extend(['--random_seed', str(self.random_seed)])

        return cmd

    def _get_vw_train_command(self, corpus_size, update=False):
        """Get list of command line arguments for running model training.

        If 'update' is set to True, this specifies that we're further training
        an existing model.
        """
        cmd = [self.vw_path,
               '-d', self._corpus_filename,
               '--power_t', str(self.decay),
               '--initial_t', str(self.offset),
               '--minibatch', str(self.chunksize),
               '--lda_D', str(corpus_size),
               '--passes', str(self.passes),
               '--cache_file', self._cache_filename,
               '--lda_epsilon', str(self.gamma_threshold),
               '--readable_model', self._topics_filename,
               '-k', # clear cache
               '-f', self._model_filename]

        if update:
            cmd.extend(['-i', self._model_filename])
        else:
            # these params are read from model file if updating
            cmd.extend(['--lda', str(self.num_topics),
                        '-b', str(_bit_length(self.num_terms)),
                        '--lda_alpha', str(self.alpha),
                        '--lda_rho', str(self.eta)])

        if self.random_seed is not None:
            cmd.extend(['--random_seed', str(self.random_seed)])

        return cmd

    def _get_vw_update_command(self, corpus_size):
        """Get list of command line arguments to update a model."""
        return self._get_vw_train_command(corpus_size, update=True)

    def _load_vw_topics(self):
        """Read topics file generated by Vowpal Wabbit, convert to numpy array.

        Output consists of many header lines, followed by a number of lines
        of:
        <word_id> <topic_1_gamma> <topic_2_gamma> ...
        """
        topics = numpy.zeros((self.num_topics, self.num_terms),
                             dtype=numpy.float32)

        with utils.smart_open(self._topics_filename) as topics_file:
            found_data = False

            for line in topics_file:
                # look for start of data
                if not found_data:
                    if line.startswith(b'0 ') and b':' not in line:
                        found_data = True
                    else:
                        continue

                fields = line.split()
                word_id = int(fields[0])

                # output contains entries for 2**b terms, where b was set
                # by the '-b' option, ignore anything past num_terms
                if word_id >= self.num_terms:
                    break

                topics[:, word_id] = fields[1:]

        # normalise to probability distribution
        self._topics = topics / topics.sum(axis=1, keepdims=True)

    def _get_topics(self):
        """Get topics matrix, load from file if necessary."""
        if self._topics is None:
            self._load_vw_topics()
        return self._topics

    def _predict(self, chunk):
        """Run given chunk of documents against currently trained model.

        Returns a tuple of prediction matrix and Vowpal Wabbit data.
        """
        corpus_size = write_corpus_as_vw(chunk, self._corpus_filename)

        cmd = self._get_vw_predict_command(corpus_size)
        vw_data = _parse_vw_output(_run_vw_command(cmd))
        vw_data['corpus_size'] = corpus_size

        predictions = numpy.zeros((corpus_size, self.num_topics),
                                  dtype=numpy.float32)

        with utils.smart_open(self._predict_filename) as fhandle:
            for i, line in enumerate(fhandle):
                predictions[i, :] = line.split()

        predictions = predictions / predictions.sum(axis=1, keepdims=True)

        return predictions, vw_data

    def __getitem__(self, bow, eps=0.01):
        is_corpus, dummy_corpus = utils.is_corpus(bow)
        if not is_corpus:
            bow = [bow]

        predictions = self._predict(bow)[0]

        topics = []
        for row in predictions:
            row_topics = []
            for topic_id, val in enumerate(row):
                if val > eps:
                    row_topics.append((topic_id, val))
            topics.append(row_topics)

        return topics if is_corpus else topics[0]

    def _get_filename(self, name):
        """Get path to given filename in temp directory."""
        return os.path.join(self.tmp_dir, name)

    @property
    def _model_filename(self):
        """Get path to file to write Vowpal Wabbit model to."""
        return self._get_filename('model.vw')

    @property
    def _cache_filename(self):
        """Get path to file to write Vowpal Wabbit cache to."""
        return self._get_filename('cache.vw')

    @property
    def _corpus_filename(self):
        """Get path to file to write Vowpal Wabbit corpus to."""
        return self._get_filename('corpus.vw')

    @property
    def _topics_filename(self):
        """Get path to file to write Vowpal Wabbit topics to."""
        return self._get_filename('topics.vw')

    @property
    def _predict_filename(self):
        """Get path to file to write Vowpal Wabbit predictions to."""
        return self._get_filename('predict.vw')

    def __str__(self):
        fields = ['num_terms', 'num_topics', 'chunksize', 'alpha', 'eta']
        kv = ["{0}={1}".format(field, getattr(self, field)) for field in fields]
        return "{0}({1})".format(self.__class__.__name__, ', '.join(kv))


def corpus_to_vw(corpus):
    """Iterate over corpus, yielding lines in Vowpal Wabbit format.

    For LDA, this consists of each document on a single line consisting of
    space separated lists of <word_id>:<count>. Each line starts with a '|'
    character.

    E.g.:
    | 4:7 14:1 22:8 6:3
    | 14:22 22:4 0:1 1:3
    | 7:2 8:2
    """
    for entries in corpus:
        line = ['|']
        for word_id, count in entries:
            line.append("{0}:{1}".format(word_id, count))
        yield ' '.join(line)


def write_corpus_as_vw(corpus, filename):
    """Iterate over corpus, writing each document as a line to given file.

    Returns the number of lines written.
    """
    LOG.debug("Writing corpus to: %s", filename)

    corpus_size = 0
    with utils.smart_open(filename, 'wb') as corpus_file:
        for line in corpus_to_vw(corpus):
            corpus_file.write(line.encode('utf-8') + b'\n')
            corpus_size += 1

    return corpus_size


def _parse_vw_output(text):
    """Get dict of useful fields from Vowpal Wabbit's output.

    Currently returns field 'average_loss', which is a lower bound on mean
    per-word log-perplexity (i.e. same as the value LdaModel.bound() returns).
    """
    data = {}
    for line in text.splitlines():
        if line.startswith('average loss'):
            data['average_loss'] = float(line.split('=')[1])
            break

    return data


def _run_vw_command(cmd):
    """Execute given Vowpal Wabbit command, log stdout and stderr."""
    LOG.info("Running Vowpal Wabbit command: %s", ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    output = proc.communicate()[0].decode('utf-8')
    LOG.debug("Vowpal Wabbit output: %s", output)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode,
                                            ' '.join(cmd),
                                            output=output)

    return output


# if python2.6 support is ever dropped, can change to using int.bit_length()
def _bit_length(num):
    """Return number of bits needed to encode given number."""
    return len(bin(num).lstrip('-0b'))

def vwmodel2ldamodel(vw_model, iterations=50):
    """
    Function to convert vowpal wabbit model to gensim LdaModel. This works by
    simply copying the training model weights (alpha, beta...) from a trained
    vwmodel into the gensim model.

    Args:
    ----
    vw_model : Trained vowpal wabbit model.
    iterations : Number of iterations to be used for inference of the new LdaModel.

    Returns:
    -------
    model_gensim : LdaModel instance; copied gensim LdaModel.
    """
    model_gensim = LdaModel(
        num_topics=vw_model.num_topics, id2word=vw_model.id2word, chunksize=vw_model.chunksize,
        passes=vw_model.passes, alpha=vw_model.alpha, eta=vw_model.eta, decay=vw_model.decay,
        offset=vw_model.offset, iterations=iterations, gamma_threshold=vw_model.gamma_threshold)
    model_gensim.expElogbeta[:] = vw_model._get_topics()
    return model_gensim
