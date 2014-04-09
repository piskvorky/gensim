#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python wrapper for Latent Dirichlet Allocation (LDA) from MALLET, the Java topic modelling
toolkit [1]_.

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents, using an (optimized version of) collapsed
gibbs sampling from MALLET.

MALLET's LDA training requires O(#corpus_words) of memory, keeping the entire corpus in RAM.
If you find yourself running out of memory, either decrease the `workers` constructor
parameter, or use `LdaModel` which needs only O(1) memory.

The wrapped model can NOT be updated with new documents for online training -- use gensim's `LdaModel` for that.

Example:

>>> model = gensim.models.LdaMallet('/Users/kofola/mallet-2.0.7/bin/mallet', corpus=my_corpus, num_topics=20, id2word=dictionary)
>>> print model[my_vector]  # print LDA topics of a document

.. [1] http://mallet.cs.umass.edu/

"""


import logging
import random
import tempfile
import os
from subprocess import call

logger = logging.getLogger('gensim.models.ldamallet')

from gensim import utils


def read_doctopics(fname, eps=1e-6):
    """
    Yield document topic vectors from MALLET's "doc-topics" format, as sparse gensim vectors.

    """
    with utils.smart_open(fname) as fin:
        fin.next()  # skip the header line
        for lineno, line in enumerate(fin):
            parts = line.split()[2:]  # skip "doc" and "source" columns
            if len(parts) % 2 != 0:
                raise RuntimeError("invalid doc topics format at line %i in %s" % (lineno + 1, fname))
            doc = [(int(id), float(weight)) for id, weight in zip(parts[::2], parts[1::2]) if abs(float(weight)) > eps]
            # explicitly normalize probs to sum up to 1.0, just to be sure...
            weights = float(sum([weight for _, weight in doc]))
            yield [] if weights == 0 else sorted((id, 1.0 * weight / weights) for id, weight in doc)



class LdaMallet(utils.SaveLoad):
    """
    Class for LDA training using MALLET. Communication between MALLET and Python
    takes place by passing around data files on disk and calling Java with subprocess.call().

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, id2word=None, workers=4, prefix=None,
                 optimize_interval=0, iterations=1000):
        """
        `mallet_path` is path to the mallet executable, e.g. `/home/kofola/mallet-2.0.7/bin/mallet`.
        `corpus` is a gensim corpus, aka a stream of sparse document vectors.
        `id2word` is a mapping between tokens ids and token.
        `workers` is the number of threads, for parallel training.
        `prefix` is the string prefix under which all data files will be stored; default: system temp + random filename prefix.
        `optimize_interval` optimize hyperparameters every N iterations (sometimes leads to Java exception; 0 to switch off hyperparameter optimization).
        `iterations` is the number of sampling iterations.

        """
        self.mallet_path = mallet_path
        self.id2word = id2word
        self.num_topics = num_topics
        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)
        self.prefix = prefix
        self.workers = workers
        self.optimize_interval = optimize_interval
        self.iterations = iterations

        if corpus is not None:
            self.train(corpus)

    def finferencer(self):
        return self.prefix + 'inferencer.mallet'

    def ftopickeys(self):
        return self.prefix + 'topickeys.txt'

    def fstate(self):
        return self.prefix + 'state.mallet'

    def fdoctopics(self):
        return self.prefix + 'doctopics.txt'

    def fcorpustxt(self):
        return self.prefix + 'corpus.txt'

    def fcorpusmallet(self):
        return self.prefix + 'corpus.mallet'

    def fwordweights(self):
        return self.prefix + 'wordweights.txt'

    def convert_input(self, corpus, infer=False):
        """
        Serialize documents (lists of unicode tokens) to a temporary text file,
        then convert that text file to MALLET format `outfile`.

        """
        logger.info("serializing temporary corpus to %s" % self.fcorpustxt())
        # write out the corpus in a file format that MALLET understands: one document per line:
        # document id[SPACE]label (not used)[SPACE]utf8-encoded tokens, whitespace delimited
        with utils.smart_open(self.fcorpustxt(), 'wb') as fout:
            for docno, doc in enumerate(corpus):
                if self.id2word:
                    tokens = sum(([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc), [])
                else:
                    tokens = sum(([str(tokenid)] * int(cnt) for tokenid, cnt in doc), [])
                fout.write("%s 0 %s\n" % (docno, utils.to_utf8(' '.join(tokens))))

        # convert the text file above into MALLET's internal format
        cmd = self.mallet_path + " import-file --keep-sequence --remove-stopwords --token-regex '\S+' --input %s --output %s"
        if infer:
            cmd += ' --use-pipe-from ' + self.fcorpusmallet()
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet() + '.infer')
        else:
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet())
        logger.info("converting temporary corpus to MALLET format with %s" % cmd)
        call(cmd, shell=True)


    def train(self, corpus):
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + " train-topics --input %s --num-topics %s --optimize-interval %s "\
            "--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s "\
            "--num-iterations %s --inferencer-filename %s"
        cmd = cmd % (self.fcorpusmallet(), self.num_topics, self.optimize_interval, self.workers,
            self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations, self.finferencer())
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s" % cmd)
        call(cmd, shell=True)


    def __getitem__(self, bow, iterations=100):
        is_corpus, corpus = utils.is_corpus(bow)
        if not is_corpus:
            bow = [bow]

        self.convert_input(bow, infer=True)
        cmd = self.mallet_path + " infer-topics --input %s --inferencer %s --output-doc-topics %s --num-iterations %s"
        cmd = cmd % (self.fcorpusmallet() + '.infer', self.finferencer(), self.fdoctopics() + '.infer', iterations)
        logger.info("inferring with MALLET LDA with %s" % cmd)
        call(cmd, shell=True)
        return list(read_doctopics(self.fdoctopics() + '.infer'))
