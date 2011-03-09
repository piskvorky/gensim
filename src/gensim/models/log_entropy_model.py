#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import math
from gensim import interfaces, matutils, utils


class LogEntropyModel(interfaces.TransformationABC):
    """
    Objects of this class realize the transformation between word-document
    co-occurence matrix (integers) into a locally/globally weighted matrix
    (positive floats).

    This is done by a log entropy normalization, optionally normalizing the
    resulting documents to unit length. The following formulas explain how
    to compute the log entropy normalization for a term i
    local_weight_i = log(doc_frequency_of_i + 1)
    P_i = doc_frequency_of_i / global_frequency_of_i

                          sum_over_docs_{j} P_i,j * log(P_i,j)
    global_weight_i = 1 + ------------------------------------
                                  log(n_contexts_i)

    the normalized value for a term i is: local_weight_i * global_weight_i

    The main methods are:

    1. constructor, which calculates the global weighting for all terms in
        a corpus.
    2. the [] method, which transforms a simple count representation into the
        log entropy normalized space.

    >>> log_ent = LogEntropyModel(corpus)
    >>> print = log_ent[some_doc]
    >>> log_ent.save('/tmp/foo.log_ent_model')

    Model persistency is achieved via its load/save methods.
    """

    def __init__(self, corpus, id2word=None, normalize=True):
        """
        `normalize` dictates whether the resulting vectors will be
        set to unit length.
        """
        self.normalize = normalize
        self.n_docs = 0
        self.n_words = 0
        self.entr = {}
        if corpus is not None:
            self.initialize(corpus)

    def __str__(self):
        return "LogEntropyModel(n_docs=%s, n_words=%s)" % (self.n_docs,
                self.n_words)

    def initialize(self, corpus):
        """
        initialization of context_count, global frequency, etc.
        """
        logging.info("calculating counts")
        glob_freq = {}
        n_context = {}
        glob_num_words = 0
        for doc_no, bow in enumerate(corpus):
            if doc_no % 10000 == 0:
                logging.info("PROGRESS: processing document #%i" % doc_no)
            glob_num_words += len(bow)
            for term_id, term_count in bow:
                n_context[term_id] = n_context.get(term_id, 0) + 1
                glob_freq[term_id] = glob_freq.get(term_id, 0) + term_count
        once = [key for key, val in n_context.items() if val < 2]
        if len(once) > 0:
            logging.error("There are words that appear only in one document")
            logging.error("this words or documents should be filtered out")
            logging.error("this cause problems in the entropy normalization")
            raise ValueError("invalid context diversity of corpus")

        # keep some stats about the training corpus
        self.n_docs = doc_no + 1
        self.n_words = glob_num_words

        # and finally compute the global weights
        logging.info(("calculating global log entropy weights for %i "
                     + "documents and %i features (%i matrix non-zeros)")
                     % (self.n_docs,
                        1 + max([-1] + n_context.keys()),
                        self.n_words))

        for key, freq_glob in glob_freq.iteritems():
            sm = 0
            for bow in corpus:
                dict_bow = dict(bow)
                if dict_bow.get(key):
                    sm += ((dict_bow.get(key) / freq_glob)
                         * math.log(dict_bow.get(key) / freq_glob))
            self.entr[key] = 1 + sm / math.log(n_context[key])

    def __getitem__(self, bow):
        """
        Return log entropy representation of the input vector and/or corpus.
        """
        # if the input vector is in fact a corpus, return a transformed corpus
        is_corpus, bow = utils.isCorpus(bow)
        if is_corpus:
            return self._apply(bow)

        # unknown (new) terms will be given zero weight (NOT infinity/huge)
        vector = [(term_id, math.log(tf + 1) * self.entr.get(term_id))
                  for term_id, tf in bow if self.entr.get(term_id)]
        if self.normalize:
            vector = matutils.unitVec(vector)
        return vector
