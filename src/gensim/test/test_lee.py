#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated test to reproduce the results of Lee et al. (2005)

Lee et al. (2005) compares different models for semantic
similarity and verifies the results with similarity judgements from humans.

As a validation of the gensim implementation we reproduced the results
of  Lee et al. (2005) in this test.

Many thanks to Michael D. Lee (michael.lee@adelaide.edu.au) who provideded us
with his corpus and similarity data.

If you need to reference this dataset, please cite:

Lee, M., Pincombe, B., & Welsh, M. (2005).
An empirical evaluation of models of text document similarity.
Proceedings of the 27th Annual Conference of the Cognitive Science Society
"""

import os
from gensim import corpora
from gensim import models
from gensim import matutils
from gensim.parsing.preprocessing import preprocess_documents
import numpy as np
from nose import tools


bg_corpus = None
corpus = None
human_sim_vector = None


def setup_module():
    """
    fixture function
    all tests are run after (setup_module) which is only executed once
    """
    global bg_corpus, corpus, human_sim_vector

    pre_path = os.path.dirname(__file__) + os.sep + 'test_data' + os.sep
    bg_corpus_file = 'lee_background.cor'
    corpus_file = 'lee.cor'
    sim_file = 'similarities0-1.txt'

    # read in the corpora
    with open(pre_path + bg_corpus_file, 'r') as f:
        bg_corpus = preprocess_documents(f.readlines())
    with open(pre_path + corpus_file, 'r') as f:
        corpus = preprocess_documents(f.readlines())

    # read the human similarity data
    sim_matrix = np.loadtxt(pre_path + sim_file)
    sim_m_size = np.shape(sim_matrix)[0]
    human_sim_vector = sim_matrix[np.triu_indices(sim_m_size, 1)]


def test_corpus():
    """availability and integrity of corpus"""
    documents_in_bg_corpus = 300
    documents_in_corpus = 50
    len_sim_vector = 1225
    tools.assert_equal(len(bg_corpus), documents_in_bg_corpus)
    tools.assert_equal(len(corpus), documents_in_corpus)
    tools.assert_equal(len(human_sim_vector), len_sim_vector)


def test_lee():
    """
    correlation with human data > 0.6

    this is the value which was achieved in the original paper
    """

    global bg_corpus, corpus

    # create a dictionary and corpus (bag of words)
    dictionary = corpora.Dictionary(bg_corpus)
    bg_corpus = [dictionary.doc2bow(text) for text in bg_corpus]
    corpus = [dictionary.doc2bow(text) for text in corpus]

    # transform the bag of words with log_entropy normalization
    log_ent = models.LogEntropyModel(bg_corpus)
    bg_corpus_ent = log_ent[bg_corpus]

    # initialize an LSI transformation from background corpus
    lsi = models.LsiModel(bg_corpus_ent, id2word=dictionary, numTopics=200)
    # transform small corpus to lsi bow->log_ent->fold-in-lsi
    corpus_lsi = lsi[log_ent[corpus]]

    # compute pairwise similarity matrix and extract upper triangular
    res = np.zeros((len(corpus), len(corpus)))
    for i, par1 in enumerate(corpus_lsi):
        for j, par2 in enumerate(corpus_lsi):
            res[i, j] = matutils.cossim(par1, par2)
    flat = res[np.triu_indices(len(corpus), 1)]

    cor = np.corrcoef(flat, human_sim_vector)
    assert cor[0, 1] > 0.6
