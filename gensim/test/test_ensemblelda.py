#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Tobias B <proxima@sezanzeb.de>

"""
Automated tests for checking the EnsembleLda Class
"""

import os
import logging
import unittest

import numpy as np
from copy import deepcopy

import pytest

from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary

NUM_TOPICS = 2
NUM_MODELS = 4
PASSES = 50

# windows tests fail due to the required assertion precision being too high
RTOL = 1e-04 if os.name == 'nt' else 1e-05


class TestModel(unittest.TestCase):
    def setUp(self):
        # same configuration for each model to make sure
        # the topics are equal
        random_state = 0

        self.elda = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS,
            passes=PASSES, num_models=NUM_MODELS, random_state=random_state,
            topic_model_class=LdaModel,
        )

        self.elda_mem_unfriendly = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS,
            passes=PASSES, num_models=NUM_MODELS, random_state=random_state,
            memory_friendly_ttda=False, topic_model_class=LdaModel,
        )

    def assert_ttda_is_valid(self, ensemble):
        """Check that ttda has one or more topic and that term probabilities add to one."""
        assert len(ensemble.ttda) > 0
        sum_over_terms = ensemble.ttda.sum(axis=1)
        expected_sum_over_terms = np.ones(len(ensemble.ttda)).astype(np.float32)
        np.testing.assert_allclose(sum_over_terms, expected_sum_over_terms, rtol=1e-04)

    def test_elda(self):
        # given that the random_state doesn't change, it should
        # always be 2 detected topics in this setup.
        assert self.elda.stable_topics.shape[1] == len(common_dictionary)
        assert len(self.elda.ttda) == NUM_MODELS * NUM_TOPICS
        self.assert_ttda_is_valid(self.elda)

        # reclustering shouldn't change anything without
        # added models or different parameters
        self.elda.recluster()

        assert self.elda.stable_topics.shape[1] == len(common_dictionary)
        assert len(self.elda.ttda) == NUM_MODELS * NUM_TOPICS
        self.assert_ttda_is_valid(self.elda)

        # compare with a pre-trained reference model
        reference = EnsembleLda.load(datapath('ensemblelda'))
        np.testing.assert_allclose(self.elda.ttda, reference.ttda, rtol=RTOL)
        # small values in the distance matrix tend to vary quite a bit around 2%,
        # so use some absolute tolerance measurement to check if the matrix is at least
        # close to the target.
        atol = reference.asymmetric_distance_matrix.max() * 1e-05
        np.testing.assert_allclose(
            self.elda.asymmetric_distance_matrix,
            reference.asymmetric_distance_matrix, atol=atol,
        )

    def test_clustering(self):
        # the following test is quite specific to the current implementation and not part of any api,
        # but it makes improving those sections of the code easier as long as sorted_clusters and the
        # cluster_model results are supposed to stay the same. Potentially this test will deprecate.

        reference = EnsembleLda.load(datapath('ensemblelda'))
        cluster_model_results = deepcopy(reference.cluster_model.results)
        valid_clusters = deepcopy(reference.valid_clusters)
        stable_topics = deepcopy(reference.get_topics())

        # continue training with the distance matrix of the pretrained reference and see if
        # the generated clusters match.
        reference.asymmetric_distance_matrix_outdated = True
        reference.recluster()

        self.assert_cluster_results_equal(reference.cluster_model.results, cluster_model_results)
        assert reference.valid_clusters == valid_clusters
        np.testing.assert_allclose(reference.get_topics(), stable_topics, rtol=RTOL)

    def test_not_trained(self):
        # should not throw errors and no training should happen

        # 0 passes
        elda = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS,
            passes=0, num_models=NUM_MODELS, random_state=0,
        )
        assert len(elda.ttda) == 0

        # no corpus
        elda = EnsembleLda(
            id2word=common_dictionary, num_topics=NUM_TOPICS,
            passes=PASSES, num_models=NUM_MODELS, random_state=0,
        )
        assert len(elda.ttda) == 0

        # 0 iterations
        elda = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS,
            iterations=0, num_models=NUM_MODELS, random_state=0,
        )
        assert len(elda.ttda) == 0

        # 0 models
        elda = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS,
            passes=PASSES, num_models=0, random_state=0
        )
        assert len(elda.ttda) == 0

    def test_mem_unfriendly(self):
        # at this point, self.elda_mem_unfriendly and self.eLDA are already trained
        # in the setUpClass function
        # both should be 100% similar (but floats cannot
        # be compared that easily, so check for threshold)
        assert len(self.elda_mem_unfriendly.tms) == NUM_MODELS
        np.testing.assert_allclose(self.elda.ttda, self.elda_mem_unfriendly.ttda, rtol=RTOL)
        np.testing.assert_allclose(self.elda.get_topics(), self.elda_mem_unfriendly.get_topics(), rtol=RTOL)
        self.assert_ttda_is_valid(self.elda_mem_unfriendly)

    def test_generate_gensim_rep(self):
        gensim_model = self.elda.generate_gensim_representation()
        topics = gensim_model.get_topics()
        np.testing.assert_allclose(self.elda.get_topics(), topics, rtol=RTOL)

    def assert_cluster_results_equal(self, a, b):
        """compares important attributes of the cluster results"""
        np.testing.assert_array_equal(
            [row["label"] for row in a],
            [row["label"] for row in b],
        )
        np.testing.assert_array_equal(
            [row["is_core"] for row in a],
            [row["is_core"] for row in b],
        )

    def test_persisting(self):
        fname = get_tmpfile('gensim_models_ensemblelda')
        self.elda.save(fname)
        loaded_elda = EnsembleLda.load(fname)
        # storing the ensemble without memory_friendy_ttda
        self.elda_mem_unfriendly.save(fname)
        loaded_elda_mu = EnsembleLda.load(fname)

        # topic_model_class will be lazy loaded and should be None first
        assert loaded_elda.topic_model_class is None

        # was it stored and loaded correctly?
        # memory friendly.
        loaded_elda_representation = loaded_elda.generate_gensim_representation()

        # generating the representation also lazily loads the topic_model_class
        assert loaded_elda.topic_model_class == LdaModel

        topics = loaded_elda_representation.get_topics()
        ttda = loaded_elda.ttda
        amatrix = loaded_elda.asymmetric_distance_matrix
        np.testing.assert_allclose(self.elda.get_topics(), topics, rtol=RTOL)
        np.testing.assert_allclose(self.elda.ttda, ttda, rtol=RTOL)
        np.testing.assert_allclose(self.elda.asymmetric_distance_matrix, amatrix, rtol=RTOL)

        a = self.elda.cluster_model.results
        b = loaded_elda.cluster_model.results

        self.assert_cluster_results_equal(a, b)

        # memory unfriendly
        loaded_elda_mem_unfriendly_representation = loaded_elda_mu.generate_gensim_representation()
        topics = loaded_elda_mem_unfriendly_representation.get_topics()
        np.testing.assert_allclose(self.elda.get_topics(), topics, rtol=RTOL)

    def test_multiprocessing(self):
        # same configuration
        random_state = 0

        # use 3 processes for the ensemble and the distance,
        # so that the 4 models and 8 topics cannot be distributed
        # to each worker evenly
        workers = 3

        # memory friendly. contains List of topic word distributions
        elda_multiprocessing = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, topic_model_class=LdaModel,
            num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS,
            random_state=random_state, ensemble_workers=workers, distance_workers=workers,
        )

        # memory unfriendly. contains List of models
        elda_multiprocessing_mem_unfriendly = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary, topic_model_class=LdaModel,
            num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS,
            random_state=random_state, ensemble_workers=workers, distance_workers=workers,
            memory_friendly_ttda=False,
        )

        np.testing.assert_allclose(self.elda.get_topics(), elda_multiprocessing.get_topics(), rtol=RTOL)
        np.testing.assert_allclose(self.elda_mem_unfriendly.get_topics(), elda_multiprocessing_mem_unfriendly.get_topics(), rtol=RTOL)

    def test_add_models_to_empty(self):
        ensemble = EnsembleLda(id2word=common_dictionary, num_models=0)
        ensemble.add_model(self.elda.ttda[0:1])
        ensemble.add_model(self.elda.ttda[1:])
        ensemble.recluster()
        np.testing.assert_allclose(ensemble.get_topics(), self.elda.get_topics(), rtol=RTOL)

        # persisting an ensemble that is entirely built from existing ttdas
        fname = get_tmpfile('gensim_models_ensemblelda')
        ensemble.save(fname)
        loaded_ensemble = EnsembleLda.load(fname)
        np.testing.assert_allclose(loaded_ensemble.get_topics(), self.elda.get_topics(), rtol=RTOL)
        self.test_inference(loaded_ensemble)

    def test_add_models(self):
        # same configuration
        num_models = self.elda.num_models

        # make sure countings and sizes after adding are correct
        # create new models and add other models to them.

        # there are a ton of configurations for the first parameter possible,
        # try them all

        # quickly train something that can be used for counting results
        num_new_models = 3
        num_new_topics = 3

        # 1. memory friendly
        elda_base = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary,
            num_topics=num_new_topics, passes=1, num_models=num_new_models,
            iterations=1, random_state=0, topic_model_class=LdaMulticore,
            workers=3, ensemble_workers=2,
        )

        # 1.1 ttda
        a = len(elda_base.ttda)
        b = elda_base.num_models
        elda_base.add_model(self.elda.ttda)
        assert len(elda_base.ttda) == a + len(self.elda.ttda)
        assert elda_base.num_models == b + 1  # defaults to 1 for one ttda matrix

        # 1.2 an ensemble
        a = len(elda_base.ttda)
        b = elda_base.num_models
        elda_base.add_model(self.elda, 5)
        assert len(elda_base.ttda) == a + len(self.elda.ttda)
        assert elda_base.num_models == b + 5

        # 1.3 a list of ensembles
        a = len(elda_base.ttda)
        b = elda_base.num_models
        # it should be totally legit to add a memory unfriendly object to a memory friendly one
        elda_base.add_model([self.elda, self.elda_mem_unfriendly])
        assert len(elda_base.ttda) == a + 2 * len(self.elda.ttda)
        assert elda_base.num_models == b + 2 * num_models

        # 1.4 a single gensim model
        model = self.elda.classic_model_representation

        a = len(elda_base.ttda)
        b = elda_base.num_models
        elda_base.add_model(model)
        assert len(elda_base.ttda) == a + len(model.get_topics())
        assert elda_base.num_models == b + 1

        # 1.5 a list gensim models
        a = len(elda_base.ttda)
        b = elda_base.num_models
        elda_base.add_model([model, model])
        assert len(elda_base.ttda) == a + 2 * len(model.get_topics())
        assert elda_base.num_models == b + 2

        self.assert_ttda_is_valid(elda_base)

        # 2. memory unfriendly
        elda_base_mem_unfriendly = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary,
            num_topics=num_new_topics, passes=1, num_models=num_new_models,
            iterations=1, random_state=0, topic_model_class=LdaMulticore,
            workers=3, ensemble_workers=2, memory_friendly_ttda=False,
        )

        # 2.1 a single ensemble
        a = len(elda_base_mem_unfriendly.tms)
        b = elda_base_mem_unfriendly.num_models
        elda_base_mem_unfriendly.add_model(self.elda_mem_unfriendly)
        assert len(elda_base_mem_unfriendly.tms) == a + num_models
        assert elda_base_mem_unfriendly.num_models == b + num_models

        # 2.2 a list of ensembles
        a = len(elda_base_mem_unfriendly.tms)
        b = elda_base_mem_unfriendly.num_models
        elda_base_mem_unfriendly.add_model([self.elda_mem_unfriendly, self.elda_mem_unfriendly])
        assert len(elda_base_mem_unfriendly.tms) == a + 2 * num_models
        assert elda_base_mem_unfriendly.num_models == b + 2 * num_models

        # 2.3 a single gensim model
        a = len(elda_base_mem_unfriendly.tms)
        b = elda_base_mem_unfriendly.num_models
        elda_base_mem_unfriendly.add_model(self.elda_mem_unfriendly.tms[0])
        assert len(elda_base_mem_unfriendly.tms) == a + 1
        assert elda_base_mem_unfriendly.num_models == b + 1

        # 2.4 a list of gensim models
        a = len(elda_base_mem_unfriendly.tms)
        b = elda_base_mem_unfriendly.num_models
        elda_base_mem_unfriendly.add_model(self.elda_mem_unfriendly.tms)
        assert len(elda_base_mem_unfriendly.tms) == a + num_models
        assert elda_base_mem_unfriendly.num_models == b + num_models

        # 2.5 topic term distributions should throw errors, because the
        # actual models are needed for the memory unfriendly ensemble
        a = len(elda_base_mem_unfriendly.tms)
        b = elda_base_mem_unfriendly.num_models
        with pytest.raises(ValueError):
            elda_base_mem_unfriendly.add_model(self.elda_mem_unfriendly.tms[0].get_topics())
        # remains unchanged
        assert len(elda_base_mem_unfriendly.tms) == a
        assert elda_base_mem_unfriendly.num_models == b

        assert elda_base_mem_unfriendly.num_models == len(elda_base_mem_unfriendly.tms)
        self.assert_ttda_is_valid(elda_base_mem_unfriendly)

    def test_add_and_recluster(self):
        # See if after adding a model, the model still makes sense
        num_new_models = 3
        num_new_topics = 3

        # train
        elda = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary,
            num_topics=num_new_topics, passes=10, num_models=num_new_models,
            iterations=30, random_state=1, topic_model_class='lda',
            distance_workers=4,
        )
        elda_mem_unfriendly = EnsembleLda(
            corpus=common_corpus, id2word=common_dictionary,
            num_topics=num_new_topics, passes=10, num_models=num_new_models,
            iterations=30, random_state=1, topic_model_class=LdaModel,
            distance_workers=4, memory_friendly_ttda=False,
        )
        # both should be similar
        np.testing.assert_allclose(elda.ttda, elda_mem_unfriendly.ttda, rtol=RTOL)
        np.testing.assert_allclose(elda.get_topics(), elda_mem_unfriendly.get_topics(), rtol=RTOL)
        # and every next step applied to both should result in similar results

        # 1. adding to ttda and tms
        elda.add_model(self.elda)
        elda_mem_unfriendly.add_model(self.elda_mem_unfriendly)
        np.testing.assert_allclose(elda.ttda, elda_mem_unfriendly.ttda, rtol=RTOL)
        assert len(elda.ttda) == len(self.elda.ttda) + num_new_models * num_new_topics
        assert len(elda_mem_unfriendly.ttda) == len(self.elda_mem_unfriendly.ttda) + num_new_models * num_new_topics
        assert len(elda_mem_unfriendly.tms) == NUM_MODELS + num_new_models
        self.assert_ttda_is_valid(elda)
        self.assert_ttda_is_valid(elda_mem_unfriendly)

        # 2. distance matrix
        elda._generate_asymmetric_distance_matrix()
        elda_mem_unfriendly._generate_asymmetric_distance_matrix()
        np.testing.assert_allclose(
            elda.asymmetric_distance_matrix,
            elda_mem_unfriendly.asymmetric_distance_matrix,
        )

        # 3. CBDBSCAN results
        elda._generate_topic_clusters()
        elda_mem_unfriendly._generate_topic_clusters()
        a = elda.cluster_model.results
        b = elda_mem_unfriendly.cluster_model.results
        self.assert_cluster_results_equal(a, b)

        # 4. finally, the stable topics
        elda._generate_stable_topics()
        elda_mem_unfriendly._generate_stable_topics()
        np.testing.assert_allclose(
            elda.get_topics(),
            elda_mem_unfriendly.get_topics(),
        )

        elda.generate_gensim_representation()
        elda_mem_unfriendly.generate_gensim_representation()

        # same random state, hence topics should be still similar
        np.testing.assert_allclose(elda.get_topics(), elda_mem_unfriendly.get_topics(), rtol=RTOL)

    def test_inference(self, elda=None):
        if elda is None:
            elda = self.elda

        # get the most likely token id from topic 0
        max_id = np.argmax(elda.get_topics()[0, :])
        assert elda.classic_model_representation.iterations > 0
        # topic 0 should be dominant in the inference.
        # the difference between the probabilities should be significant and larger than 0.3
        inferred = elda[[(max_id, 1)]]
        assert inferred[0][1] - 0.3 > inferred[1][1]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
    unittest.main()
