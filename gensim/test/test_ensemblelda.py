#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Tobias B <github.com/sezanzeb>

# TODO remove this before pushing tests:
import sys
sys.path.insert(0, '/home/mango/Data/Code/gensim/')

import logging
import numpy as np
from gensim.models import EnsembleLda
import unittest
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary

num_topics = 2
num_models = 4
passes = 50


class TestModel(unittest.TestCase):
    def setUp(self):
        # same configuration for each model to make sure
        # the topics are equal
        random_state = 0

        self.eLDA = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_topics,
                                passes=passes, num_models=num_models, random_state=random_state)

        self.eLDA_mu = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_topics,
                                   passes=passes, num_models=num_models, random_state=random_state,
                                   memory_friendly_ttda=False)

    def check_ttda(self, ensemble):
        """tests the integrity of the ttda of any ensemble"""
        self.assertGreater(len(ensemble.ttda), 0)
        a = ensemble.ttda.sum(axis=1)
        b = np.ones(len(ensemble.ttda)).astype(np.float32)
        np.testing.assert_allclose(a, b, rtol=1e-04)

    def test_eLDA(self):
        # given that the random_state doesn't change, it should
        # always be 2 detected topics in this setup.
        self.assertEqual(self.eLDA.stable_topics.shape[1], len(common_dictionary))
        self.assertEqual(len(self.eLDA.ttda), num_models * num_topics)
        self.check_ttda(self.eLDA)

        # reclustering shouldn't change anything without
        # added models or different parameters
        self.eLDA.recluster()

        self.assertEqual(self.eLDA.stable_topics.shape[1], len(common_dictionary))
        self.assertEqual(len(self.eLDA.ttda), num_models * num_topics)
        self.check_ttda(self.eLDA)

        reference = EnsembleLda.load(datapath('ensemblelda'))

        self.assertEqual(self.eLDA.cluster_model.results, reference.cluster_model.results)
        self.assertEqual(self.eLDA.sorted_clusters, reference.sorted_clusters)
        
        np.testing.assert_allclose(self.eLDA.ttda, reference.ttda, rtol=1e-05)
        np.testing.assert_allclose(self.eLDA.get_topics(), reference.get_topics(), rtol=1e-05)

        # as small values in the distance matrix are subject to rounding differences of
        # around 1-2% between python 2 and 3, use atol and select a 100000th of the
        # largest value instead of rtol. Large values of the distance matrix are not prone
        # to those rounding problems.
        atol = reference.asymmetric_distance_matrix.max() * 1e-05
        np.testing.assert_allclose(self.eLDA.asymmetric_distance_matrix,
                                   reference.asymmetric_distance_matrix, atol=atol)

        np.testing.assert_allclose(self.eLDA.classic_model_representation.get_topics(),
                                   self.eLDA.get_topics(), rtol=1e-05)

    def test_memory_unfriendly(self):
        # at this point, self.eLDA_mu and self.eLDA are already trained
        # in the setUpClass function
        # both should be 100% similar (but floats cannot
        # be compared that easily, so check for threshold)
        self.assertEqual(len(self.eLDA_mu.tms), num_models)
        np.testing.assert_allclose(self.eLDA.ttda, self.eLDA_mu.ttda, rtol=1e-05)
        np.testing.assert_allclose(self.eLDA.get_topics(), self.eLDA_mu.get_topics(), rtol=1e-05)
        self.check_ttda(self.eLDA_mu)

    def test_generate_gensim_rep(self):
        gensimModel = self.eLDA.generate_gensim_representation()
        topics = gensimModel.get_topics()
        np.testing.assert_allclose(self.eLDA.get_topics(), topics, rtol=1e-05)

    def assert_cluster_results_equal(self, a, b):
        self.assertTrue(isinstance(a, dict))
        self.assertTrue(isinstance(b, dict))
        np.testing.assert_array_equal([row["label"] for row in a.values()],
                                      [row["label"] for row in b.values()])
        np.testing.assert_array_equal([row["is_core"] for row in a.values()],
                                      [row["is_core"] for row in b.values()])
        np.testing.assert_array_equal([row["num_samples"] for row in a.values()],
                                      [row["num_samples"] for row in b.values()])

    def test_persisting(self):
        fname = get_tmpfile('gensim_models_ensemblelda')
        self.eLDA.save(fname)
        loaded_eLDA = EnsembleLda.load(fname)
        # storing the ensemble without memory_friendy_ttda
        self.eLDA_mu.save(fname)
        loaded_eLDA_mu = EnsembleLda.load(fname)

        # was it stored and loaded correctly?
        # memory friendly
        loaded_eLDA_representation = loaded_eLDA.generate_gensim_representation()
        topics = loaded_eLDA_representation.get_topics()
        ttda = loaded_eLDA.ttda
        amatrix = loaded_eLDA.asymmetric_distance_matrix
        np.testing.assert_allclose(self.eLDA.get_topics(), topics, rtol=1e-05)
        np.testing.assert_allclose(self.eLDA.ttda, ttda, rtol=1e-05)
        np.testing.assert_allclose(self.eLDA.asymmetric_distance_matrix, amatrix, rtol=1e-05)

        a = self.eLDA.cluster_model.results
        b = loaded_eLDA.cluster_model.results

        self.assert_cluster_results_equal(a, b)

        # memory unfriendly
        loaded_eLDA_mu_representation = loaded_eLDA_mu.generate_gensim_representation()
        topics = loaded_eLDA_mu_representation.get_topics()
        np.testing.assert_allclose(self.eLDA.get_topics(), topics, rtol=1e-05)

    def test_multiprocessing(self):
        # same configuration
        random_state = 0

        # use 3 processes for the ensemble and the distance,
        # so that the 4 models and 8 topics cannot be distributed
        # to each worker evenly
        workers = 3

        # memory friendly. contains List of topic word distributions
        eLDA_multi = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                                 num_topics=num_topics, passes=passes, num_models=num_models,
                                 random_state=random_state, ensemble_workers=workers, distance_workers=workers)

        # memory unfriendly. contains List of models
        eLDA_multi_mu = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                                    num_topics=num_topics, passes=passes, num_models=num_models,
                                    random_state=random_state, ensemble_workers=workers, distance_workers=workers,
                                    memory_friendly_ttda=False)

        np.testing.assert_allclose(self.eLDA.get_topics(), eLDA_multi.get_topics(), rtol=1e-05)
        np.testing.assert_allclose(self.eLDA_mu.get_topics(), eLDA_multi_mu.get_topics(), rtol=1e-05)

    def test_add_models(self):
        # same configuration
        num_models = self.eLDA.num_models

        # make sure countings and sizes after adding are correct
        # create new models and add other models to them.

        # there are a ton of configurations for the first parameter possible,
        # try them all

        # quickly train something that can be used for counting results
        num_new_models = 3
        num_new_topics = 3

        # 1. memory friendly
        eLDA_base = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                                num_topics=num_new_topics, passes=1, num_models=num_new_models,
                                iterations=1, random_state=0, topic_model_kind='ldamulticore',
                                workers=3, ensemble_workers=2)

        # 1.1 ttda
        a = len(eLDA_base.ttda)
        b = eLDA_base.num_models
        eLDA_base.add_model(self.eLDA.ttda)
        self.assertEqual(len(eLDA_base.ttda), a + len(self.eLDA.ttda))
        self.assertEqual(eLDA_base.num_models, b + 1)  # defaults to 1 for one ttda matrix

        # 1.2 an ensemble
        a = len(eLDA_base.ttda)
        b = eLDA_base.num_models
        eLDA_base.add_model(self.eLDA, 5)
        self.assertEqual(len(eLDA_base.ttda), a + len(self.eLDA.ttda))
        self.assertEqual(eLDA_base.num_models, b + 5)

        # 1.3 a list of ensembles
        a = len(eLDA_base.ttda)
        b = eLDA_base.num_models
        # it should be totally legit to add a memory unfriendly object to a memory friendly one
        eLDA_base.add_model([self.eLDA, self.eLDA_mu])
        self.assertEqual(len(eLDA_base.ttda), a + 2 * len(self.eLDA.ttda))
        self.assertEqual(eLDA_base.num_models, b + 2 * num_models)

        # 1.4 a single gensim model
        model = self.eLDA.classic_model_representation

        a = len(eLDA_base.ttda)
        b = eLDA_base.num_models
        eLDA_base.add_model(model)
        self.assertEqual(len(eLDA_base.ttda), a + len(model.get_topics()))
        self.assertEqual(eLDA_base.num_models, b + 1)

        # 1.5 a list gensim models
        a = len(eLDA_base.ttda)
        b = eLDA_base.num_models
        eLDA_base.add_model([model, model])
        self.assertEqual(len(eLDA_base.ttda), a + 2 * len(model.get_topics()))
        self.assertEqual(eLDA_base.num_models, b + 2)

        self.check_ttda(eLDA_base)

        # 2. memory unfriendly
        eLDA_base_mu = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                                   num_topics=num_new_topics, passes=1, num_models=num_new_models,
                                   iterations=1, random_state=0, topic_model_kind='ldamulticore',
                                   workers=3, ensemble_workers=2, memory_friendly_ttda=False)

        # 2.1 a single ensemble
        a = len(eLDA_base_mu.tms)
        b = eLDA_base_mu.num_models
        eLDA_base_mu.add_model(self.eLDA_mu)
        self.assertEqual(len(eLDA_base_mu.tms), a + num_models)
        self.assertEqual(eLDA_base_mu.num_models, b + num_models)

        # 2.2 a list of ensembles
        a = len(eLDA_base_mu.tms)
        b = eLDA_base_mu.num_models
        eLDA_base_mu.add_model([self.eLDA_mu, self.eLDA_mu])
        self.assertEqual(len(eLDA_base_mu.tms), a + 2 * num_models)
        self.assertEqual(eLDA_base_mu.num_models, b + 2 * num_models)

        # 2.3 a single gensim model
        a = len(eLDA_base_mu.tms)
        b = eLDA_base_mu.num_models
        eLDA_base_mu.add_model(self.eLDA_mu.tms[0])
        self.assertEqual(len(eLDA_base_mu.tms), a + 1)
        self.assertEqual(eLDA_base_mu.num_models, b + 1)

        # 2.4 a list of gensim models
        a = len(eLDA_base_mu.tms)
        b = eLDA_base_mu.num_models
        eLDA_base_mu.add_model(self.eLDA_mu.tms)
        self.assertEqual(len(eLDA_base_mu.tms), a + num_models)
        self.assertEqual(eLDA_base_mu.num_models, b + num_models)

        # 2.5 topic term distributions should throw errors, because the
        # actual models are needed for the memory unfriendly ensemble
        a = len(eLDA_base_mu.tms)
        b = eLDA_base_mu.num_models
        self.assertRaises(ValueError, lambda: eLDA_base_mu.add_model(self.eLDA_mu.tms[0].get_topics()))
        # remains unchanged
        self.assertEqual(len(eLDA_base_mu.tms), a)
        self.assertEqual(eLDA_base_mu.num_models, b)

        self.assertEqual(eLDA_base_mu.num_models, len(eLDA_base_mu.tms))
        self.check_ttda(eLDA_base_mu)

    def test_recluster(self):
        # 6.2: see if after adding a model, the model still makes sense
        num_new_models = 3
        num_new_topics = 3

        # train
        new_eLDA = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                               num_topics=num_new_topics, passes=10, num_models=num_new_models,
                               iterations=30, random_state=1, topic_model_kind='ldamulticore',
                               distance_workers=4)
        new_eLDA_mu = EnsembleLda(corpus=common_corpus, id2word=common_dictionary,
                                  num_topics=num_new_topics, passes=10, num_models=num_new_models,
                                  iterations=30, random_state=1, topic_model_kind='ldamulticore',
                                  distance_workers=4, memory_friendly_ttda=False)
        # both should be similar
        np.testing.assert_allclose(new_eLDA.ttda, new_eLDA_mu.ttda, rtol=1e-05)
        np.testing.assert_allclose(new_eLDA.get_topics(), new_eLDA_mu.get_topics(), rtol=1e-05)
        # and every next step applied to both should result in similar results

        # 1. adding to ttda and tms
        new_eLDA.add_model(self.eLDA)
        new_eLDA_mu.add_model(self.eLDA_mu)
        np.testing.assert_allclose(new_eLDA.ttda, new_eLDA_mu.ttda, rtol=1e-05)
        self.assertEqual(len(new_eLDA.ttda), len(self.eLDA.ttda) + num_new_models * num_new_topics)
        self.assertEqual(len(new_eLDA_mu.ttda), len(self.eLDA_mu.ttda) + num_new_models * num_new_topics)
        self.assertEqual(len(new_eLDA_mu.tms), num_models + num_new_models)
        self.check_ttda(new_eLDA)
        self.check_ttda(new_eLDA_mu)

        # 2. distance matrix
        new_eLDA.generate_asymmetric_distance_matrix()
        new_eLDA_mu.generate_asymmetric_distance_matrix()
        np.testing.assert_allclose(new_eLDA.asymmetric_distance_matrix,
                                   new_eLDA_mu.asymmetric_distance_matrix)

        # 3. CBDBSCAN results
        new_eLDA.generate_topic_clusters()
        new_eLDA_mu.generate_topic_clusters()
        a = new_eLDA.cluster_model.results
        b = new_eLDA_mu.cluster_model.results
        self.assert_cluster_results_equal(a, b)

        # 4. finally, the stable topics
        new_eLDA.generate_stable_topics()
        new_eLDA_mu.generate_stable_topics()
        np.testing.assert_allclose(new_eLDA.get_topics(),
                                   new_eLDA_mu.get_topics())

        new_eLDA.generate_gensim_representation()
        new_eLDA_mu.generate_gensim_representation()

        # same random state, hence topics should be still similar
        np.testing.assert_allclose(new_eLDA.get_topics(), new_eLDA_mu.get_topics(), rtol=1e-05)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
    unittest.main()
