"""

Tests to check DTM math functions and Topic-Word, Doc-Topic proportions.

"""

import numpy  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel, ldamodel
from gensim.corpora import Dictionary, bleicorpus
import os.path
import unittest
import logging


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data/DTM', fname)


class TestSSLM(unittest.TestCase):

    def setUp(self):
        self.sslm = ldaseqmodel.sslm(vocab_len=562, num_time_slices=4)

    def testUpdateZeta(self):
        # setting up mock values 
        mean = numpy.array(numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562))
        variance = numpy.array(numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562))
        zeta = numpy.zeros(4)
        self.sslm.mean = mean
        self.sslm.variance = variance
        self.sslm.zeta = zeta
        self.sslm.update_zeta()

        expected_zeta = numpy.array([ 286.24901747,  285.9899686 ,  286.03548494,  286.63929586])
        actual_zeta = self.sslm.zeta
        self.assertAlmostEqual(expected_zeta[0], actual_zeta[0], places=2)

    def testPostVariance(self):

        # setting up mock values
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        fwd_variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        chain_variance = 0.005

        self.sslm.variance = variance
        self.sslm.fwd_variance = fwd_variance
        self.sslm.chain_variance = chain_variance
        # since we only check for the 0th word of compute_post_variance, we initialise our mock values

        self.sslm.variance[0] = numpy.loadtxt(datapath('before_variance'))
        self.sslm.fwd_variance[0] = numpy.loadtxt(datapath('before_fwd_variance'))

        self.sslm.compute_post_variance(0, chain_variance)

        expected_variance = numpy.array([0.130797, 0.126054, 0.123787, 0.123906, 0.126415])
        expected_fwd_variance = numpy.array([5, 0.454587, 0.239471, 0.164191, 0.126415])

        self.assertAlmostEqual(expected_variance[0], self.sslm.variance[0][0], places=2)
        self.assertAlmostEqual(expected_fwd_variance[0], self.sslm.fwd_variance[0][0], places=2)

    def testPostMean(self):

        # setting up mock values
        obs = numpy.resize(numpy.zeros(562 * 4), (562, 4))
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        fwd_variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        fwd_mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        chain_variance = 0.005

        self.sslm.variance = variance
        self.sslm.fwd_variance = fwd_variance
        self.sslm.chain_variance = chain_variance
        self.sslm.mean = mean
        self.sslm.fwd_mean = fwd_mean
        self.sslm.obs = obs

        # since we only check for the 0th word of compute_post_mean, we initialise our mock values
        self.sslm.obs[0] = numpy.loadtxt(datapath('before_obs'))
        self.sslm.mean[0] = numpy.loadtxt(datapath('before_mean'))
        self.sslm.fwd_mean[0] = numpy.loadtxt(datapath('before_fwd_mean'))
        self.sslm.variance[0] = numpy.loadtxt(datapath('before_variance'))
        self.sslm.fwd_variance[0] = numpy.loadtxt(datapath('before_fwd_variance1'))

        self.sslm.compute_post_mean(0, chain_variance)

        expected_mean = numpy.array([-1.40784, -1.40924, -1.41058, -1.41093, -1.41111])
        expected_fwd_mean = numpy.array([0, -1.28744, -1.39419, -1.40497, -1.41111])

        self.assertAlmostEqual(expected_mean[0], self.sslm.mean[0][0], places=2)
        self.assertAlmostEqual(expected_fwd_mean[0], self.sslm.fwd_mean[0][0], places=2)

    def testLogProb(self):

        # setting up mock values
        zeta = numpy.loadtxt(datapath('eprob_zeta'))
        mean = numpy.split(numpy.loadtxt(datapath('eprob_mean')), 562)
        e_log_prob = numpy.loadtxt(datapath('eprob_before'))
        e_log_prob = numpy.resize(e_log_prob, (562, 4))
        chain_variance = 0.005

        self.sslm.chain_variance = chain_variance
        self.sslm.mean = mean
        self.sslm.e_log_prob = e_log_prob
        self.sslm.zeta = zeta
        # we are only checking the first few values;
        expected_log_prob = numpy.array([-4.75, -4.7625, -4.76608, -4.76999])
        self.sslm.compute_expected_log_prob()

        self.assertAlmostEqual(expected_log_prob[0], self.sslm.e_log_prob[0][0], places=2)

    def testMeanDeriv(self):

        deriv = numpy.array(numpy.loadtxt(datapath('before_mean_deriv')))
        fwd_variance = numpy.array(numpy.loadtxt(datapath('before_mean_deriv_variance')))
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        variance[560] = fwd_variance

        self.sslm.obs_variance = 0.500000 
        self.sslm.chain_variance = 0.005000
        self.sslm.variance = variance
        self.sslm.fwd_variance = fwd_variance

        self.sslm.compute_mean_deriv(560, 3, deriv)

        expected_deriv = numpy.array([0.175437, 0.182144, 0.189369, 0.197018, 0.204968])
        self.assertAlmostEqual(deriv[0], expected_deriv[0], places=2)

    def testObsDeriv(self): 

        zeta  = numpy.array(numpy.loadtxt(datapath('before_obs_zeta')))
        deriv = numpy.array(numpy.loadtxt(datapath('before_obs_deriv')))
        totals = numpy.array(numpy.loadtxt(datapath('before_obs_totals')))
        word_counts = numpy.array(numpy.loadtxt(datapath('before_obs_wordcounts')))
        mean_deriv_mtx = numpy.array(numpy.split(numpy.loadtxt(datapath('before_obs_mean_deriv_mtx')), 4))
        mean = numpy.array(numpy.split(numpy.loadtxt(datapath('before_obs_mean')), 562))
        variance = numpy.array(numpy.split(numpy.loadtxt(datapath('before_obs_variance')), 562))

        self.sslm.variance = variance
        self.sslm.chain_variance = 0.005000
        self.sslm.mean = mean
        self.sslm.zeta = zeta

        self.sslm.compute_obs_deriv(560, word_counts, totals, mean_deriv_mtx, deriv)

        expected_deriv = numpy.array([1.97886e-06, 1.32927e-06, -9.90162e-08, -3.65708e-07])
        self.assertAlmostEqual(deriv[0], expected_deriv[0], places=2)

class TestLdaPost(unittest.TestCase):

    def setUp(self):
        self.doc = [(549, 1), (560, 1), (561, 1)]
        self.lda = ldamodel.LdaModel(num_topics=2, alpha=0.01, id2word=Dictionary.load(datapath('test_dictionary')))
        self.ldapost = ldaseqmodel.LdaPost(max_doc_len = 116, num_topics=2, lda=self.lda, doc=self.doc)

    def testUpdatePhi(self):

        # we test update phi for one particular document
        topics = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_topics')), 562))
        lda = self.lda
        lda.topics = topics

        log_phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_logphi')), 116))
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_phi')), 116))
        gamma = numpy.array(numpy.loadtxt(datapath('before_posterior_gamma')))

        self.ldapost.phi = phi
        self.ldapost.log_phi = log_phi
        self.ldapost.gamma = gamma
        self.ldapost.lda = lda
        self.ldapost.update_phi(10, 3)

        expected_log_phi = numpy.array([[-105.04211145, 0. ], [-103.88817145, 0. ]])
        expected_phi = numpy.array([[  2.40322000e-46,   1.00000000e+00], [  7.61974000e-46,   1.00000000e+00]])

        self.assertAlmostEqual(expected_log_phi[0][0], self.ldapost.log_phi[0][0], places=2)
        self.assertAlmostEqual(expected_phi[0][0], self.ldapost.phi[0][0], places=2)

    def testUpdateGamma(self):

        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_update_phi')), 116))
        self.ldapost.gamma = [0.01, 0.01]
        self.ldapost.phi = phi

        self.ldapost.update_gamma()

        expected_gamma = numpy.array([0.01, 3.01])
        self.assertAlmostEqual(expected_gamma[1], self.ldapost.gamma[1], places=2)

    def testUpdateSeqSS(self):
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_phi')), 116))
        topic_suffstats = [numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_sstats_0')), 562)), numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_sstats_1')), 562))]
        self.ldapost.phi = phi

        self.ldapost.update_lda_seq_ss(3, self.doc, topic_suffstats)

        # we are only checking the first slice of values. Sufficient Stats is actually a list of matrices.
        expected_sstats = numpy.array([[4.00889000e-46, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [ 3., 0., 0., 0.]])
        self.assertAlmostEqual(expected_sstats[0][0], topic_suffstats[0][0][0], places=2)
        self.assertAlmostEqual(expected_sstats[1][0], topic_suffstats[1][0][0], places=2)

    def testInitLdaPost(self):
        # 116 is the number of terms in time_slice 4, and 2 is the number of topics
        phi = numpy.resize(numpy.zeros(116 * 2), (116, 2))
        gamma = numpy.array(numpy.loadtxt(datapath('before_posterior_gamma')))
        self.ldapost.phi = phi
        self.ldapost.gamma = gamma
        self.ldapost.init_lda_post()

        expected_gamma = [1.51, 1.51]
        # there will be 116 phi values for each word
        expected_phi = [0.5, 0.5]
        self.assertAlmostEqual(self.ldapost.gamma[0], expected_gamma[0], places=2)
        self.assertAlmostEqual(self.ldapost.phi[0][0], expected_phi[0], places=2)

class TestLdaSeq(unittest.TestCase):
    def setUp(self):
        corpus = bleicorpus.BleiCorpus(datapath('test_corpus'))
        dictionary = Dictionary.load(datapath('test_dictionary'))
        sstats = numpy.loadtxt(datapath('sstats_test'))
        self.ldaseq = ldaseqmodel.LdaSeqModel(corpus = corpus , id2word= dictionary, num_topics=2, initialize='own', sstats=sstats, time_slice=[10, 10, 11])

    def testTopicWord(self):

        topics = self.ldaseq.print_topics(0)
        expected_topic_word = [(0.036999999999999998, 'skills')]
        self.assertAlmostEqual(topics[0][0][0], expected_topic_word[0][0], places=2)
        self.assertEqual(topics[0][0][1], expected_topic_word[0][1])


    def testDocTopic(self):
        doc_topic = self.ldaseq.doc_topics(0)
        expected_doc_topic = 0.00066577896138482028
        self.assertAlmostEqual(doc_topic[0], expected_doc_topic, places=2)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
