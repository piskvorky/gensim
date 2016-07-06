"""

Tests to check helper DTM methods.

"""

import numpy  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
import os.path
import unittest
import logging


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data/DTM', fname)


class TestSSLM(unittest.TestCase):
    def testUpdateZeta(self):
        # setting up mock values 
        mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        obs = numpy.resize(numpy.zeros(562 * 4), (562, 4))
        zeta = numpy.zeros(4)

        # setting up sslm object
        sslm = ldaseqmodel.sslm(mean=mean, variance=variance, obs=obs, zeta=zeta, num_terms=562, num_sequence=4)
        ldaseqmodel.update_zeta(sslm)

        expected_zeta = numpy.array([ 286.24901747,  285.9899686 ,  286.03548494,  286.63929586])
        actual_zeta = sslm.zeta
        self.assertAlmostEqual(expected_zeta[0], actual_zeta[0], places=2)

    def testPostVariance(self):

        # setting up mock values
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        fwd_variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        chain_variance = 0.005

        sslm = ldaseqmodel.sslm(chain_variance=chain_variance, obs_variance=0.5, num_terms=562, num_sequence=4, variance=variance, fwd_variance=fwd_variance)

        # since we only check for the 0th word of compute_post_variance, we initialise our mock values

        sslm.variance[0] = numpy.loadtxt(datapath('before_variance'))
        sslm.fwd_variance[0] = numpy.loadtxt(datapath('before_fwd_variance'))

        ldaseqmodel.compute_post_variance(0, sslm, chain_variance)

        expected_variance = numpy.array([0.130797, 0.126054, 0.123787, 0.123906, 0.126415])
        expected_fwd_variance = numpy.array([5, 0.454587, 0.239471, 0.164191, 0.126415])

        self.assertAlmostEqual(expected_variance[0], sslm.variance[0][0], places=2)
        self.assertAlmostEqual(expected_fwd_variance[0], sslm.fwd_variance[0][0], places=2)

    def testPostMean(self):

        # setting up mock values
        obs = numpy.resize(numpy.zeros(562 * 4), (562, 4))
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        fwd_variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        fwd_mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        chain_variance = 0.005

        sslm = ldaseqmodel.sslm(chain_variance=chain_variance, obs_variance=0.5, num_terms=562, num_sequence=4, variance=variance, fwd_variance=fwd_variance, mean=mean, fwd_mean=fwd_mean, obs=obs)

        # since we only check for the 0th word of compute_post_mean, we initialise our mock values
        sslm.obs[0] = numpy.loadtxt(datapath('before_obs'))
        sslm.mean[0] = numpy.loadtxt(datapath('before_mean'))
        sslm.fwd_mean[0] = numpy.loadtxt(datapath('before_fwd_mean'))
        sslm.variance[0] = numpy.loadtxt(datapath('before_variance'))
        sslm.fwd_variance[0] = numpy.loadtxt(datapath('before_fwd_variance1'))

        ldaseqmodel.compute_post_mean(0, sslm, chain_variance)

        expected_mean = numpy.array([-1.40784, -1.40924, -1.41058, -1.41093, -1.41111])
        expected_fwd_mean = numpy.array([0, -1.28744, -1.39419, -1.40497, -1.41111])

        self.assertAlmostEqual(expected_mean[0], sslm.mean[0][0], places=2)
        self.assertAlmostEqual(expected_fwd_mean[0], sslm.fwd_mean[0][0], places=2)

    def testLogProb(self):

        # setting up mock values
        zeta = numpy.loadtxt(datapath('eprob_zeta'))
        mean = numpy.split(numpy.loadtxt(datapath('eprob_mean')), 562)
        e_log_prob = numpy.loadtxt(datapath('eprob_before'))
        e_log_prob = numpy.resize(e_log_prob, (562, 4))
        chain_variance = 0.005

        sslm = ldaseqmodel.sslm(chain_variance=chain_variance, obs_variance=0.5, num_terms=562, num_sequence=4, mean=mean, zeta=zeta, e_log_prob=e_log_prob)

        # we are only checking the first few values;
        expected_log_prob = numpy.array([-4.75, -4.7625, -4.76608, -4.76999])
        ldaseqmodel.compute_expected_log_prob(sslm)

        self.assertAlmostEqual(expected_log_prob[0], sslm.e_log_prob[0][0], places=2)


    def testUpdatePhi(self):

        # we test update phi for one particular document
        doc = ldaseqmodel.doc(nterms=3, word=[549, 560, 561])
        topics = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_topics')), 562))
        lda = ldaseqmodel.mockLDA(num_topics=2, topics=topics)

        log_phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_logphi')), 116))
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_posterior_phi')), 116))
        gamma = numpy.array(numpy.loadtxt(datapath('before_posterior_gamma')))

        lda_post = ldaseqmodel.lda_post(lda=lda, doc=doc, log_phi= log_phi, phi=phi, gamma=gamma)
        ldaseqmodel.update_phi(10, 3, lda_post, None, None)

        expected_log_phi = numpy.array([[-105.04211145, 0. ], [-103.88817145, 0. ]])
        expected_phi = numpy.array([[  2.40322000e-46,   1.00000000e+00], [  7.61974000e-46,   1.00000000e+00]])

        self.assertAlmostEqual(expected_log_phi[0][0], lda_post.log_phi[0][0], places=2)
        self.assertAlmostEqual(expected_phi[0][0], lda_post.phi[0][0], places=2)

    def testUpdateGamma(self):

        doc = ldaseqmodel.doc(nterms=3, count=[1, 1, 1])
        lda = ldaseqmodel.mockLDA(num_topics=2, alpha=[0.01, 0.01])
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_update_phi')), 116))
        lda_post = ldaseqmodel.lda_post(lda=lda, doc=doc, gamma=[0.01, 0.01], phi=phi)

        ldaseqmodel.update_gamma(lda_post)

        expected_gamma = numpy.array([0.01, 3.01])
        self.assertAlmostEqual(expected_gamma[1], lda_post.gamma[1], places=2)

    def testUpdateSeqSS(self):
        lda = ldaseqmodel.mockLDA(num_topics=2, alpha=[0.01, 0.01])
        doc = ldaseqmodel.doc(nterms=3, total=3, word=[549, 560, 561],count=[1, 1 ,1])
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_phi')), 116))
        topic_suffstats = [numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_sstats_0')), 562)), numpy.array(numpy.split(numpy.loadtxt(datapath('before_ldaseq_sstats_1')), 562))]
        lda_post = ldaseqmodel.lda_post(lda=lda, doc=doc, phi = phi)

        ldaseqmodel.update_lda_seq_ss(3, doc, lda_post, topic_suffstats)

        # note: we are only checking the first slice of values. Sufficient Stats is actually a list of matrices.
        expected_sstats = numpy.array([[4.00889000e-46, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [ 3., 0., 0., 0.]])
        self.assertAlmostEqual(expected_sstats[0][0], topic_suffstats[0][0][0], places=2)
        self.assertAlmostEqual(expected_sstats[1][0], topic_suffstats[1][0][0], places=2)

    def testInitLdaPost(self):
        lda = ldaseqmodel.mockLDA(num_topics=2, alpha=[0.01, 0.01])
        doc = ldaseqmodel.doc(nterms=3, total=3, word=[549, 560, 561], count=[1, 1, 1])

        # 116 is the number of terms in time_slice 4, and 2 is the number of topics
        phi = numpy.resize(numpy.zeros(116 * 2), (116, 2))
        gamma = numpy.array(numpy.loadtxt(datapath('before_posterior_gamma')))
        lda_post = ldaseqmodel.lda_post(lda=lda, doc=doc, gamma=gamma, phi=phi)
        ldaseqmodel.init_lda_post(lda_post)

        expected_gamma = [1.51, 1.51]
        # there will be 116 phi values for each word
        expected_phi = [0.5, 0.5]
        self.assertAlmostEqual(lda_post.gamma[0], expected_gamma[0], places=2)
        self.assertAlmostEqual(lda_post.phi[0][0], expected_phi[0], places=2)

    def testMeanDeriv(self):

        deriv = numpy.array(numpy.loadtxt(datapath('before_mean_deriv')))
        fwd_variance = numpy.array(numpy.loadtxt(datapath('before_mean_deriv_variance')))
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        variance[560] = fwd_variance
        sslm = ldaseqmodel.sslm(num_sequence=4, variance=variance, obs_variance=0.500000 , chain_variance=0.005000)
        
        ldaseqmodel.compute_mean_deriv(560, 3, sslm, deriv)

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
        m_update_coeff = numpy.array(numpy.split(numpy.loadtxt(datapath('before_obs_m_update')), 562))
        w_phi_l  = numpy.array(numpy.split(numpy.loadtxt(datapath('before_obs_w_phi_l')), 562))

        num_sequence = 4
        chain_variance = 0.005
        word = 560

        sslm = ldaseqmodel.sslm(num_sequence=num_sequence, mean=mean, variance=variance, m_update_coeff=m_update_coeff, w_phi_l=w_phi_l, chain_variance=chain_variance, zeta=zeta)
        ldaseqmodel.compute_obs_deriv(word, word_counts, totals, sslm, mean_deriv_mtx, deriv)

        expected_deriv = numpy.array([1.97886e-06, 1.32927e-06, -9.90162e-08, -3.65708e-07])
        self.assertAlmostEqual(deriv[0], expected_deriv[0], places=2)


    def testUpdateBound(self):

        totals = numpy.array(numpy.loadtxt(datapath('before_bound_totals')))
        zeta = numpy.array(numpy.loadtxt(datapath('before_bound_zeta')))
        variance = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_variance')), 562))
        fwd_variance = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_fwd_variance')), 562))
        mean = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_mean')), 562))
        fwd_mean = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_fwd_mean')), 562))
        obs = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_obs')), 562))
        w_phi_l = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_w_phi_l')), 562))
        counts = numpy.array(numpy.split(numpy.loadtxt(datapath('before_bound_counts')), 562))

        num_terms = 562
        num_sequence = 4
        chain_variance = 0.005
        obs_variance = 0.5

        sslm = ldaseqmodel.sslm(num_terms=num_terms, num_sequence=num_sequence, variance=variance, chain_variance=chain_variance, obs_variance=obs_variance,zeta=zeta, w_phi_l=w_phi_l, mean=mean, fwd_variance=fwd_variance, fwd_mean=fwd_mean, obs=obs)
        
        bound = ldaseqmodel.compute_bound(counts, totals, sslm)
        expected_bound = 40236.251641 
        # self.assertAlmostEqual(bound, expected_bound, places=2)


    def testLdaLhood(self):
        gamma = numpy.array(numpy.loadtxt(datapath('before_lhood_gamma')))
        lhood = numpy.array(numpy.loadtxt(datapath('before_lhood_lhood')))
        log_phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_lhood_log_phi')), 116))
        phi = numpy.array(numpy.split(numpy.loadtxt(datapath('before_lhood_phi')), 116))
        alpha = numpy.array(numpy.loadtxt(datapath('before_lhood_lda_alpha')))
        topics = numpy.array(numpy.split(numpy.loadtxt(datapath('before_lhood_lda_topics')), 562))

        num_topics = 2
        nterms = 3
        count = [1, 1, 1]
        word = [549, 560, 561]

        doc = ldaseqmodel.doc(nterms=nterms, count=count, word=word)
        lda = ldaseqmodel.mockLDA(num_topics=num_topics, alpha=alpha, topics=topics)
        lda_post = ldaseqmodel.lda_post(doc=doc, lda=lda, gamma=gamma, lhood=lhood, phi=phi, log_phi=log_phi)
        lhood = ldaseqmodel.compute_lda_lhood(lda_post)
        expected_lhood = -16.110510 
        self.assertAlmostEqual(lhood, expected_lhood, places=2)

    def testDfObs(self):
        variance = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_variance')), 562))
        mean = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_mean')), 562))
        fwd_variance = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_fwd_variance')), 562))
        fwd_mean = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_fwd_mean')), 562))

        obs = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_obs')), 562))
        w_phi_l = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_w_phi_l')), 562))
        m_update_coeff = numpy.array(numpy.split(numpy.loadtxt(datapath('before_fobs_mupdate')), 562))
        totals = numpy.array(numpy.loadtxt(datapath('before_fobs_totals')))
        zeta = numpy.array(numpy.loadtxt(datapath('before_fobs_zeta')))
        word_counts = numpy.array(numpy.loadtxt(datapath('before_fobs_wordcounts')))
        x =  numpy.array(numpy.loadtxt(datapath('before_fobs_x')))

        chain_variance = 0.005
        word = 560

        sslm = ldaseqmodel.sslm(obs=obs, num_sequence=4, chain_variance=chain_variance, zeta=zeta, mean=mean, variance=variance, w_phi_l=w_phi_l, m_update_coeff=m_update_coeff, fwd_mean=fwd_mean, fwd_variance=fwd_variance)
        params = ldaseqmodel.opt_params(sslm=sslm, word=word, word_counts=word_counts, totals=totals)

        val = ldaseqmodel.f_obs(x, params)
        expected_val = 0.188
        self.assertAlmostEqual(val, expected_val, places=2)

        
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
