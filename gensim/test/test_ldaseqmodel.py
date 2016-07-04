"""

Tests to check helper DTM methods.

"""

import numpy  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
import os.path
import unittest
import logging


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class TestSSLM(unittest.TestCase):
    def testUpdateZeta(self):
        # setting up mock values 
        mean = numpy.split(numpy.loadtxt(datapath('sample_mean_DTM')), 562)
        variance = numpy.split(numpy.loadtxt(datapath('sample_variance_DTM')), 562)
        obs = numpy.resize(numpy.zeros(562 * 4), (562, 4))
        zeta = numpy.zeros(4)

        # setting up sslm object
        sslm = ldaseqmodel.sslm(mean=mean, variance=variance, obs=obs, zeta=zeta)
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

# to test by Wednesday:
# 1. update_phi
# 2. update_gamma
# 3. update_ldaseq_ss
# 4. init_lda_post
# 5. compute_bound
# 6. compute_mean_deriv
# 7. compute_obs_deriv
# 8. compute_lda_lhood

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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
