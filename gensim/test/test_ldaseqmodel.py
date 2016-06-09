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
    def setUp(self):
        self.obs = numpy.resize(numpy.zeros(562 * 4), (562, 4))
        mean = numpy.loadtxt(datapath('sample_mean_DTM'))
        variance= numpy.loadtxt(datapath('sample_variance_DTM'))
        self.mean = numpy.split(mean, 562)
        self.variance = numpy.split(variance, 562)
        self.zeta = numpy.zeros(4)

    def testUpdateZeta(self):
        ldaseqmodel.update_zeta(self)
        expected_zeta = numpy.array([ 286.24901747,  285.9899686 ,  286.03548494,  286.63929586])
        actual_zeta = self.zeta
        self.assertAlmostEqual(expected_zeta[0], actual_zeta[0], places=2)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
