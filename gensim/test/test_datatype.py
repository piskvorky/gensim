import logging
import unittest

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class TestDataType(unittest.TestCase):
    def test_datatype(self):
        kv = KeyedVectors.load_word2vec_format('test.kv.txt', datatype=np.float64)
        self.assertEqual(kv['horse.n.01'][0], -0.0008546282343595379)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
