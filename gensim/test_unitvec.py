import numpy as np
from scipy import sparse
import unittest
import matutils

class UnitvecTestCase(unittest.TestCase):

	def manual_unitvec(self, vec):
		self.vec = vec
		if sparse.issparse(self.vec):
			vec_sum_of_squares = self.vec.multiply(self.vec)
			unit = 1. / np.sqrt(vec_sum_of_squares.sum())
			return self.vec.multiply(unit)
		elif not sparse.issparse(self.vec):
			sum_vec_squared = np.sum(self.vec ** 2)
			self.vec /= np.sqrt(sum_vec_squared)
			return self.vec

	def test_unitvec(self):
		input_vector = np.random.uniform(size=(5,)).astype(np.float32)
		unit_vector = matutils.unitvec(input_vector)
		self.assertEqual(input_vector.dtype, unit_vector.dtype)
		self.assertTrue(np.allclose(unit_vector, self.manual_unitvec(input_vector)))

if __name__ == '__main__':

	unittest.main()
