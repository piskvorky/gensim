import unittest
import itertools

from morfessor.evaluation import WilcoxonSignedRank

class TestWilcoxon(unittest.TestCase):
    def setUp(self):
        self.obj = WilcoxonSignedRank()

    def test_norm_cum_pdf(self):
        self.assertAlmostEqual(self.obj._norm_cum_pdf(1.9599639845400), 0.025)

    def test_accuracy_wilcoxon(self):
        #Same tests as used for scipy.stats.morestats
        freq = [1, 4, 16, 15, 8, 4, 5, 1, 2]
        nums = range(-4, 5)
        x = list(itertools.chain(*[[u] * v for u, v in zip(nums, freq)]))

        self.assertEqual(len(x), 56)

        p = self.obj._wilcoxon(x, correction=False)
        self.assertAlmostEqual(p, 0.00197547303533107)

        p = self.obj._wilcoxon(x, "wilcox", correction=False)
        self.assertAlmostEqual(p, 0.00641346115861)

        x = [120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112]
        y = [133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187]

        p = self.obj._wilcoxon([a-b for a, b in zip(x,y)])
        self.assertAlmostEqual(p, 0.7240817)
        p = self.obj._wilcoxon([a-b for a, b in zip(x,y)], correction=False)
        self.assertAlmostEqual(p, 0.6948866)


    def test_wilcoxon_tie(self):
        #Same tests as used for scipy.stats.morestats

        p = self.obj._wilcoxon([0.1] * 10, correction=False)
        self.assertAlmostEqual(p, 0.001565402)

        p = self.obj._wilcoxon([0.1] * 10)
        self.assertAlmostEqual(p, 0.001904195)



if __name__ == '__main__':
    unittest.main()