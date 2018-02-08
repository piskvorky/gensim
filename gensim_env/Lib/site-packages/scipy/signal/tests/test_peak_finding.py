from __future__ import division, print_function, absolute_import

import copy

import numpy as np
from numpy.testing import (assert_equal,
    assert_array_equal, assert_)
from scipy.signal._peak_finding import (argrelmax, argrelmin,
    find_peaks_cwt, _identify_ridge_lines)
from scipy._lib.six import xrange


def _gen_gaussians(center_locs, sigmas, total_length):
    xdata = np.arange(0, total_length).astype(float)
    out_data = np.zeros(total_length, dtype=float)
    for ind, sigma in enumerate(sigmas):
        tmp = (xdata - center_locs[ind]) / sigma
        out_data += np.exp(-(tmp**2))
    return out_data


def _gen_gaussians_even(sigmas, total_length):
    num_peaks = len(sigmas)
    delta = total_length / (num_peaks + 1)
    center_locs = np.linspace(delta, total_length - delta, num=num_peaks).astype(int)
    out_data = _gen_gaussians(center_locs, sigmas, total_length)
    return out_data, center_locs


def _gen_ridge_line(start_locs, max_locs, length, distances, gaps):
    """
    Generate coordinates for a ridge line.

    Will be a series of coordinates, starting a start_loc (length 2).
    The maximum distance between any adjacent columns will be
    `max_distance`, the max distance between adjacent rows
    will be `map_gap'.

    `max_locs` should be the size of the intended matrix. The
    ending coordinates are guaranteed to be less than `max_locs`,
    although they may not approach `max_locs` at all.
    """

    def keep_bounds(num, max_val):
        out = max(num, 0)
        out = min(out, max_val)
        return out

    gaps = copy.deepcopy(gaps)
    distances = copy.deepcopy(distances)

    locs = np.zeros([length, 2], dtype=int)
    locs[0, :] = start_locs
    total_length = max_locs[0] - start_locs[0] - sum(gaps)
    if total_length < length:
        raise ValueError('Cannot generate ridge line according to constraints')
    dist_int = length / len(distances) - 1
    gap_int = length / len(gaps) - 1
    for ind in xrange(1, length):
        nextcol = locs[ind - 1, 1]
        nextrow = locs[ind - 1, 0] + 1
        if (ind % dist_int == 0) and (len(distances) > 0):
            nextcol += ((-1)**ind)*distances.pop()
        if (ind % gap_int == 0) and (len(gaps) > 0):
            nextrow += gaps.pop()
        nextrow = keep_bounds(nextrow, max_locs[0])
        nextcol = keep_bounds(nextcol, max_locs[1])
        locs[ind, :] = [nextrow, nextcol]

    return [locs[:, 0], locs[:, 1]]


class TestRidgeLines(object):

    def test_empty(self):
        test_matr = np.zeros([20, 100])
        lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
        assert_(len(lines) == 0)

    def test_minimal(self):
        test_matr = np.zeros([20, 100])
        test_matr[0, 10] = 1
        lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
        assert_(len(lines) == 1)

        test_matr = np.zeros([20, 100])
        test_matr[0:2, 10] = 1
        lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
        assert_(len(lines) == 1)

    def test_single_pass(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 0, 1]
        test_matr = np.zeros([20, 50]) + 1e-12
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_distances = max(distances)*np.ones(20)
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_array_equal(identified_lines, [line])

    def test_single_bigdist(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 4]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 3
        max_distances = max_dist*np.ones(20)
        #This should get 2 lines, since the distance is too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_(len(identified_lines) == 2)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggap(self):
        distances = [0, 1, 2, 5]
        max_gap = 3
        gaps = [0, 4, 2, 1]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 6
        max_distances = max_dist*np.ones(20)
        #This should get 2 lines, since the gap is too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 2)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggaps(self):
        distances = [0]
        max_gap = 1
        gaps = [3, 6]
        test_matr = np.zeros([50, 50])
        length = 30
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 1
        max_distances = max_dist*np.ones(50)
        #This should get 3 lines, since the gaps are too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 3)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)


class TestArgrel(object):

    def test_empty(self):
        # Regression test for gh-2832.
        # When there are no relative extrema, make sure that
        # the number of empty arrays returned matches the
        # dimension of the input.

        empty_array = np.array([], dtype=int)

        z1 = np.zeros(5)

        i = argrelmin(z1)
        assert_equal(len(i), 1)
        assert_array_equal(i[0], empty_array)

        z2 = np.zeros((3,5))

        row, col = argrelmin(z2, axis=0)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)

        row, col = argrelmin(z2, axis=1)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)

    def test_basic(self):
        # Note: the docstrings for the argrel{min,max,extrema} functions
        # do not give a guarantee of the order of the indices, so we'll
        # sort them before testing.

        x = np.array([[1, 2, 2, 3, 2],
                      [2, 1, 2, 2, 3],
                      [3, 2, 1, 2, 2],
                      [2, 3, 2, 1, 2],
                      [1, 2, 3, 2, 1]])

        row, col = argrelmax(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [4, 0, 1])

        row, col = argrelmax(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [0, 3, 4])
        assert_equal(col[order], [3, 1, 2])

        row, col = argrelmin(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])

        row, col = argrelmin(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])

    def test_highorder(self):
        order = 2
        sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 500)
        test_data[act_locs + order] = test_data[act_locs]*0.99999
        test_data[act_locs - order] = test_data[act_locs]*0.99999
        rel_max_locs = argrelmax(test_data, order=order, mode='clip')[0]

        assert_(len(rel_max_locs) == len(act_locs))
        assert_((rel_max_locs == act_locs).all())

    def test_2d_gaussians(self):
        sigmas = [1.0, 2.0, 10.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 100)
        rot_factor = 20
        rot_range = np.arange(0, len(test_data)) - rot_factor
        test_data_2 = np.vstack([test_data, test_data[rot_range]])
        rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)

        for rw in xrange(0, test_data_2.shape[0]):
            inds = (rel_max_rows == rw)

            assert_(len(rel_max_cols[inds]) == len(act_locs))
            assert_((act_locs == (rel_max_cols[inds] - rot_factor*rw)).all())


class TestFindPeaks(object):

    def test_find_peaks_exact(self):
        """
        Generate a series of gaussians and attempt to find the peak locations.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0,
                                         min_length=None)
        np.testing.assert_array_equal(found_locs, act_locs,
                        "Found maximum locations did not equal those expected")

    def test_find_peaks_withnoise(self):
        """
        Verify that peak locations are (approximately) found
        for a series of gaussians with added noise.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        noise_amp = 0.07
        np.random.seed(18181911)
        test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        found_locs = find_peaks_cwt(test_data, widths, min_length=15,
                                         gap_thresh=1, min_snr=noise_amp / 5)

        np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' +
                                'of peaks found than expected')
        diffs = np.abs(found_locs - act_locs)
        max_diffs = np.array(sigmas) / 5
        np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' +
                                     'by more than %s' % (max_diffs))

    def test_find_peaks_nopeak(self):
        """
        Verify that no peak is found in
        data that's just noise.
        """
        noise_amp = 1.0
        num_points = 100
        np.random.seed(181819141)
        test_data = (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        widths = np.arange(10, 50)
        found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
        np.testing.assert_equal(len(found_locs), 0)

