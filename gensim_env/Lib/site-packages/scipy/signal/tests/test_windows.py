from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose,
                           assert_equal, assert_, assert_array_less)
from pytest import raises as assert_raises

from scipy._lib._numpy_compat import suppress_warnings
from scipy import signal, fftpack


window_funcs = [
    ('boxcar', ()),
    ('triang', ()),
    ('parzen', ()),
    ('bohman', ()),
    ('blackman', ()),
    ('nuttall', ()),
    ('blackmanharris', ()),
    ('flattop', ()),
    ('bartlett', ()),
    ('hanning', ()),
    ('barthann', ()),
    ('hamming', ()),
    ('kaiser', (1,)),
    ('gaussian', (0.5,)),
    ('general_gaussian', (1.5, 2)),
    ('chebwin', (1,)),
    ('slepian', (2,)),
    ('cosine', ()),
    ('hann', ()),
    ('exponential', ()),
    ('tukey', (0.5,)),
    ]


class TestBartHann(object):

    def test_basic(self):
        assert_allclose(signal.barthann(6, sym=True),
                        [0, 0.35857354213752, 0.8794264578624801,
                         0.8794264578624801, 0.3585735421375199, 0])
        assert_allclose(signal.barthann(7),
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0])
        assert_allclose(signal.barthann(6, False),
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27])


class TestBartlett(object):

    def test_basic(self):
        assert_allclose(signal.bartlett(6), [0, 0.4, 0.8, 0.8, 0.4, 0])
        assert_allclose(signal.bartlett(7), [0, 1/3, 2/3, 1.0, 2/3, 1/3, 0])
        assert_allclose(signal.bartlett(6, False),
                        [0, 1/3, 2/3, 1.0, 2/3, 1/3])


class TestBlackman(object):

    def test_basic(self):
        assert_allclose(signal.blackman(6, sym=False),
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13], atol=1e-14)
        assert_allclose(signal.blackman(7, sym=False),
                        [0, 0.09045342435412804, 0.4591829575459636,
                         0.9203636180999081, 0.9203636180999081,
                         0.4591829575459636, 0.09045342435412804], atol=1e-8)
        assert_allclose(signal.blackman(6),
                        [0, 0.2007701432625305, 0.8492298567374694,
                         0.8492298567374694, 0.2007701432625305, 0],
                        atol=1e-14)
        assert_allclose(signal.blackman(7, True),
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14)


class TestBlackmanHarris(object):

    def test_basic(self):
        assert_allclose(signal.blackmanharris(6, False),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645])
        assert_allclose(signal.blackmanharris(7, sym=False),
                        [6.0e-05, 0.03339172347815117, 0.332833504298565,
                         0.8893697722232837, 0.8893697722232838,
                         0.3328335042985652, 0.03339172347815122])
        assert_allclose(signal.blackmanharris(6),
                        [6.0e-05, 0.1030114893456638, 0.7938335106543362,
                         0.7938335106543364, 0.1030114893456638, 6.0e-05])
        assert_allclose(signal.blackmanharris(7, sym=True),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645,
                         6.0e-05])


class TestBohman(object):

    def test_basic(self):
        assert_allclose(signal.bohman(6),
                        [0, 0.1791238937062839, 0.8343114522576858,
                         0.8343114522576858, 0.1791238937062838, 0])
        assert_allclose(signal.bohman(7, sym=True),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293, 0])
        assert_allclose(signal.bohman(6, False),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293])


class TestBoxcar(object):

    def test_basic(self):
        assert_allclose(signal.boxcar(6), [1, 1, 1, 1, 1, 1])
        assert_allclose(signal.boxcar(7), [1, 1, 1, 1, 1, 1, 1])
        assert_allclose(signal.boxcar(6, False), [1, 1, 1, 1, 1, 1])


cheb_odd_true = array([0.200938, 0.107729, 0.134941, 0.165348,
                       0.198891, 0.235450, 0.274846, 0.316836,
                       0.361119, 0.407338, 0.455079, 0.503883,
                       0.553248, 0.602637, 0.651489, 0.699227,
                       0.745266, 0.789028, 0.829947, 0.867485,
                       0.901138, 0.930448, 0.955010, 0.974482,
                       0.988591, 0.997138, 1.000000, 0.997138,
                       0.988591, 0.974482, 0.955010, 0.930448,
                       0.901138, 0.867485, 0.829947, 0.789028,
                       0.745266, 0.699227, 0.651489, 0.602637,
                       0.553248, 0.503883, 0.455079, 0.407338,
                       0.361119, 0.316836, 0.274846, 0.235450,
                       0.198891, 0.165348, 0.134941, 0.107729,
                       0.200938])

cheb_even_true = array([0.203894, 0.107279, 0.133904,
                        0.163608, 0.196338, 0.231986,
                        0.270385, 0.311313, 0.354493,
                        0.399594, 0.446233, 0.493983,
                        0.542378, 0.590916, 0.639071,
                        0.686302, 0.732055, 0.775783,
                        0.816944, 0.855021, 0.889525,
                        0.920006, 0.946060, 0.967339,
                        0.983557, 0.994494, 1.000000,
                        1.000000, 0.994494, 0.983557,
                        0.967339, 0.946060, 0.920006,
                        0.889525, 0.855021, 0.816944,
                        0.775783, 0.732055, 0.686302,
                        0.639071, 0.590916, 0.542378,
                        0.493983, 0.446233, 0.399594,
                        0.354493, 0.311313, 0.270385,
                        0.231986, 0.196338, 0.163608,
                        0.133904, 0.107279, 0.203894])


class TestChebWin(object):

    def test_basic(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            assert_allclose(signal.chebwin(6, 100),
                            [0.1046401879356917, 0.5075781475823447, 1.0, 1.0,
                             0.5075781475823447, 0.1046401879356917])
            assert_allclose(signal.chebwin(7, 100),
                            [0.05650405062850233, 0.316608530648474,
                             0.7601208123539079, 1.0, 0.7601208123539079,
                             0.316608530648474, 0.05650405062850233])
            assert_allclose(signal.chebwin(6, 10),
                            [1.0, 0.6071201674458373, 0.6808391469897297,
                             0.6808391469897297, 0.6071201674458373, 1.0])
            assert_allclose(signal.chebwin(7, 10),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651, 1.0])
            assert_allclose(signal.chebwin(6, 10, False),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651])

    def test_cheb_odd_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = signal.chebwin(53, at=-40)
        assert_array_almost_equal(cheb_odd, cheb_odd_true, decimal=4)

    def test_cheb_even_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_even = signal.chebwin(54, at=40)
        assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)

    def test_cheb_odd_low_attenuation(self):
        cheb_odd_low_at_true = array([1.000000, 0.519052, 0.586405,
                                      0.610151, 0.586405, 0.519052,
                                      1.000000])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = signal.chebwin(7, at=10)
        assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)

    def test_cheb_even_low_attenuation(self):
        cheb_even_low_at_true = array([1.000000, 0.451924, 0.51027,
                                       0.541338, 0.541338, 0.51027,
                                       0.451924, 1.000000])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_even = signal.chebwin(8, at=-10)
        assert_array_almost_equal(cheb_even, cheb_even_low_at_true, decimal=4)


exponential_data = {
    (4, None, 0.2, False):
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03]),
    (4, None, 0.2, True): array([0.00055308437014783, 0.0820849986238988,
                                 0.0820849986238988, 0.00055308437014783]),
    (4, None, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
                                  0.36787944117144233]),
    (4, None, 1.0, True): array([0.22313016014842982, 0.60653065971263342,
                                 0.60653065971263342, 0.22313016014842982]),
    (4, 2, 0.2, False):
        array([4.53999297624848542e-05, 6.73794699908546700e-03,
               1.00000000000000000e+00, 6.73794699908546700e-03]),
    (4, 2, 0.2, True): None,
    (4, 2, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
                               0.36787944117144233]),
    (4, 2, 1.0, True): None,
    (5, None, 0.2, True):
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03, 4.53999297624848542e-05]),
    (5, None, 1.0, True): array([0.1353352832366127, 0.36787944117144233, 1.,
                                 0.36787944117144233, 0.1353352832366127]),
    (5, 2, 0.2, True): None,
    (5, 2, 1.0, True): None
}


def test_exponential():
    for k, v in exponential_data.items():
        if v is None:
            assert_raises(ValueError, signal.exponential, *k)
        else:
            win = signal.exponential(*k)
            assert_allclose(win, v, rtol=1e-14)


class TestFlatTop(object):

    def test_basic(self):
        assert_allclose(signal.flattop(6, sym=False),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156])
        assert_allclose(signal.flattop(7, sym=False),
                        [-0.000421051, -0.03684078115492348,
                         0.01070371671615342, 0.7808739149387698,
                         0.7808739149387698, 0.01070371671615342,
                         -0.03684078115492348])
        assert_allclose(signal.flattop(6),
                        [-0.000421051, -0.0677142520762119, 0.6068721525762117,
                         0.6068721525762117, -0.0677142520762119,
                         -0.000421051])
        assert_allclose(signal.flattop(7, True),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156, -0.000421051])


class TestGaussian(object):

    def test_basic(self):
        assert_allclose(signal.gaussian(6, 1.0),
                        [0.04393693362340742, 0.3246524673583497,
                         0.8824969025845955, 0.8824969025845955,
                         0.3246524673583497, 0.04393693362340742])
        assert_allclose(signal.gaussian(7, 1.2),
                        [0.04393693362340742, 0.2493522087772962,
                         0.7066482778577162, 1.0, 0.7066482778577162,
                         0.2493522087772962, 0.04393693362340742])
        assert_allclose(signal.gaussian(7, 3),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081, 0.6065306597126334])
        assert_allclose(signal.gaussian(6, 3, False),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081])


class TestHamming(object):

    def test_basic(self):
        assert_allclose(signal.hamming(6, False),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31])
        assert_allclose(signal.hamming(7, sym=False),
                        [0.08, 0.2531946911449826, 0.6423596296199047,
                         0.9544456792351128, 0.9544456792351128,
                         0.6423596296199047, 0.2531946911449826])
        assert_allclose(signal.hamming(6),
                        [0.08, 0.3978521825875242, 0.9121478174124757,
                         0.9121478174124757, 0.3978521825875242, 0.08])
        assert_allclose(signal.hamming(7, sym=True),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])


class TestHann(object):

    def test_basic(self):
        assert_allclose(signal.hann(6, sym=False),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25])
        assert_allclose(signal.hann(7, sym=False),
                        [0, 0.1882550990706332, 0.6112604669781572,
                         0.9504844339512095, 0.9504844339512095,
                         0.6112604669781572, 0.1882550990706332])
        assert_allclose(signal.hann(6, True),
                        [0, 0.3454915028125263, 0.9045084971874737,
                         0.9045084971874737, 0.3454915028125263, 0])
        assert_allclose(signal.hann(7),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0])


class TestKaiser(object):

    def test_basic(self):
        assert_allclose(signal.kaiser(6, 0.5),
                        [0.9403061933191572, 0.9782962393705389,
                         0.9975765035372042, 0.9975765035372042,
                         0.9782962393705389, 0.9403061933191572])
        assert_allclose(signal.kaiser(7, 0.5),
                        [0.9403061933191572, 0.9732402256999829,
                         0.9932754654413773, 1.0, 0.9932754654413773,
                         0.9732402256999829, 0.9403061933191572])
        assert_allclose(signal.kaiser(6, 2.7),
                        [0.2603047507678832, 0.6648106293528054,
                         0.9582099802511439, 0.9582099802511439,
                         0.6648106293528054, 0.2603047507678832])
        assert_allclose(signal.kaiser(7, 2.7),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844, 0.2603047507678832])
        assert_allclose(signal.kaiser(6, 2.7, False),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844])


class TestNuttall(object):

    def test_basic(self):
        assert_allclose(signal.nuttall(6, sym=False),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345])
        assert_allclose(signal.nuttall(7, sym=False),
                        [0.0003628, 0.03777576895352025, 0.3427276199688195,
                         0.8918518610776603, 0.8918518610776603,
                         0.3427276199688196, 0.0377757689535203])
        assert_allclose(signal.nuttall(6),
                        [0.0003628, 0.1105152530498718, 0.7982580969501282,
                         0.7982580969501283, 0.1105152530498719, 0.0003628])
        assert_allclose(signal.nuttall(7, True),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345, 0.0003628])


class TestParzen(object):

    def test_basic(self):
        assert_allclose(signal.parzen(6),
                        [0.009259259259259254, 0.25, 0.8611111111111112,
                         0.8611111111111112, 0.25, 0.009259259259259254])
        assert_allclose(signal.parzen(7, sym=True),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616, 0.00583090379008747])
        assert_allclose(signal.parzen(6, False),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616])


class TestTriang(object):

    def test_basic(self):

        assert_allclose(signal.triang(6, True),
                        [1/6, 1/2, 5/6, 5/6, 1/2, 1/6])
        assert_allclose(signal.triang(7),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4])
        assert_allclose(signal.triang(6, sym=False),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2])


tukey_data = {
    (4, 0.5, True): array([0.0, 1.0, 1.0, 0.0]),
    (4, 0.9, True): array([0.0, 0.84312081893436686,
                           0.84312081893436686, 0.0]),
    (4, 1.0, True): array([0.0, 0.75, 0.75, 0.0]),
    (4, 0.5, False): array([0.0, 1.0, 1.0, 1.0]),
    (4, 0.9, False): array([0.0, 0.58682408883346526,
                            1.0, 0.58682408883346526]),
    (4, 1.0, False): array([0.0, 0.5, 1.0, 0.5]),
    (5, 0.0, True): array([1.0, 1.0, 1.0, 1.0, 1.0]),
    (5, 0.8, True): array([0.0, 0.69134171618254492,
                           1.0, 0.69134171618254492, 0.0]),
    (5, 1.0, True): array([0.0, 0.5, 1.0, 0.5, 0.0]),

    (6, 0): [1, 1, 1, 1, 1, 1],
    (7, 0): [1, 1, 1, 1, 1, 1, 1],
    (6, .25): [0, 1, 1, 1, 1, 0],
    (7, .25): [0, 1, 1, 1, 1, 1, 0],
    (6,): [0, 0.9045084971874737, 1.0, 1.0, 0.9045084971874735, 0],
    (7,): [0, 0.75, 1.0, 1.0, 1.0, 0.75, 0],
    (6, .75): [0, 0.5522642316338269, 1.0, 1.0, 0.5522642316338267, 0],
    (7, .75): [0, 0.4131759111665348, 0.9698463103929542, 1.0,
               0.9698463103929542, 0.4131759111665347, 0],
    (6, 1): [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737,
             0.3454915028125263, 0],
    (7, 1): [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0],
}


class TestTukey(object):

    def test_basic(self):
        # Test against hardcoded data
        for k, v in tukey_data.items():
            if v is None:
                assert_raises(ValueError, signal.tukey, *k)
            else:
                win = signal.tukey(*k)
                assert_allclose(win, v, rtol=1e-14)

    def test_extremes(self):
        # Test extremes of alpha correspond to boxcar and hann
        tuk0 = signal.tukey(100, 0)
        box0 = signal.boxcar(100)
        assert_array_almost_equal(tuk0, box0)

        tuk1 = signal.tukey(100, 1)
        han1 = signal.hann(100)
        assert_array_almost_equal(tuk1, han1)


class TestGetWindow(object):

    def test_boxcar(self):
        w = signal.get_window('boxcar', 12)
        assert_array_equal(w, np.ones_like(w))

        # window is a tuple of len 1
        w = signal.get_window(('boxcar',), 16)
        assert_array_equal(w, np.ones_like(w))

    def test_cheb_odd(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            w = signal.get_window(('chebwin', -40), 53, fftbins=False)
        assert_array_almost_equal(w, cheb_odd_true, decimal=4)

    def test_cheb_even(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            w = signal.get_window(('chebwin', 40), 54, fftbins=False)
        assert_array_almost_equal(w, cheb_even_true, decimal=4)

    def test_kaiser_float(self):
        win1 = signal.get_window(7.2, 64)
        win2 = signal.kaiser(64, 7.2, False)
        assert_allclose(win1, win2)

    def test_invalid_inputs(self):
        # Window is not a float, tuple, or string
        assert_raises(ValueError, signal.get_window, set('hann'), 8)

        # Unknown window type error
        assert_raises(ValueError, signal.get_window, 'broken', 4)

    def test_array_as_window(self):
        # github issue 3603
        osfactor = 128
        sig = np.arange(128)

        win = signal.get_window(('kaiser', 8.0), osfactor // 2)
        assert_raises(ValueError, signal.resample,
                      (sig, len(sig) * osfactor), {'window': win})


def test_windowfunc_basics():
    for window_name, params in window_funcs:
        window = getattr(signal, window_name)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            # Check symmetry for odd and even lengths
            w1 = window(8, *params, sym=True)
            w2 = window(7, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            w1 = window(9, *params, sym=True)
            w2 = window(8, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            # Check that functions run and output lengths are correct
            assert_equal(len(window(6, *params, sym=True)), 6)
            assert_equal(len(window(6, *params, sym=False)), 6)
            assert_equal(len(window(7, *params, sym=True)), 7)
            assert_equal(len(window(7, *params, sym=False)), 7)

            # Check invalid lengths
            assert_raises(ValueError, window, 5.5, *params)
            assert_raises(ValueError, window, -7, *params)

            # Check degenerate cases
            assert_array_equal(window(0, *params, sym=True), [])
            assert_array_equal(window(0, *params, sym=False), [])
            assert_array_equal(window(1, *params, sym=True), [1])
            assert_array_equal(window(1, *params, sym=False), [1])

            # Check dtype
            assert_(window(0, *params, sym=True).dtype == 'float')
            assert_(window(0, *params, sym=False).dtype == 'float')
            assert_(window(1, *params, sym=True).dtype == 'float')
            assert_(window(1, *params, sym=False).dtype == 'float')
            assert_(window(6, *params, sym=True).dtype == 'float')
            assert_(window(6, *params, sym=False).dtype == 'float')

            # Check normalization
            assert_array_less(window(10, *params, sym=True), 1.01)
            assert_array_less(window(10, *params, sym=False), 1.01)
            assert_array_less(window(9, *params, sym=True), 1.01)
            assert_array_less(window(9, *params, sym=False), 1.01)

            # Check that DFT-even spectrum is purely real for odd and even
            assert_allclose(fftpack.fft(window(10, *params, sym=False)).imag,
                            0, atol=1e-14)
            assert_allclose(fftpack.fft(window(11, *params, sym=False)).imag,
                            0, atol=1e-14)


def test_needs_params():
    for winstr in ['kaiser', 'ksr', 'gaussian', 'gauss', 'gss',
                   'general gaussian', 'general_gaussian',
                   'general gauss', 'general_gauss', 'ggs',
                   'slepian', 'optimal', 'slep', 'dss', 'dpss',
                   'chebwin', 'cheb', 'exponential', 'poisson', 'tukey',
                   'tuk']:
        assert_raises(ValueError, signal.get_window, winstr, 7)
