"""The suite of window functions."""
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from scipy import fftpack, linalg, special
from scipy._lib.six import string_types

__all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
           'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
           'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'chebwin',
           'slepian', 'cosine', 'hann', 'exponential', 'tukey', 'get_window']


def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


def _cos_win(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window

    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_

    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)

    where

    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1

    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:

    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal._cos_win(1000, HFT90D, sym=False)
    >>> plt.plot(window)
    >>> plt.title("HFT90D window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 10000) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    >>> plt.title("Frequency response of the HFT90D window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axhline(-90.2, color='red')

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return _truncate(w, needs_trunc)


def boxcar(M, sym=True):
    """Return a boxcar or rectangular window.

    Also known as a rectangular window or Dirichlet window, this is equivalent
    to no window at all.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        Whether the window is symmetric. (Has no effect for boxcar.)

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.boxcar(51)
    >>> plt.plot(window)
    >>> plt.title("Boxcar window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the boxcar window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = np.ones(M, float)

    return _truncate(w, needs_trunc)


def triang(M, sym=True):
    """Return a triangular window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    See Also
    --------
    bartlett : A triangular window that touches zero

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.triang(51)
    >>> plt.plot(window)
    >>> plt.title("Triangular window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the triangular window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(1, (M + 1) // 2 + 1)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = np.r_[w, w[::-1]]
    else:
        w = 2 * n / (M + 1.0)
        w = np.r_[w, w[-2::-1]]

    return _truncate(w, needs_trunc)


def parzen(M, sym=True):
    """Return a Parzen window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] E. Parzen, "Mathematical Considerations in the Estimation of
           Spectra", Technometrics,  Vol. 3, No. 2 (May, 1961), pp. 167-190

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.parzen(51)
    >>> plt.plot(window)
    >>> plt.title("Parzen window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Parzen window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0 + 0.5, 1.0)
    na = np.extract(n < -(M - 1) / 4.0, n)
    nb = np.extract(abs(n) <= (M - 1) / 4.0, n)
    wa = 2 * (1 - np.abs(na) / (M / 2.0)) ** 3.0
    wb = (1 - 6 * (np.abs(nb) / (M / 2.0)) ** 2.0 +
          6 * (np.abs(nb) / (M / 2.0)) ** 3.0)
    w = np.r_[wa, wb, wa[::-1]]

    return _truncate(w, needs_trunc)


def bohman(M, sym=True):
    """Return a Bohman window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.bohman(51)
    >>> plt.plot(window)
    >>> plt.title("Bohman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bohman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = np.abs(np.linspace(-1, 1, M)[1:-1])
    w = (1 - fac) * np.cos(np.pi * fac) + 1.0 / np.pi * np.sin(np.pi * fac)
    w = np.r_[0, w, 0]

    return _truncate(w, needs_trunc)


def blackman(M, sym=True):
    r"""
    Return a Blackman window.

    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)

    The "exact Blackman" window was designed to null out the third and fourth
    sidelobes, but has discontinuities at the boundaries, resulting in a
    6 dB/oct fall-off.  This window is an approximation of the "exact" window,
    which does not null the sidelobes as well, but is smooth at the edges,
    improving the fall-off rate to 18 dB/oct. [3]_

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the Kaiser window.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.blackman(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's blackman function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cos_win(M, [0.42, 0.50, 0.08])

    return _truncate(w, needs_trunc)


def nuttall(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window according to Nuttall.

    This variation is called "Nuttall4c" by Heinzel. [2]_

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.nuttall(51)
    >>> plt.plot(window)
    >>> plt.title("Nuttall window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Nuttall window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cos_win(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411])

    return _truncate(w, needs_trunc)


def blackmanharris(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.blackmanharris(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cos_win(M, [0.35875, 0.48829, 0.14128, 0.01168])

    return _truncate(w, needs_trunc)


def flattop(M, sym=True):
    """Return a flat top window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    Flat top windows are used for taking accurate measurements of signal
    amplitude in the frequency domain, with minimal scalloping error from the
    center of a frequency bin to its edges, compared to others.  This is a
    5th-order cosine window, with the 5 terms optimized to make the main lobe
    maximally flat. [1]_

    References
    ----------
    .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
           Measurement Systems", Springer Media, 2006, p. 70
           :doi:`10.1007/0-387-28666-7`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.flattop(51)
    >>> plt.plot(window)
    >>> plt.title("Flat top window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the flat top window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    w = _cos_win(M, a)

    return _truncate(w, needs_trunc)


def bartlett(M, sym=True):
    r"""
    Return a Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The triangular window, with the first and last samples equal to zero
        and the maximum value normalized to 1 (though the value 1 does not
        appear if `M` is even and `sym` is True).

    See Also
    --------
    triang : A triangular window that does not touch zero at the ends

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \frac{2}{M-1} \left(
              \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
              \right)

    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The Fourier transform of the Bartlett is the product
    of two sinc functions.
    Note the excellent discussion in Kanasewich. [2]_

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.bartlett(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's bartlett function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    w = np.where(np.less_equal(n, (M - 1) / 2.0),
                 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))

    return _truncate(w, needs_trunc)


def hann(M, sym=True):
    r"""
    Return a Hann window.

    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hann window is defined as

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The window was named for Julius von Hann, an Austrian meteorologist. It is
    also known as the Cosine Bell. It is sometimes erroneously referred to as
    the "Hanning" window, from the use of "hann" as a verb in the original
    paper and confusion with the very similar Hamming window.

    Most references to the Hann window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.hann(51)
    >>> plt.plot(window)
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hanning function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cos_win(M, [0.5, 0.5])

    return _truncate(w, needs_trunc)


hanning = hann


def tukey(M, alpha=0.5, sym=True):
    r"""Return a Tukey window, also known as a tapered cosine window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function#Tukey_window

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.tukey(51)
    >>> plt.plot(window)
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Tukey window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)

    if alpha <= 0:
        return np.ones(M, 'd')
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    w = np.concatenate((w1, w2, w3))

    return _truncate(w, needs_trunc)


def barthann(M, sym=True):
    """Return a modified Bartlett-Hann window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.barthann(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett-Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett-Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    fac = np.abs(n / (M - 1.0) - 0.5)
    w = 0.62 - 0.48 * fac + 0.38 * np.cos(2 * np.pi * fac)

    return _truncate(w, needs_trunc)


def hamming(M, sym=True):
    r"""Return a Hamming window.

    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
    is described in Blackman and Tukey. It was recommended for smoothing the
    truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.hamming(51)
    >>> plt.plot(window)
    >>> plt.title("Hamming window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hamming window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hamming function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cos_win(M, [0.54, 0.46])

    return _truncate(w, needs_trunc)


def kaiser(M, beta, sym=True):
    r"""Return a Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66,
           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.kaiser(51, beta=14)
    >>> plt.plot(window)
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's kaiser function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    alpha = (M - 1) / 2.0
    w = (special.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(beta))

    return _truncate(w, needs_trunc)


def gaussian(M, std, sym=True):
    r"""Return a Gaussian window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    std : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.gaussian(51, std=7)
    >>> plt.plot(window)
    >>> plt.title(r"Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)

    return _truncate(w, needs_trunc)


def general_gaussian(M, p, sig, sym=True):
    r"""Return a window with a generalized Gaussian shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    p : float
        Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is
        the same shape as the Laplace distribution.
    sig : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left|\frac{n}{\sigma}\right|^{2p} }

    the half-power point is at

    .. math::  (2 \log(2))^{1/(2 p)} \sigma

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.general_gaussian(51, p=1.5, sig=7)
    >>> plt.plot(window)
    >>> plt.title(r"Generalized Gaussian window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Freq. resp. of the gen. Gaussian "
    ...           "window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M) - (M - 1.0) / 2.0
    w = np.exp(-0.5 * np.abs(n / sig) ** (2 * p))

    return _truncate(w, needs_trunc)


# `chebwin` contributed by Kumar Appaiah.
def chebwin(M, at, sym=True):
    r"""Return a Dolph-Chebyshev window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    at : float
        Attenuation (in dB).
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value always normalized to 1

    Notes
    -----
    This window optimizes for the narrowest main lobe width for a given order
    `M` and sidelobe equiripple attenuation `at`, using Chebyshev
    polynomials.  It was originally developed by Dolph to optimize the
    directionality of radio antenna arrays.

    Unlike most windows, the Dolph-Chebyshev is defined in terms of its
    frequency response:

    .. math:: W(k) = \frac
              {\cos\{M \cos^{-1}[\beta \cos(\frac{\pi k}{M})]\}}
              {\cosh[M \cosh^{-1}(\beta)]}

    where

    .. math:: \beta = \cosh \left [\frac{1}{M}
              \cosh^{-1}(10^\frac{A}{20}) \right ]

    and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).

    The time domain window is then generated using the IFFT, so
    power-of-two `M` are the fastest to generate, and prime number `M` are
    the slowest.

    The equiripple condition in the frequency domain creates impulses in the
    time domain, which appear at the ends of the window.

    References
    ----------
    .. [1] C. Dolph, "A current distribution for broadside arrays which
           optimizes the relationship between beam width and side-lobe level",
           Proceedings of the IEEE, Vol. 34, Issue 6
    .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
           American Meteorological Society (April 1997)
           http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
    .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
           No. 1, January 1978

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.chebwin(51, at=100)
    >>> plt.plot(window)
    >>> plt.title("Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if np.abs(at) < 45:
        warnings.warn("This window is not suitable for spectral analysis "
                      "for attenuation values lower than about 45dB because "
                      "the equivalent noise bandwidth of a Chebyshev window "
                      "does not grow monotonically with increasing sidelobe "
                      "attenuation when the attenuation is smaller than "
                      "about 45 dB.")
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    # compute the parameter beta
    order = M - 1.0
    beta = np.cosh(1.0 / order * np.arccosh(10 ** (np.abs(at) / 20.)))
    k = np.r_[0:M] * 1.0
    x = beta * np.cos(np.pi * k / M)
    # Find the window's DFT coefficients
    # Use analytic definition of Chebyshev polynomial instead of expansion
    # from scipy.special. Using the expansion in scipy.special leads to errors.
    p = np.zeros(x.shape)
    p[x > 1] = np.cosh(order * np.arccosh(x[x > 1]))
    p[x < -1] = (1 - 2 * (order % 2)) * np.cosh(order * np.arccosh(-x[x < -1]))
    p[np.abs(x) <= 1] = np.cos(order * np.arccos(x[np.abs(x) <= 1]))

    # Appropriate IDFT and filling up
    # depending on even/odd M
    if M % 2:
        w = np.real(fftpack.fft(p))
        n = (M + 1) // 2
        w = w[:n]
        w = np.concatenate((w[n - 1:0:-1], w))
    else:
        p = p * np.exp(1.j * np.pi / M * np.r_[0:M])
        w = np.real(fftpack.fft(p))
        n = M // 2 + 1
        w = np.concatenate((w[n - 1:0:-1], w[1:n]))
    w = w / max(w)

    return _truncate(w, needs_trunc)


def slepian(M, width, sym=True):
    """Return a digital Slepian (DPSS) window.

    Used to maximize the energy concentration in the main lobe.  Also called
    the digital prolate spheroidal sequence (DPSS).

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    width : float
        Bandwidth
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value always normalized to 1

    References
    ----------
    .. [1] D. Slepian & H. O. Pollak: "Prolate spheroidal wave functions,
           Fourier analysis and uncertainty-I," Bell Syst. Tech. J., vol.40,
           pp.43-63, 1961. https://archive.org/details/bstj40-1-43
    .. [2] H. J. Landau & H. O. Pollak: "Prolate spheroidal wave functions,
           Fourier analysis and uncertainty-II," Bell Syst. Tech. J. , vol.40,
           pp.65-83, 1961. https://archive.org/details/bstj40-1-65

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.slepian(51, width=0.3)
    >>> plt.plot(window)
    >>> plt.title("Slepian (DPSS) window (BW=0.3)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Slepian window (BW=0.3)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    # our width is the full bandwidth
    width = width / 2
    # to match the old version
    width = width / 2
    m = np.arange(M, dtype='d')
    H = np.zeros((2, M))
    H[0, 1:] = m[1:] * (M - m[1:]) / 2
    H[1, :] = ((M - 1 - 2 * m) / 2)**2 * np.cos(2 * np.pi * width)

    _, win = linalg.eig_banded(H, select='i', select_range=(M-1, M-1))
    win = win.ravel() / win.max()

    return _truncate(win, needs_trunc)


def cosine(M, sym=True):
    """Return a window with a simple cosine shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.cosine(51)
    >>> plt.plot(window)
    >>> plt.title("Cosine window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the cosine window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.show()

    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = np.sin(np.pi / M * (np.arange(0, M) + .5))

    return _truncate(w, needs_trunc)


def exponential(M, center=None, tau=1., sym=True):
    r"""Return an exponential (or Poisson) window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    center : float, optional
        Parameter defining the center location of the window function.
        The default value if not given is ``center = (M-1) / 2``.  This
        parameter must take its default value for symmetric windows.
    tau : float, optional
        Parameter defining the decay.  For ``center = 0`` use
        ``tau = -(M-1) / ln(x)`` if ``x`` is the fraction of the window
        remaining at the end.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Exponential window is defined as

    .. math::  w(n) = e^{-|n-center| / \tau}

    References
    ----------
    S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",
    Technical Review 3, Bruel & Kjaer, 1987.

    Examples
    --------
    Plot the symmetric window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> M = 51
    >>> tau = 3.0
    >>> window = signal.exponential(M, tau=tau)
    >>> plt.plot(window)
    >>> plt.title("Exponential Window (tau=3.0)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -35, 0])
    >>> plt.title("Frequency response of the Exponential window (tau=3.0)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    This function can also generate non-symmetric windows:

    >>> tau2 = -(M-1) / np.log(0.01)
    >>> window2 = signal.exponential(M, 0, tau2, False)
    >>> plt.figure()
    >>> plt.plot(window2)
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    """
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    if center is None:
        center = (M-1) / 2

    n = np.arange(0, M)
    w = np.exp(-np.abs(n-center) / tau)

    return _truncate(w, needs_trunc)


_win_equiv_raw = {
    ('barthann', 'brthan', 'bth'): (barthann, False),
    ('bartlett', 'bart', 'brt'): (bartlett, False),
    ('blackman', 'black', 'blk'): (blackman, False),
    ('blackmanharris', 'blackharr', 'bkh'): (blackmanharris, False),
    ('bohman', 'bman', 'bmn'): (bohman, False),
    ('boxcar', 'box', 'ones',
        'rect', 'rectangular'): (boxcar, False),
    ('chebwin', 'cheb'): (chebwin, True),
    ('cosine', 'halfcosine'): (cosine, False),
    ('exponential', 'poisson'): (exponential, True),
    ('flattop', 'flat', 'flt'): (flattop, False),
    ('gaussian', 'gauss', 'gss'): (gaussian, True),
    ('general gaussian', 'general_gaussian',
        'general gauss', 'general_gauss', 'ggs'): (general_gaussian, True),
    ('hamming', 'hamm', 'ham'): (hamming, False),
    ('hanning', 'hann', 'han'): (hann, False),
    ('kaiser', 'ksr'): (kaiser, True),
    ('nuttall', 'nutl', 'nut'): (nuttall, False),
    ('parzen', 'parz', 'par'): (parzen, False),
    ('slepian', 'slep', 'optimal', 'dpss', 'dss'): (slepian, True),
    ('triangle', 'triang', 'tri'): (triang, False),
    ('tukey', 'tuk'): (tukey, True),
}

# Fill dict with all valid window name strings
_win_equiv = {}
for k, v in _win_equiv_raw.items():
    for key in k:
        _win_equiv[key] = v[0]

# Keep track of which windows need additional parameters
_needs_param = set()
for k, v in _win_equiv_raw.items():
    if v[1]:
        _needs_param.update(k)


def get_window(window, Nx, fftbins=True):
    """
    Return a window.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        `fftpack.fftfreq`).
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`

    Notes
    -----
    Window types:

        `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
        `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,
        `barthann`, `kaiser` (needs beta), `gaussian` (needs standard
        deviation), `general_gaussian` (needs power, width), `slepian`
        (needs width), `chebwin` (needs attenuation), `exponential`
        (needs decay scale), `tukey` (needs taper fraction)

    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> from scipy import signal
    >>> signal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> signal.get_window(('kaiser', 4.0), 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])
    >>> signal.get_window(4.0, 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])

    """
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError):
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif isinstance(window, string_types):
            if window in _needs_param:
                raise ValueError("The '" + window + "' window needs one or "
                                 "more parameters -- pass a tuple.")
            else:
                winstr = window
        else:
            raise ValueError("%s as window type is not supported." %
                             str(type(window)))

        try:
            winfunc = _win_equiv[winstr]
        except KeyError:
            raise ValueError("Unknown window type.")

        params = (Nx,) + args + (sym,)
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)

    return winfunc(*params)
