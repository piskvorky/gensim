"""Data structures and functions of general utility,
shared between different modules and variants of the software.
"""

import logging
import math
import random
import sys
import types


LOGPROB_ZERO = 1000000


# Progress bar for generators (length unknown):
# Print a dot for every GENERATOR_DOT_FREQ:th dot.
# Set to <= 0 to disable progress bar.
GENERATOR_DOT_FREQ = 500


show_progress_bar = True


def _progress(iter_func):
    """Decorator/function for displaying a progress bar when iterating
    through a list.

    iter_func can be both a function providing a iterator (for decorator
    style use) or an iterator itself.

    No progressbar is displayed when the show_progress_bar variable is set to
     false.

    If the progressbar module is available a fancy percentage style
    progressbar is displayed. Otherwise 60 dots are printed as indicator.

    """

    if not show_progress_bar:
        return iter_func

    #Try to see or the progressbar module is available, else fabricate our own
    try:
        from progressbar import ProgressBar
    except ImportError:
        class SimpleProgressBar:
            """Create a simple progress bar that prints 60 dots on a single
            line, proportional to the progress """
            NUM_DOTS = 60

            def __call__(self, it):
                self.it = iter(it)
                self.i = 0

                # Dot frequency is determined as ceil(len(it) / NUM_DOTS)
                self.dotfreq = (len(it) + self.NUM_DOTS - 1) // self.NUM_DOTS
                if self.dotfreq < 1:
                    self.dotfreq = 1

                return self

            def __iter__(self):
                return self

            def __next__(self):
                self.i += 1
                if self.i % self.dotfreq == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
                try:
                    return next(self.it)
                except StopIteration:
                    sys.stderr.write('\n')
                    raise

            #Needed to be compatible with both Python2 and 3
            next = __next__

        ProgressBar = SimpleProgressBar

    # In case of a decorator (argument is a function),
    # wrap the functions result in a ProgressBar and return the new function
    if isinstance(iter_func, types.FunctionType):
        def i(*args, **kwargs):
            if logging.getLogger(__name__).isEnabledFor(logging.INFO):
                return ProgressBar()(iter_func(*args, **kwargs))
            else:
                return iter_func(*args, **kwargs)
        return i

    #In case of an iterator, wrap it in a ProgressBar and return it.
    elif hasattr(iter_func, '__iter__'):
        return ProgressBar()(iter_func)

    #If all else fails, just return the original.
    return iter_func


class Sparse(dict):
    """A defaultdict-like data structure, which tries to remain as sparse
    as possible. If a value becomes equal to the default value, it (and the
    key associated with it) are transparently removed.

    Only supports immutable values, e.g. namedtuples.
    """

    def __init__(self, *pargs, **kwargs):
        """Create a new Sparse datastructure.
        Keyword arguments:
            default: Default value. Unlike defaultdict this should be a
                       prototype immutable, not a factory.
        """

        self._default = kwargs.pop('default')
        dict.__init__(self, *pargs, **kwargs)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self._default

    def __setitem__(self, key, value):
        # attribute check is necessary for unpickling
        if '_default' in self and value == self._default:
            if key in self:
                del self[key]
        else:
            dict.__setitem__(self, key, value)


def ngrams(sequence, n=2):
    """Returns all ngram tokens in an input sequence, for a specified n.
    E.g. ngrams(['A', 'B', 'A', 'B', 'D'], n=2) yields
    ('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'D')
    """

    window = []
    for item in sequence:
        window.append(item)
        if len(window) > n:
            # trim back to size
            window = window[-n:]
        if len(window) == n:
            yield(tuple(window))


def minargmin(sequence):
    """Returns the minimum value and the first index at which it can be
    found in the input sequence."""
    best = (None, None)
    for (i, value) in enumerate(sequence):
        if best[0] is None or value < best[0]:
            best = (value, i)
    return best


def zlog(x):
    """Logarithm which uses constant value for log(0) instead of -inf"""
    assert x >= 0.0
    if x == 0:
        return LOGPROB_ZERO
    return -math.log(x)


def _nt_zeros(constructor, zero=0):
    """Convenience function to return a namedtuple initialized to zeros,
    without needing to know the number of fields."""
    zeros = [zero] * len(constructor._fields)
    return constructor(*zeros)


def weighted_sample(data, num_samples):
    """Samples with replacement from the data set so that the probability
    of each data point being selected is proportional to the occurrence count.
    Arguments:
        data: A list of tuples (weight, ...)
        num_samples: The number of samples to return
    Returns:
        a sorted list of indices to data
    """
    tokens = sum(x[0] for x in data)
    token_indices = sorted([random.randint(0, tokens - 1)
                            for _ in range(num_samples)])

    data_indices = []
    d = enumerate(x[0] for x in data)
    di = 0
    ti = -1
    for sample_token_index in token_indices:
        while ti < sample_token_index:
            (di, weight) = d.next()
            ti += weight
        data_indices.append(di)
    return data_indices


def _generator_progress(generator):
    """Prints a progress bar for visualizing flow through a generator.
    The length of a generator is not known in advance, so the bar has
    no fixed length. GENERATOR_DOT_FREQ controls the frequency of dots.

    This function wraps the argument generator, returning a new generator.
    """

    if GENERATOR_DOT_FREQ <= 0:
        return generator

    def _progress_wrapper(generator):
        for (i, x) in enumerate(generator):
            if i % GENERATOR_DOT_FREQ == 0:
                sys.stderr.write('.')
                sys.stderr.flush()
            yield x
        sys.stderr.write('\n')

    return _progress_wrapper(generator)


def _is_string(obj):
    try:
        # Python 2
        return isinstance(obj, basestring)
    except NameError:
        # Python 3
        return isinstance(obj, str)
