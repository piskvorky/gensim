from functools import wraps
from testfixtures import Comparison

param_docs = """

    :param exception: This can be one of the following:

                      * `None`, indicating that an exception must be
                        raised, but the type is unimportant.

                      * An exception class, indicating that the type
                        of the exception is important but not the
                        parameters it is created with.

                      * An exception instance, indicating that an
                        exception exactly matching the one supplied
                        should be raised.

    :param unless: Can be passed a boolean that, when ``True`` indicates that
                   no exception is expected. This is useful when checking
                   that exceptions are only raised on certain versions of
                   Python.
"""


class ShouldRaise(object):
    __doc__ = """
    This context manager is used to assert that an exception is raised
    within the context it is managing.
    """ + param_docs

    #: The exception captured by the context manager.
    #: Can be used to inspect specific attributes of the exception.
    raised = None

    def __init__(self, exception=None, unless=False):
        self.exception = exception
        self.expected = not unless

    def __enter__(self):
        return self

    def __exit__(self, type, actual, traceback):

        __tracebackhide__ = True
        
        # bug in python :-(
        if type is not None and not isinstance(actual, type):
            # fixed in 2.7 onwards!
            actual = type(actual)  # pragma: no cover

        self.raised = actual

        if self.expected:
            if self.exception:
                comparison = Comparison(self.exception)
                if comparison != actual:
                    repr_actual = repr(actual)
                    repr_expected = repr(self.exception)
                    message = '%s raised, %s expected' % (
                        repr_actual, repr_expected
                    )
                    if repr_actual == repr_expected:
                        extra = [', attributes differ:']
                        extra.extend(str(comparison).split('\n')[2:-1])
                        message += '\n'.join(extra)
                    raise AssertionError(message)

            elif not actual:
                raise AssertionError('No exception raised!')
        elif actual:
            raise AssertionError('%r raised, no exception expected' % actual)

        return True


class should_raise:
    __doc__ = """
    A decorator to assert that the decorated function will raised
    an exception. An exception class or exception instance may be
    passed to check more specifically exactly what exception will be
    raised.
    """ + param_docs

    def __init__(self, exception=None, unless=None):
        self.exception = exception
        self.unless = unless

    def __call__(self, target):

        @wraps(target)
        def _should_raise_wrapper(*args, **kw):
            with ShouldRaise(self.exception, self.unless):
                target(*args, **kw)

        return _should_raise_wrapper
