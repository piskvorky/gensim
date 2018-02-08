import warnings

from testfixtures import Comparison as C, compare


class ShouldWarn(warnings.catch_warnings):
    """
    This context manager is used to assert that warnings are issued
    within the context it is managing.

    :param expected: This should be a sequence made up of one or more elements,
                     each of one of the following types:

                      * A warning class, indicating that the type
                        of the warnings is important but not the
                        parameters it is created with.

                      * A warning instance, indicating that a
                        warning exactly matching the one supplied
                        should have been issued.

                    If no expected warnings are passed, you will need to inspect
                    the contents of the list returned by the context manager.
    """

    _empty_okay = False

    def __init__(self, *expected):
        super(ShouldWarn, self).__init__(record=True)
        self.expected = [C(e) for e in expected]

    def __enter__(self):
        self.recorded = super(ShouldWarn, self).__enter__()
        warnings.simplefilter("always")
        return self.recorded

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(ShouldWarn, self).__exit__(exc_type, exc_val, exc_tb)
        if not self.recorded and self._empty_okay:
            return
        if not self.expected and self.recorded and not self._empty_okay:
            return
        compare(self.expected, actual=[wm.message for wm in self.recorded])


class ShouldNotWarn(ShouldWarn):
    """
    This context manager is used to assert that no warnings are issued
    within the context it is managing.
    """

    _empty_okay = True

    def __init__(self):
        super(ShouldNotWarn, self).__init__()
