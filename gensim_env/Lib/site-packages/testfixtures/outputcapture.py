import sys

from testfixtures.comparison import compare
from testfixtures.compat import StringIO


class OutputCapture(object):
    """
    A context manager for capturing output to the
    :attr:`sys.stdout` and :attr:`sys.stderr` streams.

    :param separate: If ``True``, ``stdout`` and ``stderr`` will be captured
                     separately and their expected values must be passed to
                     :meth:`~OutputCapture.compare`.

    .. note:: If ``separate`` is passed as ``True``,
              :attr:`OutputCapture.captured` will be an empty string.
    """

    original_stdout = None
    original_stderr = None

    def __init__(self, separate=False):
        self.separate = separate

    def __enter__(self):
        self.output = StringIO()
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

    def disable(self):
        "Disable the output capture if it is enabled."
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def enable(self):
        "Enable the output capture if it is disabled."
        if self.original_stdout is None:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
        if self.separate:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
        else:
            sys.stdout = sys.stderr = self.output

    @property
    def captured(self):
        "A property containing any output that has been captured so far."
        return self.output.getvalue()

    def compare(self, expected='', stdout='', stderr=''):
        """
        Compare the captured output to that expected. If the output is
        not the same, an :class:`AssertionError` will be raised.

        :param expected: A string containing the expected combined output
                         of ``stdout`` and ``stderr``.

        :param stdout: A string containing the expected output to ``stdout``.

        :param stderr: A string containing the expected output to ``stderr``.
        """
        for prefix, _expected, captured in (
                (None, expected, self.captured),
                ('stdout', stdout, self.stdout.getvalue()),
                ('stderr', stderr, self.stderr.getvalue()),
        ):
            compare(_expected.strip(), actual=captured.strip(), prefix=prefix)
