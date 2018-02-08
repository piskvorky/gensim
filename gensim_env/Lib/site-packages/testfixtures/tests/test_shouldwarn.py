from unittest import TestCase

import warnings

from testfixtures import (
    ShouldWarn, compare, ShouldRaise, ShouldNotWarn,
    Comparison as C
)
from testfixtures.compat import PY3, PY_36_PLUS

if PY3:
    warn_module = 'builtins'
else:
    warn_module = 'exceptions'


class ShouldWarnTests(TestCase):

    def test_warn_expected(self):
        with warnings.catch_warnings(record=True) as backstop:
            with ShouldWarn(UserWarning('foo')):
                warnings.warn('foo')
        compare(len(backstop), expected=0)

    def test_warn_not_expected(self):
        with ShouldRaise(AssertionError(
            "sequence not as expected:\n\n"
            "same:\n[]\n\n"
            "expected:\n[]\n\n"
            "actual:\n[UserWarning('foo',)]"
        )):
            with warnings.catch_warnings(record=True) as backstop:
                with ShouldNotWarn():
                    warnings.warn('foo')
        compare(len(backstop), expected=0)

    def test_no_warn_expected(self):
        with ShouldNotWarn():
            pass

    def test_no_warn_not_expected(self):
        with ShouldRaise(AssertionError(
            "sequence not as expected:\n\n"
            "same:\n[]\n\n"
            "expected:\n[\n  <C:"+warn_module+".UserWarning>\n"
            "  args:('foo',)\n  </C>]"
            "\n\nactual:\n[]"
        )):
            with ShouldWarn(UserWarning('foo')):
                pass

    def test_filters_removed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ShouldWarn(UserWarning("foo")):
                warnings.warn('foo')

    def test_multiple_warnings(self):
        with ShouldRaise(AssertionError) as s:
            with ShouldWarn(UserWarning('foo')):
                warnings.warn('foo')
                warnings.warn('bar')
        content = str(s.raised)
        self.assertTrue('foo' in content)
        self.assertTrue('bar' in content)

    def test_minimal_ok(self):
        with ShouldWarn(UserWarning):
            warnings.warn('foo')

    def test_minimal_bad(self):
        with ShouldRaise(AssertionError(
            "sequence not as expected:\n\n"
            "same:\n[]\n\n"
            "expected:\n"
            "[<C(failed):"+warn_module+".DeprecationWarning>wrong type</C>]\n\n"
            "actual:\n[UserWarning('foo',)]"
        )):
            with ShouldWarn(DeprecationWarning):
                warnings.warn('foo')

    def test_maximal_ok(self):
        with ShouldWarn(DeprecationWarning('foo')):
            warnings.warn_explicit(
                'foo', DeprecationWarning, 'bar.py', 42, 'bar_module'
            )

    def test_maximal_bad(self):
        with ShouldRaise(AssertionError(
            "sequence not as expected:\n\n"
            "same:\n[]\n\n"
            "expected:\n[\n"
            "  <C(failed):"+warn_module+".DeprecationWarning>\n"
            "  args:('bar',) != ('foo',)"
            "\n  </C>]\n\n"
            "actual:\n[DeprecationWarning('foo',)]"
        )):
            with ShouldWarn(DeprecationWarning('bar')):
                warnings.warn_explicit(
                    'foo', DeprecationWarning, 'bar.py', 42, 'bar_module'
                )

    def test_maximal_explore(self):
        with ShouldWarn() as recorded:
            warnings.warn_explicit(
                'foo', DeprecationWarning, 'bar.py', 42, 'bar_module'
            )
        compare(len(recorded), expected=1)

        expected_attrs = dict(
            _category_name='DeprecationWarning',
            category=DeprecationWarning,
            file=None,
            filename='bar.py',
            line=None,
            lineno=42,
            message=C(DeprecationWarning('foo')),
        )

        if PY_36_PLUS:
            expected_attrs['source'] = None

        compare(expected=C(warnings.WarningMessage, **expected_attrs),
            actual=recorded[0])
