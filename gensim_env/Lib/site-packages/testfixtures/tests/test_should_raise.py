from testfixtures import Comparison as C, ShouldRaise, should_raise
from unittest import TestCase

from ..compat import PY3, PY_36_PLUS


class TestShouldRaise(TestCase):

    def test_no_params(self):
        def to_test():
            raise ValueError('wrong value supplied')
        should_raise(ValueError('wrong value supplied'))(to_test)()

    def test_no_exception(self):
        def to_test():
            pass
        try:
            should_raise(ValueError())(to_test)()
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError('None raised, ValueError() expected'))
                )
        else:
            self.fail('No exception raised!')

    def test_wrong_exception(self):
        def to_test():
            raise ValueError('bar')
        try:
            should_raise(ValueError('foo'))(to_test)()
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError(
                    "ValueError('bar',) raised, ValueError('foo',) expected"
                )))
        else:
            self.fail('No exception raised!')

    def test_only_exception_class(self):
        def to_test():
            raise ValueError('bar')
        should_raise(ValueError)(to_test)()

    def test_no_supplied_or_raised(self):
        # effectvely we're saying "something should be raised!"
        # but we want to inspect s.raised rather than making
        # an up-front assertion
        def to_test():
            pass
        try:
            should_raise()(to_test)()
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError("No exception raised!"))
                )
        else:
            self.fail('No exception raised!')

    def test_args(self):
        def to_test(*args):
            raise ValueError('%s' % repr(args))
        should_raise(ValueError('(1,)'))(to_test)(1)

    def test_kw_to_args(self):
        def to_test(x):
            raise ValueError('%s' % x)
        should_raise(ValueError('1'))(to_test)(x=1)

    def test_kw(self):
        def to_test(**kw):
            raise ValueError('%r' % kw)
        should_raise(ValueError("{'x': 1}"))(to_test)(x=1)

    def test_both(self):
        def to_test(*args, **kw):
            raise ValueError('%r %r' % (args, kw))
        should_raise(ValueError("(1,) {'x': 2}"))(to_test)(1, x=2)

    def test_method_args(self):
        class X:
            def to_test(self, *args):
                self.args = args
                raise ValueError()
        x = X()
        should_raise(ValueError)(x.to_test)(1, 2, 3)
        self.assertEqual(x.args, (1, 2, 3))

    def test_method_kw(self):
        class X:
            def to_test(self, **kw):
                self.kw = kw
                raise ValueError()
        x = X()
        should_raise(ValueError)(x.to_test)(x=1, y=2)
        self.assertEqual(x.kw, {'x': 1, 'y': 2})

    def test_method_both(self):
        class X:
            def to_test(self, *args, **kw):
                self.args = args
                self.kw = kw
                raise ValueError()
        x = X()
        should_raise(ValueError)(x.to_test)(1, y=2)
        self.assertEqual(x.args, (1, ))
        self.assertEqual(x.kw, {'y': 2})

    def test_class_class(self):
        class Test:
            def __init__(self, x):
                # The TypeError is raised due to the mis-matched parameters
                # so the pass never gets executed
                pass  # pragma: no cover
        should_raise(TypeError)(Test)()

    def test_raised(self):
        with ShouldRaise() as s:
            raise ValueError('wrong value supplied')
        self.assertEqual(s.raised, C(ValueError('wrong value supplied')))

    def test_catch_baseexception_1(self):
        with ShouldRaise(SystemExit):
            raise SystemExit()

    def test_catch_baseexception_2(self):
        with ShouldRaise(KeyboardInterrupt):
            raise KeyboardInterrupt()

    def test_with_exception_class_supplied(self):
        with ShouldRaise(ValueError):
            raise ValueError('foo bar')

    def test_with_exception_supplied(self):
        with ShouldRaise(ValueError('foo bar')):
            raise ValueError('foo bar')

    def test_with_exception_supplied_wrong_args(self):
        try:
            with ShouldRaise(ValueError('foo')):
                raise ValueError('bar')
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError(
                    "ValueError('bar',) raised, ValueError('foo',) expected"
                )))
        else:
            self.fail('No exception raised!')

    def test_neither_supplied(self):
        with ShouldRaise():
            raise ValueError('foo bar')

    def test_with_no_exception_when_expected(self):
        try:
            with ShouldRaise(ValueError('foo')):
                pass
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError("None raised, ValueError('foo',) expected"))
                )
        else:
            self.fail('No exception raised!')

    def test_with_no_exception_when_neither_expected(self):
        try:
            with ShouldRaise():
                pass
        except AssertionError as e:
            self.assertEqual(
                e,
                C(AssertionError("No exception raised!"))
                )
        else:
            self.fail('No exception raised!')

    def test_with_getting_raised_exception(self):
        with ShouldRaise() as s:
            raise ValueError('foo bar')
        self.assertEqual(C(ValueError('foo bar')), s.raised)

    def test_import_errors_1(self):
        if PY3:
            message = "No module named 'textfixtures'"
        else:
            message = 'No module named textfixtures.foo.bar'

        exception = ModuleNotFoundError if PY_36_PLUS else ImportError

        with ShouldRaise(exception(message)):
            import textfixtures.foo.bar

    def test_import_errors_2(self):
        with ShouldRaise(ImportError('X')):
            raise ImportError('X')

    def test_custom_exception(self):

        class FileTypeError(Exception):
            def __init__(self, value):
                self.value = value

        with ShouldRaise(FileTypeError('X')):
            raise FileTypeError('X')

    def test_assert_keyerror_raised(self):
        expected = "KeyError('foo',) raised, AttributeError('foo',) expected"

        class Dodgy(dict):
            def __getattr__(self, name):
                # NB: we forgot to turn our KeyError into an attribute error
                return self[name]
        try:
            with ShouldRaise(AttributeError('foo')):
                Dodgy().foo
        except AssertionError as e:
            self.assertEqual(
                C(AssertionError(expected)),
                e
                )
        else:
            self.fail('No exception raised!')

    def test_decorator_usage(self):

        @should_raise(ValueError('bad'))
        def to_test():
            raise ValueError('bad')

        to_test()

    def test_unless_false_okay(self):
        with ShouldRaise(unless=False):
            raise AttributeError()

    def test_unless_false_bad(self):
        try:
            with ShouldRaise(unless=False):
                pass
        except AssertionError as e:
            self.assertEqual(e, C(AssertionError("No exception raised!")))
        else:
            self.fail('No exception raised!')

    def test_unless_true_okay(self):
        with ShouldRaise(unless=True):
            pass

    def test_unless_true_not_okay(self):
        try:
            with ShouldRaise(unless=True):
                raise AttributeError('foo')
        except AssertionError as e:
            self.assertEqual(e, C(AssertionError(
                "AttributeError('foo',) raised, no exception expected"
                )))
        else:
            self.fail('No exception raised!')

    def test_unless_decorator_usage(self):

        @should_raise(unless=True)
        def to_test():
            pass

        to_test()

    def test_identical_reprs(self):
        class AnnoyingException(Exception):
            def __init__(self, **kw):
                self.other = kw.get('other')

        try:
            with ShouldRaise(AnnoyingException(other='bar')):
                raise AnnoyingException(other='baz')
        except AssertionError as e:
            self.assertEqual(
                C(AssertionError(
                    "AnnoyingException() raised, AnnoyingException() expected,"
                    " attributes differ:\n"
                    "  other:'bar' != 'baz'"
                )),
                e,
                )
        else:
            self.fail('No exception raised!')
