import os
from tempfile import mkdtemp
from unittest import TestCase
from warnings import catch_warnings

from .mock import Mock

from testfixtures import (
    TempDirectory, Replacer, ShouldRaise, compare, OutputCapture
)
from ..compat import Unicode, PY3
from ..rmtree import rmtree

if PY3:
    some_bytes = '\xa3'.encode('utf-8')
    some_text = '\xa3'
else:
    some_bytes = '\xc2\xa3'
    some_text = '\xc2\xa3'.decode('utf-8')


class TestTempDirectory(TestCase):

    def test_cleanup(self):
        d = TempDirectory()
        p = d.path
        assert os.path.exists(p) is True
        p = d.write('something', b'stuff')
        d.cleanup()
        assert os.path.exists(p) is False

    def test_cleanup_all(self):
        d1 = TempDirectory()
        d2 = TempDirectory()

        assert os.path.exists(d1.path) is True
        p1 = d1.path
        assert os.path.exists(d2.path) is True
        p2 = d2.path

        TempDirectory.cleanup_all()

        assert os.path.exists(p1) is False
        assert os.path.exists(p2) is False

    def test_with_statement(self):
        with TempDirectory() as d:
           p = d.path
           assert os.path.exists(p) is True
           d.write('something', b'stuff')
           assert os.listdir(p) == ['something']
           with open(os.path.join(p, 'something')) as f:
               assert f.read() == 'stuff'
        assert os.path.exists(p) is False

    def test_listdir_sort(self):  # pragma: no branch
        with TempDirectory() as d:
            d.write('ga', b'')
            d.write('foo1', b'')
            d.write('Foo2', b'')
            d.write('g.o', b'')
            with OutputCapture() as output:
                d.listdir()
        output.compare('Foo2\nfoo1\ng.o\nga')


class TempDirectoryTests(TestCase):

    def test_write_with_slash_at_start(self):
        with TempDirectory() as d:
            with ShouldRaise(ValueError(
                    'Attempt to read or write outside the temporary Directory'
                    )):
                d.write('/some/folder', 'stuff')

    def test_makedir_with_slash_at_start(self):
        with TempDirectory() as d:
            with ShouldRaise(ValueError(
                    'Attempt to read or write outside the temporary Directory'
                    )):
                d.makedir('/some/folder')

    def test_read_with_slash_at_start(self):
        with TempDirectory() as d:
            with ShouldRaise(ValueError(
                    'Attempt to read or write outside the temporary Directory'
                    )):
                d.read('/some/folder')

    def test_listdir_with_slash_at_start(self):
        with TempDirectory() as d:
            with ShouldRaise(ValueError(
                    'Attempt to read or write outside the temporary Directory'
                    )):
                d.listdir('/some/folder')

    def test_compare_with_slash_at_start(self):
        with TempDirectory() as d:
            with ShouldRaise(ValueError(
                    'Attempt to read or write outside the temporary Directory'
                    )):
                d.compare((), path='/some/folder')

    def test_read_with_slash_at_start_ok(self):
        with TempDirectory() as d:
            path = d.write('foo', b'bar')
            compare(d.read(path), b'bar')

    def test_dont_cleanup_with_path(self):
        d = mkdtemp()
        fp = os.path.join(d, 'test')
        with open(fp, 'w') as f:
            f.write('foo')
        try:
            td = TempDirectory(path=d)
            self.assertEqual(d, td.path)
            td.cleanup()
            # checks
            self.assertEqual(os.listdir(d), ['test'])
            with open(fp) as f:
                self.assertEqual(f.read(), 'foo')
        finally:
            rmtree(d)

    def test_dont_create_with_path(self):
        d = mkdtemp()
        rmtree(d)
        td = TempDirectory(path=d)
        self.assertEqual(d, td.path)
        self.failIf(os.path.exists(d))

    def test_deprecated_check(self):
        with TempDirectory() as d:
            d.write('x', b'')
            d.check('x')

    def test_deprecated_check_dir(self):
        with TempDirectory() as d:
            d.write('foo/x', b'')
            d.check_dir('foo', 'x')

    def test_deprecated_check_all(self):
        with TempDirectory() as d:
            d.write('a/b/c', b'')
            d.check_all('', 'a/', 'a/b/', 'a/b/c')
            d.check_all('a', 'b/', 'b/c')

    def test_compare_sort_actual(self):
        with TempDirectory() as d:
            d.write('ga', b'')
            d.write('foo1', b'')
            d.write('Foo2', b'')
            d.write('g.o', b'')
            d.compare(['Foo2', 'foo1', 'g.o', 'ga'])

    def test_compare_sort_expected(self):
        with TempDirectory() as d:
            d.write('ga', b'')
            d.write('foo1', b'')
            d.write('Foo2', b'')
            d.write('g.o', b'')
            d.compare(['Foo2', 'ga', 'foo1', 'g.o'])

    def test_compare_path_tuple(self):
        with TempDirectory() as d:
            d.write('a/b/c', b'')
            d.compare(path=('a', 'b'),
                      expected=['c'])

    def test_recursive_ignore(self):
        with TempDirectory(ignore=['.svn']) as d:
            d.write('.svn/rubbish', b'')
            d.write('a/.svn/rubbish', b'')
            d.write('a/b/.svn', b'')
            d.write('a/b/c', b'')
            d.write('a/d/.svn/rubbish', b'')
            d.compare([
                'a/',
                'a/b/',
                'a/b/c',
                'a/d/',
                ])

    def test_files_only(self):
        with TempDirectory() as d:
            d.write('a/b/c', b'')
            d.compare(['a/b/c'], files_only=True)

    def test_path(self):
        with TempDirectory() as d:
            expected1 = d.makedir('foo')
            expected2 = d.write('baz/bob', b'')
            expected3 = d.getpath('a/b/c')

            actual1 = d.getpath('foo')
            actual2 = d.getpath('baz/bob')
            actual3 = d.getpath(('a', 'b', 'c'))

        self.assertEqual(expected1, actual1)
        self.assertEqual(expected2, actual2)
        self.assertEqual(expected3, actual3)

    def test_atexit(self):
        # http://bugs.python.org/issue25532
        from .mock import call

        m = Mock()
        with Replacer() as r:
            # make sure the marker is false, other tests will
            # probably have set it
            r.replace('testfixtures.TempDirectory.atexit_setup', False)
            r.replace('atexit.register', m.register)

            d = TempDirectory()

            expected = [call.register(d.atexit)]

            compare(expected, m.mock_calls)

            with catch_warnings(record=True) as w:
                d.atexit()
                self.assertTrue(len(w), 1)
                compare(str(w[0].message), (  # pragma: no branch
                    "TempDirectory instances not cleaned up by shutdown:\n" +
                    d.path
                    ))

            d.cleanup()

            compare(set(), TempDirectory.instances)

            # check re-running has no ill effects
            d.atexit()

    def test_read_decode(self):
        with TempDirectory() as d:
            with open(os.path.join(d.path, 'test.file'), 'wb') as f:
                f.write(b'\xc2\xa3')
            compare(d.read('test.file', 'utf8'), some_text)

    def test_read_no_decode(self):
        with TempDirectory() as d:
            with open(os.path.join(d.path, 'test.file'), 'wb') as f:
                f.write(b'\xc2\xa3')
            compare(d.read('test.file'), b'\xc2\xa3')

    def test_write_bytes(self):
        with TempDirectory() as d:
            d.write('test.file', b'\xc2\xa3')
            with open(os.path.join(d.path, 'test.file'), 'rb') as f:
                compare(f.read(), b'\xc2\xa3')

    def test_write_unicode(self):
        with TempDirectory() as d:
            d.write('test.file', some_text, 'utf8')
            with open(os.path.join(d.path, 'test.file'), 'rb') as f:
                compare(f.read(), b'\xc2\xa3')

    def test_write_unicode_bad(self):
        if PY3:
            expected = TypeError(
                "a bytes-like object is required, not 'str'"
                )
        else:
            expected = UnicodeDecodeError(
                'ascii', '\xa3', 0, 1, 'ordinal not in range(128)'
                )
        with TempDirectory() as d:
            with ShouldRaise(expected):
                d.write('test.file', Unicode('\xa3'))

    def test_just_empty_non_recursive(self):
        with TempDirectory() as d:
            d.makedir('foo/bar')
            d.makedir('foo/baz')
            d.compare(path='foo',
                      expected=['bar', 'baz'],
                      recursive=False)

    def test_just_empty_dirs(self):
        with TempDirectory() as d:
            d.makedir('foo/bar')
            d.makedir('foo/baz')
            d.compare(['foo/', 'foo/bar/', 'foo/baz/'])

    def test_symlink(self):
        with TempDirectory() as d:
            d.write('foo/bar.txt', b'x')
            os.symlink(d.getpath('foo'), d.getpath('baz'))
            d.compare(['baz/', 'foo/', 'foo/bar.txt'])

    def test_follow_symlinks(self):
        with TempDirectory() as d:
            d.write('foo/bar.txt', b'x')
            os.symlink(d.getpath('foo'), d.getpath('baz'))
            d.compare(['baz/', 'baz/bar.txt', 'foo/', 'foo/bar.txt'],
                      followlinks=True)

    def test_trailing_slash(self):
        with TempDirectory() as d:
            d.write('source/foo/bar.txt', b'x')
            d.compare(path='source/', expected=['foo/', 'foo/bar.txt'])

    def test_default_encoding(self):
        encoded = b'\xc2\xa3'
        decoded = encoded.decode('utf-8')
        with TempDirectory(encoding='utf-8') as d:
            d.write('test.txt', decoded)
            compare(d.read('test.txt'), expected=decoded)

    def test_override_default_encoding(self):
        encoded = b'\xc2\xa3'
        decoded = encoded.decode('utf-8')
        with TempDirectory(encoding='ascii') as d:
            d.write('test.txt', decoded, encoding='utf-8')
            compare(d.read('test.txt', encoding='utf-8'), expected=decoded)
