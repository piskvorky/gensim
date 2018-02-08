from __future__ import print_function

import sys
from unittest import TestCase

from testfixtures import OutputCapture


class TestOutputCapture(TestCase):

    def test_compare_strips(self):
        with OutputCapture() as o:
            print(' Bar! ')
        o.compare('Bar!')

    def test_stdout_and_stderr(self):
        with OutputCapture() as o:
            print('hello', file=sys.stdout)
            print('out', file=sys.stderr)
            print('there', file=sys.stdout)
            print('now', file=sys.stderr)
        o.compare("hello\nout\nthere\nnow\n")

    def test_unicode(self):
        with OutputCapture() as o:
            print(u'\u65e5', file=sys.stdout)
        o.compare(u'\u65e5\n')

    def test_separate_capture(self):
        with OutputCapture(separate=True) as o:
            print('hello', file=sys.stdout)
            print('out', file=sys.stderr)
            print('there', file=sys.stdout)
            print('now', file=sys.stderr)
        o.compare(stdout="hello\nthere\n",
                  stderr="out\nnow\n")

    def test_original_restore(self):
        o_out, o_err = sys.stdout, sys.stderr
        with OutputCapture() as o:
            self.assertFalse(sys.stdout is o_out)
            self.assertFalse(sys.stderr is o_err)
        self.assertTrue(sys.stdout is o_out)
        self.assertTrue(sys.stderr is o_err)

    def test_double_disable(self):
        o_out, o_err = sys.stdout, sys.stderr
        with OutputCapture() as o:
            self.assertFalse(sys.stdout is o_out)
            self.assertFalse(sys.stderr is o_err)
            o.disable()
            self.assertTrue(sys.stdout is o_out)
            self.assertTrue(sys.stderr is o_err)
            o.disable()
            self.assertTrue(sys.stdout is o_out)
            self.assertTrue(sys.stderr is o_err)
        self.assertTrue(sys.stdout is o_out)
        self.assertTrue(sys.stderr is o_err)

    def test_double_enable(self):
        o_out, o_err = sys.stdout, sys.stderr
        with OutputCapture() as o:
            o.disable()
            self.assertTrue(sys.stdout is o_out)
            self.assertTrue(sys.stderr is o_err)
            o.enable()
            self.assertFalse(sys.stdout is o_out)
            self.assertFalse(sys.stderr is o_err)
            o.enable()
            self.assertFalse(sys.stdout is o_out)
            self.assertFalse(sys.stderr is o_err)
        self.assertTrue(sys.stdout is o_out)
        self.assertTrue(sys.stderr is o_err)
