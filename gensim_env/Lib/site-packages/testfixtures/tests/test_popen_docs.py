# NB: This file is used in the documentation, if you make changes, ensure
#     you update the line numbers in popen.txt!

from subprocess import Popen, PIPE


def my_func():
    process = Popen('svn ls -R foo', stdout=PIPE, stderr=PIPE, shell=True)
    out, err = process.communicate()
    if process.returncode:
        raise RuntimeError('something bad happened')
    return out

dotted_path = 'testfixtures.tests.test_popen_docs.Popen'

from unittest import TestCase

from .mock import call
from testfixtures import Replacer, ShouldRaise, compare
from testfixtures.popen import MockPopen


class TestMyFunc(TestCase):

    def setUp(self):
        self.Popen = MockPopen()
        self.r = Replacer()
        self.r.replace(dotted_path, self.Popen)
        self.addCleanup(self.r.restore)

    def test_example(self):
        # set up
        self.Popen.set_command('svn ls -R foo', stdout=b'o', stderr=b'e')

        # testing of results
        compare(my_func(), b'o')

        # testing calls were in the right order and with the correct parameters:
        compare([
            call.Popen('svn ls -R foo',
                       shell=True, stderr=PIPE, stdout=PIPE),
            call.Popen_instance.communicate()
            ], Popen.mock.method_calls)

    def test_example_bad_returncode(self):
        # set up
        Popen.set_command('svn ls -R foo', stdout=b'o', stderr=b'e',
                          returncode=1)

        # testing of error
        with ShouldRaise(RuntimeError('something bad happened')):
            my_func()

    def test_communicate_with_input(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        out, err = process.communicate('foo')
        # test call list
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.communicate('foo'),
                ], Popen.mock.method_calls)

    def test_read_from_stdout_and_stderr(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stdout=b'foo', stderr=b'bar')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        compare(process.stdout.read(), b'foo')
        compare(process.stderr.read(), b'bar')
        # test call list
        compare([
                call.Popen('a command', shell=True, stderr=PIPE, stdout=PIPE),
                ], Popen.mock.method_calls)

    def test_wait_and_return_code(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', returncode=3)
        # usage
        process = Popen('a command')
        compare(process.returncode, None)
        # result checking
        compare(process.wait(), 3)
        compare(process.returncode, 3)
        # test call list
        compare([
                call.Popen('a command'),
                call.Popen_instance.wait(),
                ], Popen.mock.method_calls)

    def test_send_signal(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        process.send_signal(0)
        # result checking
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.send_signal(0),
                ], Popen.mock.method_calls)

    def test_poll_until_result(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', returncode=3, poll_count=2)
        # example usage
        process = Popen('a command')
        while process.poll() is None:
            # you'd probably have a sleep here, or go off and
            # do some other work.
            pass
        # result checking
        compare(process.returncode, 3)
        compare([
                call.Popen('a command'),
                call.Popen_instance.poll(),
                call.Popen_instance.poll(),
                call.Popen_instance.poll(),
                ], Popen.mock.method_calls)

    def test_default_behaviour(self):
        # set up
        self.Popen.set_default(stdout=b'o', stderr=b'e')

        # testing of results
        compare(my_func(), b'o')

        # testing calls were in the right order and with the correct parameters:
        compare([
            call.Popen('svn ls -R foo',
                       shell=True, stderr=PIPE, stdout=PIPE),
            call.Popen_instance.communicate()
            ], Popen.mock.method_calls)
