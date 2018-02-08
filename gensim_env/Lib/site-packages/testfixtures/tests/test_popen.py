from subprocess import PIPE, STDOUT
from unittest import TestCase

from .mock import call
from testfixtures import ShouldRaise, compare

from testfixtures.popen import MockPopen
from testfixtures.compat import PY2

import signal


class Tests(TestCase):

    def test_command_min_args(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE)
        # process started, no return code
        compare(process.pid, 1234)
        compare(None, process.returncode)

        out, err = process.communicate()

        # test the rest
        compare(out, b'')
        compare(err, b'')
        compare(process.returncode, 0)
        # test call list
        compare([
                call.Popen('a command', stderr=-1, stdout=-1),
                call.Popen_instance.communicate(),
                ], Popen.mock.method_calls)

    def test_command_max_args(self):

        Popen = MockPopen()
        Popen.set_command('a command', b'out', b'err', 1, 345)

        process = Popen('a command', stdout=PIPE, stderr=PIPE)
        compare(process.pid, 345)
        compare(None, process.returncode)

        out, err = process.communicate()

        # test the rest
        compare(out, b'out')
        compare(err, b'err')
        compare(process.returncode, 1)
        # test call list
        compare([
                call.Popen('a command', stderr=-1, stdout=-1),
                call.Popen_instance.communicate(),
                ], Popen.mock.method_calls)

    def test_command_is_sequence(self):
        Popen = MockPopen()
        Popen.set_command('a command')

        process = Popen(['a', 'command'], stdout=PIPE, stderr=PIPE)

        compare(process.wait(), 0)
        compare([
                call.Popen(['a', 'command'], stderr=-1, stdout=-1),
                call.Popen_instance.wait(),
                ], Popen.mock.method_calls)

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

    def test_read_from_stdout(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stdout=b'foo')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        self.assertTrue(isinstance(process.stdout.fileno(), int))
        compare(process.stdout.read(), b'foo')
        # test call list
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                ], Popen.mock.method_calls)

    def test_read_from_stderr(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stderr=b'foo')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        self.assertTrue(isinstance(process.stdout.fileno(), int))
        compare(process.stderr.read(), b'foo')
        # test call list
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                ], Popen.mock.method_calls)

    def test_read_from_stdout_with_stderr_redirected_check_stdout_contents(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stdout=b'foo', stderr=b'bar')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=STDOUT, shell=True)
        # test stdout contents
        compare(b'foobar', process.stdout.read())
        compare(process.stderr, None)

    def test_read_from_stdout_with_stderr_redirected_check_stdout_stderr_interleaved(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stdout=b'o1\no2\no3\no4\n', stderr=b'e1\ne2\n')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=STDOUT, shell=True)
        self.assertTrue(isinstance(process.stdout.fileno(), int))
        # test stdout contents
        compare(b'o1\ne1\no2\ne2\no3\no4\n', process.stdout.read())

    def test_communicate_with_stderr_redirected_check_stderr_is_none(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', stdout=b'foo', stderr=b'bar')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=STDOUT, shell=True)
        out, err = process.communicate()
        # test stderr is None
        compare(out, b'foobar')
        compare(err, None)

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

    def test_multiple_uses(self):
        Popen = MockPopen()
        Popen.set_command('a command', b'a')
        Popen.set_command('b command', b'b')
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        out, err = process.communicate('foo')
        compare(out, b'a')
        process = Popen(['b', 'command'], stdout=PIPE, stderr=PIPE, shell=True)
        out, err = process.communicate('foo')
        compare(out, b'b')
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.communicate('foo'),
                call.Popen(['b', 'command'], shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.communicate('foo'),
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

    def test_terminate(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        process.terminate()
        # result checking
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.terminate(),
                ], Popen.mock.method_calls)

    def test_kill(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        process.kill()
        # result checking
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.kill(),
                ], Popen.mock.method_calls)

    def test_all_signals(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command')
        process.send_signal(signal.SIGINT)
        process.terminate()
        process.kill()
        # test call list
        compare([
                call.Popen('a command'),
                call.Popen_instance.send_signal(signal.SIGINT),
                call.Popen_instance.terminate(),
                call.Popen_instance.kill(),
                ], Popen.mock.method_calls)

    def test_poll_no_setup(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        compare(process.poll(), None)
        compare(process.poll(), None)
        compare(process.wait(), 0)
        compare(process.poll(), 0)
        # result checking
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.poll(),
                call.Popen_instance.poll(),
                call.Popen_instance.wait(),
                call.Popen_instance.poll(),
                ], Popen.mock.method_calls)

    def test_poll_setup(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command', poll_count=1)
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)
        compare(process.poll(), None)
        compare(process.poll(), 0)
        compare(process.wait(), 0)
        compare(process.poll(), 0)
        # result checking
        compare([
                call.Popen('a command', shell=True, stderr=-1, stdout=-1),
                call.Popen_instance.poll(),
                call.Popen_instance.poll(),
                call.Popen_instance.wait(),
                call.Popen_instance.poll(),
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

    def test_command_not_specified(self):
        Popen = MockPopen()
        with ShouldRaise(KeyError(
            "Nothing specified for command 'a command'"
        )):
            Popen('a command', stdout=PIPE, stderr=PIPE, shell=True)

    def test_default_command_min_args(self):
        # setup
        Popen = MockPopen()
        Popen.set_default()
        # usage
        process = Popen('a command', stdout=PIPE, stderr=PIPE)
        # process started, no return code
        compare(process.pid, 1234)
        compare(None, process.returncode)

        out, err = process.communicate()

        # test the rest
        compare(out, b'')
        compare(err, b'')
        compare(process.returncode, 0)
        # test call list
        compare([
            call.Popen('a command', stderr=-1, stdout=-1),
            call.Popen_instance.communicate(),
        ], Popen.mock.method_calls)

    def test_default_command_max_args(self):
        Popen = MockPopen()
        Popen.set_default(b'out', b'err', 1, 345)

        process = Popen('a command', stdout=PIPE, stderr=PIPE)
        compare(process.pid, 345)
        compare(None, process.returncode)

        out, err = process.communicate()

        # test the rest
        compare(out, b'out')
        compare(err, b'err')
        compare(process.returncode, 1)
        # test call list
        compare([
            call.Popen('a command', stderr=-1, stdout=-1),
            call.Popen_instance.communicate(),
        ], Popen.mock.method_calls)

    def test_invalid_parameters(self):
        Popen = MockPopen()
        with ShouldRaise(TypeError(
                "Popen() got an unexpected keyword argument 'foo'"
        )):
            Popen(foo='bar')

    def test_invalid_method_or_attr(self):
        Popen = MockPopen()
        Popen.set_command('command')
        process = Popen('command')
        with ShouldRaise(
                AttributeError("Mock object has no attribute 'foo'")):
            process.foo()

    def test_invalid_attribute(self):
        Popen = MockPopen()
        Popen.set_command('command')
        process = Popen('command')
        with ShouldRaise(AttributeError("Mock object has no attribute 'foo'")):
            process.foo

    def test_invalid_communicate_call(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        with ShouldRaise(TypeError(
                "communicate() got an unexpected keyword argument 'foo'"
        )):
            process.communicate(foo='bar')

    def test_invalid_wait_call(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        with ShouldRaise(TypeError(
                "wait() got an unexpected keyword argument 'foo'"
        )):
            process.wait(foo='bar')

    def test_invalid_send_signal(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        with ShouldRaise(TypeError(
                "send_signal() got an unexpected keyword argument 'foo'"
        )):
            process.send_signal(foo='bar')

    def test_invalid_terminate(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        with ShouldRaise(TypeError(
                "terminate() got an unexpected keyword argument 'foo'"
        )):
            process.terminate(foo='bar')

    def test_invalid_kill(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        if PY2:
            text = 'kill() takes exactly 1 argument (2 given)'
        else:
            text = 'kill() takes 1 positional argument but 2 were given'
        with ShouldRaise(TypeError(text)):
            process.kill('moo')

    def test_invalid_poll(self):
        Popen = MockPopen()
        Popen.set_command('bar')
        process = Popen('bar')
        if PY2:
            text = 'poll() takes exactly 1 argument (2 given)'
        else:
            text = 'poll() takes 1 positional argument but 2 were given'
        with ShouldRaise(TypeError(text)):
            process.poll('moo')

    def test_non_pipe(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        process = Popen('a command')
        # checks
        compare(process.stdout, expected=None)
        compare(process.stderr, expected=None)
        out, err = process.communicate()
        # test the rest
        compare(out, expected=None)
        compare(err, expected=None)
        # test call list
        compare([
                call.Popen('a command'),
                call.Popen_instance.communicate(),
                ], Popen.mock.method_calls)

    def test_use_as_context_manager(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        if PY2:

            process = Popen('a command')
            with ShouldRaise(AttributeError):
                process.__enter__
            with ShouldRaise(AttributeError):
                process.__exit__

        else:

            # usage
            with Popen('a command', stdout=PIPE, stderr=PIPE) as process:
                # process started, no return code
                compare(process.pid, 1234)
                compare(None, process.returncode)

                out, err = process.communicate()

            # test the rest
            compare(out, b'')
            compare(err, b'')
            compare(process.returncode, 0)

            compare(process.stdout.closed, expected=True)
            compare(process.stderr.closed, expected=True)

            # test call list
            compare([
                call.Popen('a command', stderr=-1, stdout=-1),
                call.Popen_instance.communicate(),
                call.Popen_instance.wait(),
            ], Popen.mock.method_calls)

    def test_start_new_session(self):
        # setup
        Popen = MockPopen()
        Popen.set_command('a command')
        # usage
        Popen('a command', start_new_session=True)
        # test call list
        compare([
            call.Popen('a command', start_new_session=True),
        ], Popen.mock.method_calls)
