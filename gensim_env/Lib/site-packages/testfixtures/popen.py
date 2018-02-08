from itertools import chain
from subprocess import Popen as Popen, STDOUT, PIPE
from tempfile import TemporaryFile
from testfixtures.compat import basestring, PY3, zip_longest
from testfixtures.utils import extend_docstring

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock


class MockPopen(object):
    """
    A specialised mock for testing use of :class:`subprocess.Popen`.
    An instance of this class can be used in place of the
    :class:`subprocess.Popen` and is often inserted where it's needed using
    :func:`mock.patch` or a :class:`Replacer`.
    """

    default_command = None

    def __init__(self):
        self.commands = {}
        self.mock = mock = Mock()
        self.mock.Popen.side_effect = self.Popen
        mock.Popen_instance = Mock(spec=Popen)
        inst = mock.Popen.return_value = mock.Popen_instance
        inst.communicate.side_effect = self.communicate
        inst.wait.side_effect = self.wait
        inst.send_signal.side_effect = self.send_signal
        inst.terminate.side_effect = self.terminate
        inst.kill.side_effect = self.kill
        inst.poll.side_effect = self.poll
        if PY3:
            def __enter__(self):
                return inst
            inst.__enter__ = __enter__

            def __exit__(self, exc_type, exc_val, exc_tb):
                inst.wait()
                for stream in inst.stdout, inst.stderr:
                    stream.close()

            inst.__exit__ = __exit__

    def set_command(self, command, stdout=b'', stderr=b'', returncode=0,
                    pid=1234, poll_count=3):
        """
        Set the behaviour of this mock when it is used to simulate the
        specified command.

        :param command: A string representing the command to be simulated.
        """
        self.commands[command] = (stdout, stderr, returncode, pid, poll_count)

    def set_default(self, stdout=b'', stderr=b'', returncode=0,
                    pid=1234, poll_count=3):
        """
        Set the behaviour of this mock when it is used to simulate commands
        that have no explicit behavior specified using
        :meth:`~MockPopen.set_command`.
        """
        self.default_command = (stdout, stderr, returncode, pid, poll_count)

    def __call__(self, *args, **kw):
        return self.mock.Popen(*args, **kw)

    def Popen(self, args, bufsize=0, executable=None,
              stdin=None, stdout=None, stderr=None,
              preexec_fn=None, close_fds=False, shell=False, cwd=None,
              env=None, universal_newlines=False,
              startupinfo=None, creationflags=0, restore_signals=True,
              start_new_session=False, pass_fds=(), encoding=None, errors=None):

        if isinstance(args, basestring):
            cmd = args
        else:
            cmd = ' '.join(args)

        behaviour = self.commands.get(cmd, self.default_command)
        if behaviour is None:
            raise KeyError('Nothing specified for command %r' % cmd)

        stdout_value, stderr_value, self.returncode, pid, poll = behaviour

        if stderr == STDOUT:
            line_iterator = chain.from_iterable(zip_longest(
                stdout_value.splitlines(True),
                stderr_value.splitlines(True)
            ))
            stdout_value = b''.join(l for l in line_iterator if l)
            stderr_value = None

        self.poll_count = poll
        for name, option, mock_value in (
            ('stdout', stdout, stdout_value),
            ('stderr', stderr, stderr_value)
        ):
            value = None
            if option is PIPE:
                value = TemporaryFile()
                value.write(mock_value)
                value.flush()
                value.seek(0)
            setattr(self.mock.Popen_instance, name, value)

        self.mock.Popen_instance.pid = pid
        self.mock.Popen_instance.returncode = None

        return self.mock.Popen_instance

    def wait(self):
        "Simulate calls to :meth:`subprocess.Popen.wait`"
        self.mock.Popen_instance.returncode = self.returncode
        return self.returncode

    def communicate(self, input=None):
        "Simulate calls to :meth:`subprocess.Popen.communicate`"
        self.wait()
        i = self.mock.Popen_instance
        return (i.stdout and i.stdout.read(),
                i.stderr and i.stderr.read())

    def poll(self):
        "Simulate calls to :meth:`subprocess.Popen.poll`"
        while self.poll_count and self.mock.Popen_instance.returncode is None:
            self.poll_count -= 1
            return None
        # This call to wait() is NOT how poll() behaves in reality.
        # poll() NEVER sets the returncode.
        # The returncode is *only* ever set by process completion.
        # The following is an artifact of the fixture's implementation.
        return self.wait()

    # These are here to check parameter types
    def send_signal(self, signal):
        "Simulate calls to :meth:`subprocess.Popen.send_signal`"
        pass

    def terminate(self):
        "Simulate calls to :meth:`subprocess.Popen.terminate`"
        pass

    def kill(self):
        "Simulate calls to :meth:`subprocess.Popen.kill`"
        pass


set_command_params = """
:param stdout:
    A string representing the simulated content written by the process
    to the stdout pipe.
:param stderr:
    A string representing the simulated content written by the process
    to the stderr pipe.
:param returncode:
    An integer representing the return code of the simulated process.
:param pid:
    An integer representing the process identifier of the simulated
    process. This is useful if you have code the prints out the pids
    of running processes.
:param poll_count:
    Specifies the number of times :meth:`MockPopen.poll` can be
    called before :attr:`MockPopen.returncode` is set and returned
    by :meth:`MockPopen.poll`.
"""


# add the param docs, so we only have one copy of them!
extend_docstring(set_command_params,
                 [MockPopen.set_command, MockPopen.set_default])
