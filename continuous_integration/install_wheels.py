"""Install wheels for numpy and scipy.

Without wheels, installation requires doing a build, which is too much.
The versions of the packages for which wheels are available depends on
the current Python version.

We use this when building/testing gensim in a CI environment (Travis, AppVeyor,
etc).

The versions we currently target are:

    - 3.5 (AppVeyor, TravisCI)
    - 3.6 (AppVeyor, TravisCI)
    - 3.7 (AppVeyor, TravisCI, CircleCI)

AppVeyor builds are Windows.
CircleCI builds are Linux, and they build documentation only.
TravisCI builds are Linux and MacOS.

We want to pick numpy and scipy versions that have wheels for the current
Python version and OS.

You can check whether wheels are available for a particular numpy release here::

    https://pypi.org/project/numpy/1.17.4/#files

or by running::

    python continuous_integration/check_wheels.py numpy
"""

import subprocess
import sys


def main():
    #
    # We don't support Py2 anymore, so the most recent versions of both
    # numpy and scipy have what we need.
    #
    packages = ['numpy==1.17.4', 'scipy==1.4.1']
    command = [sys.executable, '-m', 'pip', 'install'] + packages

    print('sys.executable: %r' % sys.executable, file=sys.stderr)
    print('sys.version_info: %r' % list(sys.version_info), file=sys.stderr)
    print('command: %r' % command, file=sys.stderr)

    subprocess.check_call(command)


if __name__ == '__main__':
    main()
