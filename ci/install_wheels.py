"""Install wheels for numpy and scipy.

Without wheels, installation requires doing a build, which is too much.
The versions of the packages for which wheels are available depends on
the current Python version.

We use this when building/testing gensim in a CI environment (Travis, AppVeyor,
etc).
"""

import subprocess
import sys


def main():
    if sys.version[:3] == (3, 7, 0):
        packages = ['numpy==1.14.5', 'scipy==1.1.0']
    else:
        packages = ['numpy==1.11.3', 'scipy==1.0.0']
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)


if __name__ == '__main__':
    main()
