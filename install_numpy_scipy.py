"""Install the best versions of Numpy and Scipy for the current environment.

"Best" means wheels are available, so installation does not require building.

We use this for setting up CI environments.

"""

import pip
import sys


def main():
    if sys.version[:2] == (3, 7):
        pip.main(['install', 'numpy==1.14.5', 'scipy==1.1.0'])
    else:
        pip.main(['install', 'numpy==1.11.3', 'scipy==1.0.0'])


if __name__ == '__main__':
    main()
