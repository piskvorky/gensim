"""Upgrade pip to the most recent version before running pip install."""
import os
import sys


def run(command):
    print(command)
    os.system(command)


def main():
    opts_and_packages = ' '.join(sys.argv[1:])
    run('python -m pip install --upgrade pip')
    run('python -m pip install %s' % opts_and_packages)


if __name__ == '__main__':
    main()
