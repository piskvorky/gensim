"""
Bump the version of Gensim in all the required places.

Usage: python3 bump_version.py <OLD_VERSION> <NEW_VERSION>

Example:
    python3 bump_version.py "4.0.0beta" "4.0.0rc1"

"""

import os.path
import re
import sys


def bump(path, pattern, repl, check=True):
    with open(path) as fin:
        contents = fin.read()

    new_contents = pattern.sub(repl, contents)

    if check and new_contents == contents:
        print('*' * 79)
        print('WARNING: contents of %r unchanged after version bump' % path)
        print('*' * 79)

    with open(path, 'w') as fout:
        fout.write(new_contents)

def bump_setup_py(root, previous_version, new_version):
    path = os.path.join(root, 'setup.py')
    pattern = re.compile("^    version='%s',$" % previous_version, re.MULTILINE)
    repl = "    version='%s'," % new_version
    bump(path, pattern, repl)


def bump_docs_src_conf_py(root, previous_version, new_version):
    path = os.path.join(root, 'docs', 'src', 'conf.py')

    short_previous_version = '.'.join(previous_version.split('.')[:2])
    short_new_version = new_version  # '.'.join(new_version.split('.')[:2])
    pattern = re.compile("^version = '%s'$" % short_previous_version, re.MULTILINE)
    repl = "version = '%s'" % short_new_version
    bump(path, pattern, repl, check=False)  # short version won't always change

    pattern = re.compile("^release = '%s'$" % previous_version, re.MULTILINE)
    repl = "release = '%s'" % new_version
    bump(path, pattern, repl)


def bump_gensim_init_py(root, previous_version, new_version):
    path = os.path.join(root, 'gensim', '__init__.py')
    pattern = re.compile("__version__ = '%s'$" % previous_version, re.MULTILINE)
    repl = "__version__ = '%s'" % new_version
    bump(path, pattern, repl)


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    previous_version, new_version = sys.argv[1:3]

    bump_setup_py(root, previous_version, new_version)
    bump_docs_src_conf_py(root, previous_version, new_version)
    bump_gensim_init_py(root, previous_version, new_version)


if __name__ == '__main__':
    main()
