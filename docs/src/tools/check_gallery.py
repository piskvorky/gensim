"""Check that the gallery output is up to date with the input.

We do this so we can know in advance if "make html" is going to rebuild the
gallery.  That's helpful to know because rebuilding usually takes a long time,
so we want to avoid it under some environments (e.g. CI).

The script returns non-zero if there are any problems. At that stage, you may
fail the CI build immediately, as further building will likely take too long.
If you run the script interactively, it will give you tips about what you may
want to do, on standard output.
"""

import os
import os.path
import re
import sys


def get_friends(py_file):
    for ext in ('.py', '.py.md5', '.rst', '.ipynb'):
        friend = re.sub(r'\.py$', ext, py_file)
        if os.path.isfile(friend):
            yield friend


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(curr_dir, '../auto_examples/'))
    src_dir = os.path.abspath(os.path.join(curr_dir, '..'))

    retval = 0

    for root, dirs, files in os.walk(output_dir):
        py_files = [os.path.join(root, f) for f in files if f.endswith('.py')]
        for out_file in py_files:
            in_file = out_file.replace('/auto_examples/', '/gallery/')
            if not os.path.isfile(in_file):
                print('%s is stale, consider removing it and its friends:' % in_file)
                for friend in get_friends(out_file):
                    print('\tgit rm -f %s' % friend)
                retval = 1
                continue

            with open(in_file) as fin:
                in_py = fin.read()
            with open(out_file) as fin:
                out_py = fin.read()

            if in_py != out_py:
                print('%s is stale, consider rebuilding the gallery:' % in_file)
                print('\tmake -C %s html' % src_dir)
                retval = 1

    return retval


if __name__ == '__main__':
    sys.exit(main())
