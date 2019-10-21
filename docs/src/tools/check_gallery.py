"""Check that the gallery output is up to date with the input.

We do this so we can know in advance if "make html" is going to rebuild the
gallery.  That's helpful to know because rebuilding usually takes a long time,
so we want to avoid it under some environments (e.g. CI).

The script returns non-zero if there are any problems. At that stage, you may
fail the CI build immediately, as further building will likely take too long.
If you run the script interactively, it will give you tips about what you may
want to do, on standard output.

If you run this script with the --apply option set, it will automatically run
the suggested commands for you.
"""

import argparse
import os
import os.path
import re
import sys
import shlex
import subprocess


def get_friends(py_file):
    for ext in ('.py', '.py.md5', '.rst', '.ipynb'):
        friend = re.sub(r'\.py$', ext, py_file)
        if os.path.isfile(friend):
            yield friend


def is_under_version_control(path):
    command = ['git', 'ls-files', '--error-unmatch', path]
    popen = subprocess.Popen(
        command,
        cwd=os.path.dirname(path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    popen.communicate()
    popen.wait()
    return popen.returncode == 0


def find_unbuilt_examples(gallery_subdir):
    """Returns True if there are any examples that have not been built yet."""
    for root, dirs, files in os.walk(gallery_subdir):
        in_files = [os.path.join(root, f) for f in files if f.endswith('.py')]
        for in_file in in_files:
            out_file = in_file.replace('/gallery/', '/auto_examples/') 
            friends = list(get_friends(out_file))
            if any([not os.path.isfile(f) for f in friends]):
                yield in_file


def diff(f1, f2):
    """Returns True if the files are different."""
    with open(f1) as fin:
        f1_contents = fin.read()
    with open(f2) as fin:
        f2_contents = fin.read()
    return f1_contents != f2_contents


def find_py_files(subdir):
    for root, dirs, files in os.walk(subdir):
        for f in files:
            if f.endswith('.py'):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--apply', action='store_true',
        help='Apply any suggestions made by this script',
    )
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(curr_dir, '../auto_examples/'))

    retval = 0
    rebuild = False
    suggestions = []

    #
    # Check for stale output.
    #
    for out_file in find_py_files(output_dir):
        in_file = out_file.replace('/auto_examples/', '/gallery/')
        if not os.path.isfile(in_file):
            print('%s is stale, consider removing it and its friends.' % in_file)
            for friend in get_friends(out_file):
                suggestions.append('git rm -f %s' % friend)
            retval = 1
            continue

        for friend in get_friends(out_file):
            if not is_under_version_control(friend):
                print('%s is not under version control, consider adding it.' % friend)
                suggestions.append('git add %s' % friend)

        if diff(in_file, out_file):
            print('%s is stale.' % in_file)
            rebuild = True
            retval = 1

    gallery_dir = output_dir.replace('/auto_examples', '/gallery')
    unbuilt = list(find_unbuilt_examples(gallery_dir))
    if unbuilt:
        for u in unbuilt:
            print('%s has not been built yet' % u)
        rebuild = True
        retval = 1

    if rebuild:
        src_dir = os.path.abspath(os.path.join(gallery_dir, '..'))
        print('consider rebuilding the gallery:')
        print('\tmake -C %s html' % src_dir)

    if suggestions:
        print('consider running the following commands (or rerun this script with --apply option):')
        for command in suggestions:
            print('\t' + command)

    if args.apply:
        for command in suggestions:
            subprocess.check_call(shlex.split(command))

    return retval


if __name__ == '__main__':
    sys.exit(main())
