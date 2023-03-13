#!/usr/bin/env python

"""
A simple testrunner for nose (or anything else).

Watch for changes in all file types specified in 'EXTENSIONS'.
If changes, run test executable in 'EXECUTABLE', with default
arguments 'DEFAULTARGS'.

The --with-color option needs the "rudolf" nose plugin. See:
https://pypi.org/project/rudolf/

Originally by Jeff Winkler, http://jeffwinkler.net
Forked from wkral https://github.com/wkral/Nosy
"""

import os
import stat
import time
import datetime
import sys
import fnmatch


EXTENSIONS = ['*.py']
EXECUTABLE = 'nosetests test/'
DEFAULTARGS = '--with-color -exe'  # -w tests'


def check_sum():
    """
    Return a long which can be used to know if any .py files have changed.
    """
    val = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for extension in EXTENSIONS:
            for f in fnmatch.filter(files, extension):
                stats = os.stat(os.path.join(root, f))
                val += stats[stat.ST_SIZE] + stats[stat.ST_MTIME]
    return val


if __name__ == '__main__':
    val = 0
    try:
        while True:
            if check_sum() != val:
                val = check_sum()
                os.system('%s %s %s' % (EXECUTABLE, DEFAULTARGS, ' '.join(sys.argv[1:])))
                print(datetime.datetime.now().__str__())
                print('=' * 77)
            time.sleep(1)
    except KeyboardInterrupt:
        print('Goodbye')
