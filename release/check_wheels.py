# -*- coding: utf-8 -*-
#
# Authors: Michael Penkov <m@penkov.dev>
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""Check that our wheels are all there."""
import os
import os.path
import re
import sys

#
# We expect this to be set as part of the release process.
#
release = os.environ['RELEASE']
assert re.match(r'^\d+.\d+.\d+', release), 'expected %r to be in major.minor.bugfix format'

dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dist')
dist_path = os.path.abspath(dist_path)
assert os.path.isdir(dist_path), 'expected %r to be an existing subdirectory' % dist_path

expected = [
    'gensim-%(release)s-cp27-cp27m-win32.whl',
    'gensim-%(release)s-cp27-cp27m-win_amd64.whl',
    'gensim-%(release)s-cp35-cp35m-win32.whl',
    'gensim-%(release)s-cp35-cp35m-win_amd64.whl',
    'gensim-%(release)s-cp36-cp36m-win32.whl',
    'gensim-%(release)s-cp36-cp36m-win_amd64.whl',
    'gensim-%(release)s-cp37-cp37m-win32.whl',
    'gensim-%(release)s-cp37-cp37m-win_amd64.whl',
    'gensim-%(release)s.win32-py2.7.exe',
    'gensim-%(release)s.win32-py3.5.exe',
    'gensim-%(release)s.win32-py3.6.exe',
    'gensim-%(release)s.win32-py3.7.exe',
    'gensim-%(release)s.win-amd64-py2.7.exe',
    'gensim-%(release)s.win-amd64-py3.5.exe',
    'gensim-%(release)s.win-amd64-py3.6.exe',
    'gensim-%(release)s.win-amd64-py3.7.exe',
    'gensim-%(release)s-cp27-cp27m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl',
    'gensim-%(release)s-cp27-cp27m-manylinux1_i686.whl',
    'gensim-%(release)s-cp27-cp27m-manylinux1_x86_64.whl',
    'gensim-%(release)s-cp27-cp27mu-manylinux1_i686.whl',
    'gensim-%(release)s-cp27-cp27mu-manylinux1_x86_64.whl',
    'gensim-%(release)s-cp35-cp35m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl',
    'gensim-%(release)s-cp35-cp35m-manylinux1_i686.whl',
    'gensim-%(release)s-cp35-cp35m-manylinux1_x86_64.whl',
    'gensim-%(release)s-cp36-cp36m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl',
    'gensim-%(release)s-cp36-cp36m-manylinux1_i686.whl',
    'gensim-%(release)s-cp36-cp36m-manylinux1_x86_64.whl',
    'gensim-%(release)s-cp37-cp37m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl',
    'gensim-%(release)s-cp37-cp37m-manylinux1_i686.whl',
    'gensim-%(release)s-cp37-cp37m-manylinux1_x86_64.whl',
    'gensim-%(release)s.tar.gz',
]

fail = False
for f in expected:
    wheel_path = os.path.join(dist_path, f % dict(release=release))
    if not os.path.isfile(wheel_path):
        print('FAIL: %s' % wheel_path)
        fail = True

if not fail:
    print('OK')

sys.exit(1 if fail else 0)
