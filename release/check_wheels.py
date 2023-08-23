# -*- coding: utf-8 -*-
#
# Authors: Michael Penkov <m@penkov.dev>
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
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
    'gensim-%(release)s-cp310-cp310-macosx_10_9_x86_64.whl',
    'gensim-%(release)s-cp310-cp310-macosx_11_0_arm64.whl',
    'gensim-%(release)s-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
    'gensim-%(release)s-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
    'gensim-%(release)s-cp310-cp310-win_amd64.whl',
    'gensim-%(release)s-cp310-cp310-win_arm64.whl',
    'gensim-%(release)s-cp311-cp311-macosx_10_9_x86_64.whl',
    'gensim-%(release)s-cp311-cp311-macosx_11_0_arm64.whl',
    'gensim-%(release)s-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
    'gensim-%(release)s-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
    'gensim-%(release)s-cp311-cp311-win_amd64.whl',
    'gensim-%(release)s-cp311-cp311-win_arm64.whl',
    'gensim-%(release)s-cp38-cp38-macosx_10_9_x86_64.whl',
    'gensim-%(release)s-cp38-cp38-macosx_11_0_arm64.whl',
    'gensim-%(release)s-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
    'gensim-%(release)s-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
    'gensim-%(release)s-cp38-cp38-win_amd64.whl',
    'gensim-%(release)s-cp39-cp39-macosx_10_9_x86_64.whl',
    'gensim-%(release)s-cp39-cp39-macosx_11_0_arm64.whl',
    'gensim-%(release)s-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
    'gensim-%(release)s-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
    'gensim-%(release)s-cp39-cp39-win_amd64.whl',
    'gensim-%(release)s-cp39-cp39-win_arm64.whl',
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
