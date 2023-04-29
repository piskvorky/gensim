#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
"""Print available wheels for a particular Python package."""
import re
import sys

import requests

def to_int(value):
    value = ''.join((x for x in value if x.isdigit()))
    try:
        return int(value)
    except Exception:
        return 0


def to_tuple(version):
    return tuple(to_int(x) for x in version.split('.'))


def main():
    project = sys.argv[1]
    json = requests.get('https://pypi.org/pypi/%s/json' % project).json()
    for version in sorted(json['releases'], key=to_tuple):
        print(version)
        wheel_packages = [
            p for p in json['releases'][version]
            if p['packagetype'] == 'bdist_wheel'
        ]
        for p in wheel_packages:
            print('    %(python_version)s %(filename)s' % p)


if __name__ == '__main__':
    main()
