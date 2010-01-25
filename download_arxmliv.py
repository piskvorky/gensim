#!/usr/bin/env python2.5

import urllib
import os
import os.path
import re

pat = re.compile('<td><a href="([0-9]*)/">')

for num, line in enumerate(open('index.html')):
#    if num <= 36271:
#        continue
    parts = pat.findall(line)
    if not parts or len(parts) > 1:
        continue
    part = parts[0]
    dest = os.path.join(part[-2 : ], '#%s' % part)
    if os.path.exists(dest):
        continue
    url = 'http://arxmliv.kwarc.info/files/math/papers/%s/%s.tex.xml' % (part, part)
    try:
        conn = urllib.urlopen(url)
        if not 'application/xml' in conn.info().values():
            raise IOError
    except IOError:
        print "failed to download %s" % url
        continue
    content = conn.read()
    conn.close()
    os.makedirs(dest)
    fout = open(os.path.join(dest, 'fulltext.xml'), 'w')
    fout.write(content)
    fout.close()
    print "processed url #%i: %s" % (num, url)
