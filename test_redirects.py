import csv
import os

import requests


with open('redirects.csv') as fin:
    reader = csv.reader(fin)
    redirects = {r[0]: r[1] for r in reader}

for source, dest in redirects.items():
    if os.environ.get('NUMFOCUS'):
        dest = dest.replace('/gensim/', '/gensim/gensim_numfocus/')
    source_status = requests.head(source).status_code
    dest_status = requests.head(dest).status_code
    if (source_status not in (200, 301)) or (dest_status not in (200, 301)):
        print('source: %r (HTTP %d)' % (source, source_status))
        print('dest: %r (HTTP %d)' % (dest, dest_status))
        break

print('OK')
