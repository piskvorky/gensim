import csv
import requests

with open('redirects.csv') as fin:
    reader = csv.reader(fin)
    redirects = {r[0]: r[1] for r in reader}

for source, dest in redirects.items():
    source_status = requests.head(source).status_code
    dest_status = requests.head(dest).status_code
    print('source: %r (HTTP %d)' % (source, source_status))
    print('dest: %r (HTTP %d)' % (dest, dest_status))
    print()
