"""Updates the changelog with PRs merged since the last version."""
import datetime
import json
import os.path
import sys

import requests


URL = 'https://api.github.com/repos/RaRe-Technologies/gensim'


def summarize_prs(since_version):
    """Go through all closed PRs, summarize those merged after the previous release.

    Yields one-line summaries of each relevant PR as a string.
    """
    releases = requests.get(URL + '/releases').json()
    most_recent_release = releases[0]['tag_name']
    assert most_recent_release  == since_version, 'unexpected most_recent_release: %r' % most_recent_release

    published_at = releases[0]['published_at']

    pulls = requests.get(URL + '/pulls', params={'state': 'closed'}).json()
    for pr in pulls:
        merged_at = pr['merged_at']
        if merged_at is None or merged_at < published_at:
            continue

        summary = "* {msg} (__[{author}]({author_url})__, [#{pr}]({purl}))".format(
            msg=pr['title'],
            author=pr['user']['login'],
            author_url=pr['user']['html_url'],
            pr=pr['number'],
            purl=pr['html_url'],
        )
        print(summary)
        yield summary


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    previous_version, new_version = sys.argv[1:3]

    path = os.path.join(root, 'CHANGELOG.md')
    with open(path) as fin:
        contents = fin.read().split('\n')

    header, contents = contents[:2], contents[2:]
    header.append('## %s, %s\n' % (new_version, datetime.date.today().isoformat()))
    header.append("""
### :star2: New Features

### :red_circle: Bug fixes

### :books: Tutorial and doc improvements

### :+1: Improvements

### :warning: Deprecations (will be removed in the next major release)

**COPY-PASTE DEPRECATIONS FROM THE PREVIOUS RELEASE HERE**

Please organize the PR summaries from below into the above sections
You may remove empty sections.  Be sure to include all deprecations.
""")

    header += list(summarize_prs(previous_version))

    with open(path, 'w') as fout:
        fout.write('\n'.join(header + contents))


if __name__ == '__main__':
    main()
