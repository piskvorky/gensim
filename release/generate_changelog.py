#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Copyright (C) 2020 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Generate changelog entries for all PRs merged since the last release."""
import re
import requests
import sys
import time


def throttle_get(*args, seconds=10, **kwargs):
    print(args, kwargs, file=sys.stderr)
    result = requests.get(*args, **kwargs)
    result.raise_for_status()

    # Avoid Github API throttling; see https://github.com/RaRe-Technologies/gensim/pull/3203#issuecomment-887453109
    time.sleep(seconds)

    return result


#
# The releases get sorted in reverse chronological order, so the first release
# in the list is the most recent.
#
get = throttle_get('https://api.github.com/repos/RaRe-Technologies/gensim/releases')
most_recent_release = get.json()[0]
release_timestamp = most_recent_release['published_at']


def iter_merged_prs(since=release_timestamp):
    page = 1
    while True:
        get = throttle_get(
            'https://api.github.com/repos/RaRe-Technologies/gensim/pulls',
            params={'state': 'closed', 'page': page},
        )

        pulls = get.json()

        count = 0
        for i, pr in enumerate(pulls):
            if pr['merged_at'] and pr['merged_at'] > since:
                count += 1
                yield pr

        if count == 0:
            break

        page += 1


def iter_closed_issues(since=release_timestamp):
    page = 1
    while True:
        get = throttle_get(
            'https://api.github.com/repos/RaRe-Technologies/gensim/issues',
            params={'state': 'closed', 'page': page, 'since': since},
        )
        issues = get.json()
        if not issues:
            break

        count = 0
        for i, issue in enumerate(issues):
            #
            # In the github API, all pull requests are issues, but not vice versa.
            #
            if 'pull_request' not in issue and issue['closed_at'] > since:
                count += 1
                yield issue

        if count == 0:
            break

        page += 1


fixed_issue_numbers = set()
for pr in iter_merged_prs(since=release_timestamp):
    pr['user_login'] = pr['user']['login']
    pr['user_html_url'] = pr['user']['html_url']
    print('* [#%(number)d](%(html_url)s): %(title)s, by [@%(user_login)s](%(user_html_url)s)' % pr)

    #
    # Unfortunately, the GitHub API doesn't link PRs to issues that they fix,
    # so we have do it ourselves.
    #
    if pr['body'] is None:
        #
        # Weird edge case, PR with no body
        #
        continue

    for match in re.finditer(r'fix(es)? #(?P<number>\d+)\b', pr['body'], flags=re.IGNORECASE):
        fixed_issue_numbers.add(int(match.group('number')))


print()
print('### :question: Closed issues')
print()
print('TODO: move each issue to its appropriate section or delete if irrelevant')
print()

for issue in iter_closed_issues(since=release_timestamp):
    if 'pull_request' in issue or issue['number'] in fixed_issue_numbers:
        continue
    print('* [#%(number)d](%(html_url)s): %(title)s' % issue)
