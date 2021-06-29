"""Helper script for including change log entries in an open PR.

Automatically constructs the change log entry from the PR title.
Copies the entry to the window manager clipboard.
Opens the change log belonging to the specific PR in a browser window.
All you have to do is paste and click "commit changes".
"""
import json
import sys
import webbrowser

import smart_open


def copy_to_clipboard(text):
    try:
        import pyperclip
    except ImportError:
        print('pyperclip <https://pypi.org/project/pyperclip/> is missing.', file=sys.stderr)
        print('copy-paste the following text manually:', file=sys.stderr)
        print('\t', text, file=sys.stderr)
    else:
        pyperclip.copy(text)


prid = int(sys.argv[1])
url = "https://api.github.com/repos/RaRe-Technologies/gensim/pulls/%d" % prid
with smart_open.open(url) as fin:
    prinfo = json.load(fin)

prinfo['user_login'] = prinfo['user']['login']
prinfo['user_html_url'] = prinfo['user']['html_url']
text = '[#%(number)s](%(html_url)s): %(title)s, by [@%(user_login)s](%(user_html_url)s)' % prinfo
copy_to_clipboard(text)

prinfo['head_repo_html_url'] = prinfo['head']['repo']['html_url']
prinfo['head_ref'] = prinfo['head']['ref']
edit_url = '%(head_repo_html_url)s/edit/%(head_ref)s/CHANGELOG.md' % prinfo
webbrowser.open(edit_url)
