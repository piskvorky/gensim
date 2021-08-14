"""Hijack a PR to add commits as a maintainer.

This is a two-step process:

    1. Add a git remote that points to the contributor's repo
    2. Check out the actual contribution by reference

As a maintainer, you can add changes by making new commits and pushing them
back to the remote.
"""
import json
import subprocess
import sys

import smart_open

prid = int(sys.argv[1])
url = f"https://api.github.com/repos/RaRe-Technologies/gensim/pulls/{prid}"
with smart_open.open(url) as fin:
    prinfo = json.load(fin)

user = prinfo['head']['user']['login']
ssh_url = prinfo['head']['repo']['ssh_url']

remotes = subprocess.check_output(['git', 'remote']).strip().decode('utf-8').split('\n')
if user not in remotes:
    subprocess.check_call(['git', 'remote', 'add', user, ssh_url])

subprocess.check_call(['git', 'fetch', user])

ref = prinfo['head']['ref']
subprocess.check_call(['git', 'checkout', f'{user}/{ref}'])
subprocess.check_call(['git', 'switch', '-c', f'{ref}'])
