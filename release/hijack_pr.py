#!/usr/bin/env python
"""Hijack a PR to add commits as a maintainer.

This is a two-step process:

    1. Add a git remote that points to the contributor's repo
    2. Check out the actual contribution by reference

As a maintainer, you can add changes by making new commits and pushing them
back to the remote.

An example session:

    $ release/hijack_pr.py 1234
    $ git merge upstream/develop  # or any other changes you want to make
    $ release/hijack_pr.py push

The above commands would check out the code for the PR, make changes to them, and push them back.
Obviously, this requires the PR to be writable, but most gensim PRs are.
If they aren't, then leave it up to the PR author to make the required changes.
"""
import json
import subprocess
import sys

import smart_open

def check_output(command):
    return subprocess.check_output(command).strip().decode('utf-8')


if sys.argv[1] == "push":
    command = "git rev-parse --abbrev-ref HEAD@{upstream}".split()
    remote, remote_branch = check_output(command).split('/')
    current_branch = check_output(['git', 'branch', '--show-current'])
    check_output(['git', 'push', remote, f'{current_branch}:{remote_branch}'])

    #
    # Cleanup to prevent remotes and branches from piling up
    #
    check_output(['git', 'branch', '--delete', current_branch])
    check_output(['git', 'remote', 'remove', remote])
    sys.exit(0)

prid = int(sys.argv[1])
url = f"https://api.github.com/repos/RaRe-Technologies/gensim/pulls/{prid}"
with smart_open.open(url) as fin:
    prinfo = json.load(fin)

user = prinfo['head']['user']['login']
ssh_url = prinfo['head']['repo']['ssh_url']

remotes = check_output(['git', 'remote']).split('\n')
if user not in remotes:
    subprocess.check_call(['git', 'remote', 'add', user, ssh_url])

subprocess.check_call(['git', 'fetch', user])

ref = prinfo['head']['ref']
subprocess.check_call(['git', 'checkout', f'{user}/{ref}'])

#
# Prefix the local branch name with the user to avoid naming clashes with
# existing branches, e.g. develop
#
subprocess.check_call(['git', 'switch', '-c', f'{user}_{ref}'])

#
# Set the upstream so we can push back to it more easily
#
subprocess.check_call(['git', 'branch', '--set-upstream-to', f'{user}/{ref}'])
