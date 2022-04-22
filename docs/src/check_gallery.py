"""Check that the cached gallery files are up to date.

If they are stale, then Sphinx will attempt to rebuild them from source.  When
running the documentation build on CI, we want to avoid rebuilding the gallery,
because that takes too long.  Instead, we use this script to warn the author of
the PR that they need to rebuild the docs themselves.
"""

import hashlib
import os
import sys


def different(path1, path2):
    with open(path1) as fin:
        f1 = fin.read()
    with open(path2) as fin:
        f2 = fin.read()
    return f1 != f2


curr_dir = os.path.dirname(__file__)
stale = []
for root, dirs, files in os.walk(os.path.join(curr_dir, 'gallery')):
    for f in files:
        if f.endswith('.py'):
            source_path = os.path.join(root, f)
            cache_path = source_path.replace('docs/src/gallery/', 'docs/src/auto_examples/')

            #
            # We check two things:
            #
            #   1) Actual file content
            #   2) MD5 checksums
            #
            # We check 1) because that's the part that matters to the user -
            # it's what will appear in the documentation.  We check 2) because
            # that's what Sphinx Gallery relies on to decide what it needs to
            # rebuild.  In practice, only one of these checks is necessary,
            # but we run them both because it's trivial.
            #
            if different(source_path, cache_path):
                stale.append(cache_path)
                continue

            actual_md5 = hashlib.md5()
            with open(source_path, 'rb') as fin:
                actual_md5.update(fin.read())

            md5_path = cache_path + '.md5'
            with open(md5_path) as fin:
                expected_md5 = fin.read()

            if actual_md5.hexdigest() != expected_md5:
                stale.append(cache_path)

if stale:
    print(f"""The gallery cache appears stale.

Rebuild the documentation using the following commands from the gensim root subdirectory:

    pip install -e .[docs]
    make -C docs/src html

and then run `git add docs/src/auto_examples` to update the cache.

Stale files: {stale}
""", file=sys.stderr)
    sys.exit(1)
