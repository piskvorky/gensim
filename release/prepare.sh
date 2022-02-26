#
# This script initiates a new release.
#
# Run it like so:
#
#   bash start.sh 1.2.3
#
# where 1.2.3 is the release version.
#

set -euxo pipefail

previous_version=$(python -c 'import gensim;print(gensim.__version__)')

RELEASE=$1
export RELEASE="$RELEASE"

script_dir="$(dirname "${BASH_SOURCE[0]}")"
root=$(cd "$script_dir/.." && pwd)
cd "$script_dir"

git fetch upstream
git checkout upstream/develop

#
# Get rid of the local release branch, if it exists.
#
set +e
git branch -D release-"$RELEASE"
set -e

git checkout -b release-"$RELEASE"
python bump_version.py "$previous_version" "$RELEASE"

#
# N.B. grep exits with nonzero if the target string is not found
#
grep "$RELEASE" "$root/setup.py"
grep "$RELEASE" "$root/docs/src/conf.py"
grep "$RELEASE" "$root/gensim/__init__.py"

set +e
git diff | cat
git commit -a -m "bumped version to $RELEASE"
set -e

echo "Now update CHANGELOG.md and include the PRs in this release."
read -p "Press Enter to continue.  An editor window will open."
python update_changelog.py "$previous_version" "$RELEASE"
$EDITOR "$root/CHANGELOG.md"

set +e
git commit "$root/CHANGELOG.md" -m "updated CHANGELOG.md for version $RELEASE"
set -e

echo "Have a look at the current branch, and if all looks good, run merge.sh"
