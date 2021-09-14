#
# Push local branches to upstream (github.com).
#
# Run this after you've verified the results of merge.sh.
#
set -euxo pipefail
release=$RELEASE

git push --tags upstream master
git push upstream develop
git push upstream release-"$release"
