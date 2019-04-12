#
# This script performs merges between three branches used in a release:
#
# - develop: Our development branch.  We merge all PRs into this branch.
# - release-$RELEASE: A local branch containing commits specific to this release.
#   This is a local-only branch, we never push this anywhere.
# - master: Our "clean" release branch.  Contains tags.
#
# The relationships between the three branches are illustrated below:
#
#   github.com PRs
#          \
# develop --+--+----------------------------------+---
#               \                                /
#   (new branch) \ commits (CHANGELOG.md, etc)  /
#                 \   v                        /
# release          ---*-----X (delete branch) / (merge 2)
#                         \                  /
#                (merge 1) \       TAG      /
#                           \       v      /
# master  -------------------+------*-----+-----------
#

echo "RELEASE: $RELEASE"
set -euxo pipefail


#
# Delete local branches, because we'll be cloning them entirely from the remote.
# You'll need to be on the release branch to be able to delete the other two.
#
git checkout release-${RELEASE}
set +e
git branch -D master develop
set -e

git fetch upstream
git checkout upstream/master -b master

#
# Merge the release branch into master.  This is merge 1 in the diagram above.
#
git merge --no-ff release-${RELEASE}
git tag -a ${RELEASE} -m "${RELEASE}"

#
# Merge the master branch into develop.  This is merge 2 in the diagram above.
#
git checkout upstream/develop -b develop
git merge --no-ff master
