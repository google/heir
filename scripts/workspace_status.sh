#!/bin/bash
# Used to map BuildBuddy invocations to a repo/branch/commit.
# See https://www.buildbuddy.io/docs/guide-metadata/
echo "COMMIT_SHA $(git rev-parse HEAD)"

# Branch is a bit tricky, since on GitHub Actions the checkout is a detached HEAD,
# so `git rev-parse --abbrev-ref HEAD` is just "HEAD", but we can check for ENV variables
# such as GITHUB_HEAD_REF (PR source branch) or GITHUB_REF_NAME (branch on push).
# NB: the workspace-status key is GIT_BRANCH, not BRANCH_NAME. BRANCH_NAME is the
# --build_metadata flag name; workspace-status ingestion uses GIT_BRANCH (see
# buildbuddy-io/buildbuddy:workspace_status.sh). Don't "fix" this to BRANCH_NAME.
echo "GIT_BRANCH ${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}}"

# Repo is even trickier, since a checkout might have multiple (or no) remotes.
# We once again read it from ENV for GitHub actions,
if [ -n "$GITHUB_REPOSITORY" ]; then
  repo_url="${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY"
else
  # otherwise try:
  # 1.) current branch's configured remote
  branch=$(git rev-parse --abbrev-ref HEAD)
  remote=$(git config --get "branch.$branch.remote")
  # 2.) remote.pushDefault setting if it is set
  [ -z "$remote" ] && remote=$(git config --get remote.pushDefault)
  # 3.) the first remote, if it exists
  [ -z "$remote" ] && remote=$(git remote | head -n1)
  # Normalize ssh to https: git@host:owner/repo(.git) -> https://host/owner/repo
  repo_url=$(git remote get-url "$remote" 2>/dev/null | sed -E -e 's#^git@([^:]+):#https://\1/#' -e 's#\.git$##')
fi
if [ -n "$repo_url" ]; then echo "REPO_URL $repo_url"; fi
