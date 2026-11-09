"""Post a welcome comment on a first-time contributor's pull request.

Invoked by .github/workflows/first_interaction.yml, which runs on `workflow_run`
from the default branch with a write-scoped GITHUB_TOKEN. Inputs come from the
GitHub-provided `workflow_run` payload; nothing produced by the (potentially
untrusted) PR workflow run is consumed.

This script depends only on the Python standard library and the `gh` CLI, both
of which are preinstalled on GitHub-hosted runners, so the workflow does not
need to install anything. `gh` authenticates via the GH_TOKEN environment
variable (as in CI) or an existing `gh auth login` session (for local use).

Example (dry run, prints the decision without commenting):

  python3 scripts/first_interaction.py \
      --head-owner=someuser --head-branch=my-branch --head-sha=abc123 \
      --dry-run

Or, to see what the decision would be for a specific (open or closed) PR:

  python3 scripts/first_interaction.py --pr=1234 --dry-run
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from typing import Any, Protocol

# Hidden marker used to detect a previously posted welcome comment.
MARKER = "<!-- heir-first-interaction-welcome -->"

WELCOME_MESSAGE = f"""{MARKER}
Welcome to HEIR!

Thank you for opening your first pull request! We appreciate you joining our community.

Please note that our [contributing policy](https://github.com/google/heir/blob/main/CONTRIBUTING.md) requires unrecognized users to meet a maintainer in a non-textual format before their first PR will be reviewed.

If you have already met a maintainer, or you're sure the maintainers know who you are, you can safely ignore this message. Otherwise, a maintainer will likely close this PR shortly.

Don't take it personally! Instead, we welcome you to attend one of our open meetings on our [community calendar](https://heir.dev/community), schedule a dedicated call with a maintainer, or reach out in the `#heir` channel on [Discord](https://discord.fhe.org/).

We look forward to connecting with you!
"""

# Author associations indicating the author is already known to the project.
# CONTRIBUTOR means GitHub already attributes a commit in the repo to them. See
# https://docs.github.com/en/graphql/reference/enums#commentauthorassociation
KNOWN_ASSOCIATIONS = frozenset(
    {"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}
)


class GitHubClient(Protocol):
  """The subset of the GitHub REST API used by this script."""

  def list_open_pulls_from_head(self, head: str) -> list[dict[str, Any]]:
    ...

  def get_pull(self, number: int) -> dict[str, Any]:
    ...

  def has_commits_by_author(self, branch: str, author: str) -> bool:
    ...

  def list_issues_by_creator(self, creator: str) -> list[dict[str, Any]]:
    ...

  def list_issue_comments(self, issue_number: int) -> list[dict[str, Any]]:
    ...

  def create_issue_comment(self, issue_number: int, body: str) -> None:
    ...


class GhCliClient:
  """A GitHubClient that shells out to `gh api`."""

  def __init__(self, repo: str, gh: str = "gh"):
    self._repo = repo
    self._gh = gh

  def _api(
      self,
      path: str,
      fields: dict[str, str | int] | None = None,
      method: str = "GET",
      paginate: bool = False,
  ) -> Any:
    """Calls `gh api` on a repo-relative path and returns the decoded JSON.

    Args:
      path: The API path relative to `repos/{repo}/`.
      fields: Query parameters for GET requests, or body fields otherwise.
      method: The HTTP method. Always passed explicitly, since `gh api`
        otherwise switches to POST whenever fields are given.
      paginate: Whether to fetch all pages of a list endpoint and return their
        concatenated items.

    Returns:
      The decoded JSON response.

    Raises:
      subprocess.CalledProcessError: if `gh` fails. Its stderr is not captured,
        so the error appears in the job log.
    """
    cmd = [self._gh, "api", "--method", method, f"repos/{self._repo}/{path}"]
    for key, value in (fields or {}).items():
      # `-f` passes values verbatim (unlike `-F`, which interprets `@file`,
      # numbers, and booleans).
      cmd += ["-f", f"{key}={value}"]
    if paginate:
      # Emit each item of each page as a line of compact JSON. This avoids
      # `--slurp`, which requires gh >= 2.48.
      cmd += ["--paginate", "--jq", ".[]"]
    stdout = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    if paginate:
      return [json.loads(line) for line in stdout.splitlines() if line]
    return json.loads(stdout)

  def list_open_pulls_from_head(self, head: str) -> list[dict[str, Any]]:
    return self._api(
        "pulls",
        {"state": "open", "head": head, "per_page": 100},
        paginate=True,
    )

  def get_pull(self, number: int) -> dict[str, Any]:
    return self._api(f"pulls/{number}")

  def has_commits_by_author(self, branch: str, author: str) -> bool:
    commits = self._api(
        "commits", {"sha": branch, "author": author, "per_page": 1}
    )
    return bool(commits)

  def list_issues_by_creator(self, creator: str) -> list[dict[str, Any]]:
    return self._api(
        "issues",
        {"state": "all", "creator": creator, "per_page": 100},
        paginate=True,
    )

  def list_issue_comments(self, issue_number: int) -> list[dict[str, Any]]:
    return self._api(
        f"issues/{issue_number}/comments", {"per_page": 100}, paginate=True
    )

  def create_issue_comment(self, issue_number: int, body: str) -> None:
    self._api(f"issues/{issue_number}/comments", {"body": body}, method="POST")


def select_triggering_pr(
    pulls: list[dict[str, Any]], head_sha: str
) -> dict[str, Any] | None:
  """Returns the PR whose head commit is `head_sha`, if any."""
  return next((pr for pr in pulls if pr["head"]["sha"] == head_sha), None)


def known_author_reason(pr: dict[str, Any]) -> str | None:
  """Returns why the PR author is already known, or None if they are not."""
  if pr["user"]["type"] == "Bot":
    return "author is a bot"
  association = pr.get("author_association", "")
  if association in KNOWN_ASSOCIATIONS:
    return f"author is already associated with the repo ({association})"
  return None


def has_earlier_pulls(issues: list[dict[str, Any]], pr_number: int) -> bool:
  """Whether `issues` contains a PR opened before `pr_number`.

  The issues API returns both issues and PRs; PRs have a `pull_request` field.
  Issue and PR numbers share one increasing sequence, so a lower number means
  an earlier PR. Ignoring later PRs means the author's first PR is still
  welcomed if they quickly open a second one, and keeps `--pr` dry runs on
  older PRs accurate.

  Args:
    issues: Issues returned by the GitHub issues API.
    pr_number: The PR being considered.

  Returns:
    True if any element of `issues` is a PR numbered lower than `pr_number`.
  """
  return any(
      issue.get("pull_request") is not None and issue["number"] < pr_number
      for issue in issues
  )


def has_welcome_comment(comments: list[dict[str, Any]]) -> bool:
  return any(MARKER in (comment.get("body") or "") for comment in comments)


@dataclasses.dataclass(frozen=True)
class Decision:
  """Whether to welcome the author of `pr_number`, and why."""

  should_welcome: bool
  reason: str
  pr_number: int | None = None


def decide(
    client: GitHubClient,
    default_branch: str,
    head_owner: str,
    head_branch: str,
    head_sha: str,
) -> Decision:
  """Decides whether the PR at the given head should get a welcome comment."""
  if not head_owner:
    return Decision(False, "head repository no longer exists")

  pulls = client.list_open_pulls_from_head(f"{head_owner}:{head_branch}")
  pr = select_triggering_pr(pulls, head_sha)
  if pr is None:
    return Decision(
        False, f"no open PR found for {head_owner}:{head_branch}@{head_sha}"
    )

  return decide_for_pr(client, default_branch, pr)


def decide_for_pr(
    client: GitHubClient, default_branch: str, pr: dict[str, Any]
) -> Decision:
  """Decides whether the author of `pr` should get a welcome comment."""
  pr_number = pr["number"]
  author = pr["user"]["login"]

  if reason := known_author_reason(pr):
    return Decision(False, reason, pr_number)

  if client.has_commits_by_author(default_branch, author):
    return Decision(
        False, f"{author} already has commits on {default_branch}", pr_number
    )

  if has_earlier_pulls(client.list_issues_by_creator(author), pr_number):
    return Decision(False, f"{author} has opened earlier PRs", pr_number)

  if has_welcome_comment(client.list_issue_comments(pr_number)):
    return Decision(False, "welcome comment already posted", pr_number)

  return Decision(True, f"{author} is a first-time contributor", pr_number)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--repo", default="google/heir")
  parser.add_argument("--default-branch", default="main")
  parser.add_argument(
      "--head-owner",
      default="",
      help="Owner of the PR head repo; empty if it was deleted.",
  )
  parser.add_argument("--head-branch")
  parser.add_argument("--head-sha")
  parser.add_argument(
      "--pr",
      type=int,
      help=(
          "Evaluate this PR number instead of looking one up by head. Requires"
          " --dry-run. Note that checks reflect the repo's current state, e.g.,"
          " the author's commits on the default branch since the PR was opened."
      ),
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the decision without posting a comment.",
  )
  args = parser.parse_args(argv)

  if args.pr is not None:
    if not args.dry_run:
      parser.error("--pr requires --dry-run")
  elif args.head_branch is None or args.head_sha is None:
    parser.error(
        "either --pr or both --head-branch and --head-sha are required"
    )

  client = GhCliClient(args.repo)
  if args.pr is not None:
    decision = decide_for_pr(
        client, args.default_branch, client.get_pull(args.pr)
    )
  else:
    decision = decide(
        client,
        default_branch=args.default_branch,
        head_owner=args.head_owner,
        head_branch=args.head_branch,
        head_sha=args.head_sha,
    )
  pr_desc = f"PR #{decision.pr_number}" if decision.pr_number else "No PR"
  print(f"{pr_desc}: {decision.reason}.")

  if decision.should_welcome and decision.pr_number is not None:
    if args.dry_run:
      print("Dry run; would have posted the welcome comment.")
    else:
      client.create_issue_comment(decision.pr_number, WELCOME_MESSAGE)
      print("Posted welcome comment.")
  elif args.dry_run:
    print("Dry run; would not have posted the welcome comment.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
