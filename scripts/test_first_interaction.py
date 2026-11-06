"""Tests for first_interaction.py."""

import contextlib
import io
import json
import subprocess
from typing import Any

from absl.testing import absltest
from scripts import first_interaction

patch = absltest.mock.patch

Decision = first_interaction.Decision
GhCliClient = first_interaction.GhCliClient
MARKER = first_interaction.MARKER
decide = first_interaction.decide
decide_for_pr = first_interaction.decide_for_pr
has_earlier_pulls = first_interaction.has_earlier_pulls
has_welcome_comment = first_interaction.has_welcome_comment
known_author_reason = first_interaction.known_author_reason
main = first_interaction.main
select_triggering_pr = first_interaction.select_triggering_pr

HEAD_OWNER = "newuser"
HEAD_BRANCH = "my-feature"
HEAD_SHA = "abc123"
PR_NUMBER = 42


def make_pr(
    number: int = PR_NUMBER,
    sha: str = HEAD_SHA,
    login: str = HEAD_OWNER,
    user_type: str = "User",
    association: str = "FIRST_TIME_CONTRIBUTOR",
) -> dict[str, Any]:
  return {
      "number": number,
      "head": {"sha": sha},
      "user": {"login": login, "type": user_type},
      "author_association": association,
  }


def make_issue(number: int, is_pr: bool) -> dict[str, Any]:
  issue: dict[str, Any] = {"number": number}
  if is_pr:
    issue["pull_request"] = {"url": f"https://example.com/pulls/{number}"}
  return issue


class FakeGitHubClient:
  """An in-memory fake of first_interaction.GitHubClient."""

  def __init__(
      self,
      pulls: list[dict[str, Any]] | None = None,
      commit_authors: set[str] | None = None,
      issues: list[dict[str, Any]] | None = None,
      comments: list[dict[str, Any]] | None = None,
  ):
    self.pulls = pulls if pulls is not None else [make_pr()]
    self.commit_authors = commit_authors or set()
    # Issues created by the PR author, including the PR itself.
    self.issues = (
        issues if issues is not None else [make_issue(PR_NUMBER, is_pr=True)]
    )
    self.comments = comments or []
    self.requested_heads: list[str] = []
    self.posted: list[tuple[int, str]] = []

  def list_open_pulls_from_head(self, head: str) -> list[dict[str, Any]]:
    self.requested_heads.append(head)
    return self.pulls

  def get_pull(self, number: int) -> dict[str, Any]:
    return next(pr for pr in self.pulls if pr["number"] == number)

  def has_commits_by_author(self, branch: str, author: str) -> bool:
    del branch  # Unused.
    return author in self.commit_authors

  def list_issues_by_creator(self, creator: str) -> list[dict[str, Any]]:
    del creator  # Unused.
    return self.issues

  def list_issue_comments(self, issue_number: int) -> list[dict[str, Any]]:
    del issue_number  # Unused.
    return self.comments

  def create_issue_comment(self, issue_number: int, body: str) -> None:
    self.posted.append((issue_number, body))


def run_decide(
    client: FakeGitHubClient, head_owner: str = HEAD_OWNER
) -> Decision:
  return decide(
      client,
      default_branch="main",
      head_owner=head_owner,
      head_branch=HEAD_BRANCH,
      head_sha=HEAD_SHA,
  )


class HelperTest(absltest.TestCase):

  def test_select_triggering_pr_matches_sha(self):
    pulls = [make_pr(number=1, sha="other"), make_pr(number=2, sha=HEAD_SHA)]
    self.assertEqual(select_triggering_pr(pulls, HEAD_SHA)["number"], 2)

  def test_select_triggering_pr_no_match(self):
    self.assertIsNone(select_triggering_pr([make_pr(sha="other")], HEAD_SHA))
    self.assertIsNone(select_triggering_pr([], HEAD_SHA))

  def test_known_author_reason_unknown_author(self):
    for association in ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"):
      with self.subTest(association=association):
        self.assertIsNone(known_author_reason(make_pr(association=association)))

  def test_known_author_reason_known_author(self):
    for association in ("OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"):
      with self.subTest(association=association):
        self.assertIsNotNone(
            known_author_reason(make_pr(association=association))
        )

  def test_known_author_reason_bot(self):
    self.assertIsNotNone(known_author_reason(make_pr(user_type="Bot")))

  def test_has_earlier_pulls_ignores_issues_and_current_pr(self):
    issues = [make_issue(PR_NUMBER, is_pr=True), make_issue(7, is_pr=False)]
    self.assertFalse(has_earlier_pulls(issues, PR_NUMBER))

  def test_has_earlier_pulls_finds_earlier_pr(self):
    issues = [make_issue(PR_NUMBER, is_pr=True), make_issue(7, is_pr=True)]
    self.assertTrue(has_earlier_pulls(issues, PR_NUMBER))

  def test_has_earlier_pulls_ignores_later_pr(self):
    issues = [make_issue(PR_NUMBER, is_pr=True), make_issue(99, is_pr=True)]
    self.assertFalse(has_earlier_pulls(issues, PR_NUMBER))

  def test_has_welcome_comment(self):
    self.assertFalse(has_welcome_comment([]))
    self.assertFalse(has_welcome_comment([{"body": "LGTM"}, {"body": None}]))
    self.assertTrue(has_welcome_comment([{"body": f"{MARKER}\nWelcome!"}]))


class DecideTest(absltest.TestCase):

  def test_welcomes_first_time_contributor(self):
    client = FakeGitHubClient()
    decision = run_decide(client)
    self.assertTrue(decision.should_welcome)
    self.assertEqual(decision.pr_number, PR_NUMBER)
    self.assertEqual(client.requested_heads, [f"{HEAD_OWNER}:{HEAD_BRANCH}"])

  def test_welcomes_author_with_prior_issues_only(self):
    client = FakeGitHubClient(
        issues=[make_issue(PR_NUMBER, is_pr=True), make_issue(3, is_pr=False)]
    )
    self.assertTrue(run_decide(client).should_welcome)

  def test_skips_deleted_head_repo(self):
    client = FakeGitHubClient()
    decision = run_decide(client, head_owner="")
    self.assertFalse(decision.should_welcome)
    self.assertEqual(client.requested_heads, [])

  def test_skips_when_no_matching_pr(self):
    client = FakeGitHubClient(pulls=[make_pr(sha="pushed-since")])
    decision = run_decide(client)
    self.assertFalse(decision.should_welcome)
    self.assertIsNone(decision.pr_number)

  def test_skips_known_author(self):
    client = FakeGitHubClient(pulls=[make_pr(association="MEMBER")])
    self.assertFalse(run_decide(client).should_welcome)

  def test_skips_bot(self):
    client = FakeGitHubClient(pulls=[make_pr(user_type="Bot")])
    self.assertFalse(run_decide(client).should_welcome)

  def test_skips_author_with_commits_on_main(self):
    client = FakeGitHubClient(commit_authors={HEAD_OWNER})
    decision = run_decide(client)
    self.assertFalse(decision.should_welcome)
    self.assertIn("commits", decision.reason)

  def test_skips_author_with_prior_pr(self):
    client = FakeGitHubClient(
        issues=[make_issue(PR_NUMBER, is_pr=True), make_issue(3, is_pr=True)]
    )
    decision = run_decide(client)
    self.assertFalse(decision.should_welcome)
    self.assertIn("earlier PRs", decision.reason)

  def test_welcomes_first_pr_despite_later_pr(self):
    client = FakeGitHubClient(
        issues=[make_issue(PR_NUMBER, is_pr=True), make_issue(99, is_pr=True)]
    )
    self.assertTrue(run_decide(client).should_welcome)

  def test_skips_when_already_welcomed(self):
    client = FakeGitHubClient(comments=[{"body": f"{MARKER}\nWelcome!"}])
    self.assertFalse(run_decide(client).should_welcome)

  def test_decide_does_not_post(self):
    client = FakeGitHubClient()
    run_decide(client)
    self.assertEqual(client.posted, [])

  def test_decide_for_pr_skips_head_lookup(self):
    client = FakeGitHubClient()
    decision = decide_for_pr(client, "main", make_pr())
    self.assertTrue(decision.should_welcome)
    self.assertEqual(decision.pr_number, PR_NUMBER)
    self.assertEqual(client.requested_heads, [])


class MainTest(absltest.TestCase):

  def run_main(self, client: FakeGitHubClient, *argv: str) -> str:
    """Runs main() with `client` as the GitHub client and returns stdout."""
    stdout = io.StringIO()
    with (
        patch.object(first_interaction, "GhCliClient", return_value=client),
        contextlib.redirect_stdout(stdout),
    ):
      self.assertEqual(main(list(argv)), 0)
    return stdout.getvalue()

  def test_pr_requires_dry_run(self):
    with (
        self.assertRaises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
      main([f"--pr={PR_NUMBER}"])

  def test_requires_pr_or_head(self):
    with (
        self.assertRaises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
      main(["--dry-run", "--head-branch=x"])

  def test_pr_dry_run_would_welcome(self):
    client = FakeGitHubClient()
    out = self.run_main(client, f"--pr={PR_NUMBER}", "--dry-run")
    self.assertIn(f"PR #{PR_NUMBER}: {HEAD_OWNER} is a first-time", out)
    self.assertIn("would have posted", out)
    self.assertEqual(client.posted, [])
    self.assertEqual(client.requested_heads, [])

  def test_pr_dry_run_would_not_welcome(self):
    client = FakeGitHubClient(commit_authors={HEAD_OWNER})
    out = self.run_main(client, f"--pr={PR_NUMBER}", "--dry-run")
    self.assertIn("already has commits on main", out)
    self.assertIn("would not have posted", out)
    self.assertEqual(client.posted, [])

  def test_head_dry_run_does_not_post(self):
    client = FakeGitHubClient()
    out = self.run_main(
        client,
        f"--head-owner={HEAD_OWNER}",
        f"--head-branch={HEAD_BRANCH}",
        f"--head-sha={HEAD_SHA}",
        "--dry-run",
    )
    self.assertIn("would have posted", out)
    self.assertEqual(client.posted, [])

  def test_head_posts_welcome(self):
    client = FakeGitHubClient()
    self.run_main(
        client,
        f"--head-owner={HEAD_OWNER}",
        f"--head-branch={HEAD_BRANCH}",
        f"--head-sha={HEAD_SHA}",
    )
    self.assertEqual(
        client.posted, [(PR_NUMBER, first_interaction.WELCOME_MESSAGE)]
    )


def completed(stdout: Any) -> subprocess.CompletedProcess[str]:
  return subprocess.CompletedProcess(
      args=[], returncode=0, stdout=json.dumps(stdout)
  )


@patch("subprocess.run")
class GhCliClientTest(absltest.TestCase):

  def test_get_passes_explicit_method_and_query_fields(self, mock_run):
    mock_run.return_value = completed([{"sha": "abc"}])

    self.assertTrue(GhCliClient("o/r").has_commits_by_author("main", "newuser"))

    cmd = mock_run.call_args.args[0]
    self.assertEqual(
        cmd,
        [
            "gh",
            "api",
            "--method",
            "GET",
            "repos/o/r/commits",
            "-f",
            "sha=main",
            "-f",
            "author=newuser",
            "-f",
            "per_page=1",
        ],
    )
    self.assertTrue(mock_run.call_args.kwargs["check"])

  def test_has_commits_by_author_none(self, mock_run):
    mock_run.return_value = completed([])
    self.assertFalse(
        GhCliClient("o/r").has_commits_by_author("main", "newuser")
    )

  def test_paginated_list_parses_one_item_per_line(self, mock_run):
    # With `--paginate --jq '.[]'`, gh prints each item of every page as a
    # line of compact JSON.
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"number":1}\n{"number":2}\n{"number":3}\n',
    )

    issues = GhCliClient("o/r").list_issues_by_creator("newuser")

    self.assertEqual([issue["number"] for issue in issues], [1, 2, 3])
    cmd = mock_run.call_args.args[0]
    self.assertEqual(
        cmd[:5], ["gh", "api", "--method", "GET", "repos/o/r/issues"]
    )
    self.assertIn("creator=newuser", cmd)
    self.assertEqual(cmd[-3:], ["--paginate", "--jq", ".[]"])

  def test_paginated_list_no_results(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=""
    )
    self.assertEqual(GhCliClient("o/r").list_issue_comments(PR_NUMBER), [])

  def test_create_issue_comment_posts_body_verbatim(self, mock_run):
    mock_run.return_value = completed({"id": 1})
    body = "@not-a-file\nline two"

    GhCliClient("o/r").create_issue_comment(PR_NUMBER, body)

    cmd = mock_run.call_args.args[0]
    self.assertEqual(
        cmd,
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/o/r/issues/{PR_NUMBER}/comments",
            "-f",
            f"body={body}",
        ],
    )

  def test_gh_failure_raises(self, mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(1, ["gh"])
    with self.assertRaises(subprocess.CalledProcessError):
      GhCliClient("o/r").list_open_pulls_from_head("newuser:branch")


if __name__ == "__main__":
  absltest.main()
