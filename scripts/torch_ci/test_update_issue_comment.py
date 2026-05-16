"""Unit tests for update_issue_comment.py."""

import json
import os
import sys
import unittest

from scripts.torch_ci.update_issue_comment import GithubIssueUpdater


class FakeGitHubApi:

  def __init__(self, comments=None):
    self.comments = comments or []
    self.updated_comments = {}
    self.created_comments = []

  def get_comments(self):
    return self.comments

  def update_comment(self, comment_id, body_file):
    with open(body_file, "r") as f:
      self.updated_comments[comment_id] = f.read()
    return "mock_stdout"

  def create_comment(self, body_file):
    with open(body_file, "r") as f:
      self.created_comments.append(f.read())
    return "mock_stdout"


class TestGithubIssueUpdater(unittest.TestCase):

  def test_find_comment_found(self):
    fake_api = FakeGitHubApi(
        [{"id": 123, "body": "<!-- TORCH_COVERAGE_TABLE -->\nTable"}]
    )
    updater = GithubIssueUpdater(api=fake_api)
    comment_id = updater.find_comment()
    self.assertEqual(comment_id, 123)

  def test_find_comment_not_found(self):
    fake_api = FakeGitHubApi([{"id": 123, "body": "Some other comment"}])
    updater = GithubIssueUpdater(api=fake_api)
    comment_id = updater.find_comment()
    self.assertIsNone(comment_id)

  def test_run_update(self):
    fake_api = FakeGitHubApi(
        [{"id": 123, "body": "<!-- TORCH_COVERAGE_TABLE -->\nTable"}]
    )
    updater = GithubIssueUpdater(api=fake_api)
    table_file = "dummy_table.md"
    with open(table_file, "w") as f:
      f.write("New Table")

    updater.run(table_file)
    os.remove(table_file)

    self.assertIn(123, fake_api.updated_comments)
    self.assertIn("New Table", fake_api.updated_comments[123])

  def test_run_create(self):
    fake_api = FakeGitHubApi([])
    updater = GithubIssueUpdater(api=fake_api)
    table_file = "dummy_table.md"
    with open(table_file, "w") as f:
      f.write("New Table")

    updater.run(table_file)
    os.remove(table_file)

    self.assertEqual(len(fake_api.created_comments), 1)
    self.assertIn("New Table", fake_api.created_comments[0])


if __name__ == "__main__":
  unittest.main()
