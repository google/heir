"""Updates a designated GitHub issue comment with coverage results.

This script uses the `gh` CLI to find a comment on a specific issue
containing a magic string, and updates it with new content (e.g., a
markdown table). If no such comment is found, it creates a new one.
"""

import os
import subprocess
import sys
import json


class GitHubApi:
  """Wraps the subcommands for specific tasks and returns raw responses."""

  def __init__(self, repo="google/heir", issue_number=2923):
    self.repo = repo
    self.issue_number = issue_number

  def run_gh_command(self, cmd):
    if "GH_TOKEN" not in os.environ and "GITHUB_TOKEN" not in os.environ:
      print(
          "Warning: GH_TOKEN or GITHUB_TOKEN not found in environment. gh"
          " command may fail."
      )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=False
    )
    return result

  def get_comments(self):
    cmd = f"gh api repos/{self.repo}/issues/{self.issue_number}/comments"
    result = self.run_gh_command(cmd)
    if result.returncode != 0:
      raise Exception(f"Error listing comments: {result.stderr}")
    return json.loads(result.stdout)

  def update_comment(self, comment_id, body_file):
    cmd = (
        f"gh api -X PATCH repos/{self.repo}/issues/comments/{comment_id} -F"
        f" body=@{body_file}"
    )
    result = self.run_gh_command(cmd)
    if result.returncode != 0:
      raise Exception(f"Error updating comment: {result.stderr}")
    return result.stdout

  def create_comment(self, body_file):
    cmd = f"gh issue comment {self.issue_number} --body-file {body_file}"
    result = self.run_gh_command(cmd)
    if result.returncode != 0:
      raise Exception(f"Error creating comment: {result.stderr}")
    return result.stdout


class GithubIssueUpdater:
  """Updates a GitHub issue comment with coverage results."""

  def __init__(self, api=None, magic_string="<!-- TORCH_COVERAGE_TABLE -->"):
    self.api = api or GitHubApi()
    self.magic_string = magic_string

  def find_comment(self):
    try:
      comments = self.api.get_comments()
      for comment in comments:
        if self.magic_string in comment["body"]:
          return comment["id"]
    except Exception as e:
      print(e)
    return None

  def run(self, table_file):
    with open(table_file, "r") as f:
      table_content = f.read()

    comment_id = self.find_comment()
    temp_file = "temp_body.md"
    with open(temp_file, "w") as f:
      f.write(self.magic_string + "\n\n" + table_content)

    try:
      if comment_id:
        print(f"Found existing comment {comment_id}. Updating...")
        self.api.update_comment(comment_id, temp_file)
      else:
        print("No existing comment found. Creating new one...")
        self.api.create_comment(temp_file)
      print("Successfully updated issue comment.")
    except Exception as e:
      print(f"Error: {e}")
      sys.exit(1)
    finally:
      if os.path.exists(temp_file):
        os.remove(temp_file)


def main():
  if len(sys.argv) < 2:
    print("Usage: python update_issue_comment.py <table_file>")
    sys.exit(1)

  table_file = sys.argv[1]
  updater = GithubIssueUpdater()
  updater.run(table_file)


if __name__ == "__main__":
  main()
