"""Compute statistics about (non-Googler, non-bot) authors of commits."""

import collections
import csv
import datetime
import sys
import time
import requests

defaultdict = collections.defaultdict
datetime = datetime.datetime

# Constants
GITHUB_API_URL = "https://api.github.com"
REPO = "google/heir"  # replace with the desired repo
# TOKEN = "your_github_token_here"

# Define email patterns to exclude
excluded_email_patterns = ["dependabot", "no-reply@google.com"]
# use this for only non-Google-employed authors
# excluded_patterns = ["google.com", "j2kun", "kun.jeremy", "dependabot"]

# Define commit message patterns to exclude
excluded_message_patterns = [
    "Integrate LLVM at llvm/llvm-project@",
    "dependabot",
]


def should_exclude(commit):
  """Checks if a commit should be excluded."""
  email = commit.get("email")
  message = commit.get("message")
  exclude_for_email = (
      any(pattern in email for pattern in excluded_email_patterns)
      if email
      else False
  )
  exclude_for_message = (
      any(pattern in message for pattern in excluded_message_patterns)
      if message
      else False
  )
  return exclude_for_email or exclude_for_message


def fetch_commits(repo):
  """Fetches all commits from a repository, handling pagination."""
  url = f"{GITHUB_API_URL}/repos/{repo}/commits"
  # headers = {"Authorization": f"token {TOKEN}"}
  params = {"per_page": 100}  # Fetch up to 100 commits per page
  commits = []
  page = 0

  while url:
    print(f"Fetching page {page} of commits...")
    time.sleep(1)
    response = requests.get(url, params=params)
    # for token-based auth:
    # response = requests.get(url, params=params, headers=headers)

    response.raise_for_status()  # Raise an error if the request failed
    data = response.json()
    commits.extend(data)

    # Check if there's another page of results
    if "next" in response.links:
      url = response.links["next"]["url"]
      page += 1
    else:
      url = None

  return commits


def extract_commits(all_commits):
  """Extracts relevant commit information and filters out excluded commits."""
  extracted = []
  for commit in all_commits:
    commit_data = commit["commit"]
    author_data = commit_data.get("author", {})
    message = commit_data.get("message")
    date_str = author_data.get("date")
    email = author_data.get("email")
    name = author_data.get("name")

    if should_exclude({"email": email, "message": message}):
      continue

    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    extracted.append({
        "date": date,
        "author": name,
        "email": email,
        "message": message.split("\n")[0],
    })

  return extracted


def compute_stats(commits):
  """Computes monthly author and commit statistics."""
  monthly_authors = defaultdict(set)
  author_commit_count = defaultdict(int)
  monthly_commits = defaultdict(int)

  for commit in commits:
    name = commit["author"]
    date = commit["date"]
    year_month = date.strftime("%Y-%m")

    monthly_authors[year_month].add(name)
    monthly_commits[year_month] += 1
    author_commit_count[name] += 1

  monthly_stats = []
  for date, authors in monthly_authors.items():
    monthly_stats.append({
        "date": date,
        "distinct authors": len(authors),
        "commits": monthly_commits[date],
    })

  return monthly_stats


if __name__ == "__main__":
  extracted_commits = []
  stats = []
  try:
    extracted_commits = extract_commits(fetch_commits(REPO))
    stats = compute_stats(extracted_commits)
  except requests.exceptions.RequestException as e:
    print(f"Error fetching commits: {e}")
    sys.exit(1)

  with open("all_commits.csv", "w") as f:
    writer = csv.DictWriter(
        f, fieldnames=["date", "author", "email", "message"]
    )
    writer.writeheader()
    writer.writerows(extracted_commits)

  with open("monthly_author_stats.csv", "w") as f:
    writer = csv.DictWriter(
        f, fieldnames=["date", "distinct authors", "commits"]
    )
    writer.writeheader()
    writer.writerows(stats)
