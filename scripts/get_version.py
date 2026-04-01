"""Script to calculate the version for the HEIR Python package."""

import argparse
import datetime
import os
import re
import sys
import requests


def get_pypi_versions(package_name):
  """Fetch all published versions for a package from PyPI."""
  url = f"https://pypi.org/pypi/{package_name}/json"
  try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
      data = response.json()
      return list(data.get("releases", {}).keys())
  except Exception as e:
    print(f"Error fetching from PyPI: {e}", file=sys.stderr)
  return []


def get_next_dev_version(package_name):
  """Calculate the next .devN version for today's date."""
  today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y.%m.%d")
  versions = get_pypi_versions(package_name)

  # Find versions matching today's date and the .dev suffix
  pattern = re.compile(rf"^{re.escape(today)}\.dev(\d+)$")
  max_dev = -1

  for v in versions:
    match = pattern.match(v)
    if match:
      max_dev = max(max_dev, int(match.group(1)))

  next_dev = max_dev + 1
  return f"{today}.dev{next_dev}"


def main():
  parser = argparse.ArgumentParser(
      description="Calculate HEIR package version."
  )
  parser.add_argument(
      "--event",
      default="workflow_dispatch",
      help="GitHub event name (e.g., release, workflow_dispatch)",
  )
  parser.add_argument(
      "--ref",
      default="refs/heads/main",
      help="GitHub ref (e.g., refs/heads/main)",
  )
  parser.add_argument("--tag", help="Release tag name")
  parser.add_argument("--package", default="heir_py", help="PyPI package name")
  parser.add_argument(
      "--gha", action="store_true", help="Output for GitHub Actions"
  )

  args = parser.parse_args()

  match args.event:
    case "release":
      if args.tag:
        version = args.tag.lstrip("v")
        should_publish = "true"

    case "workflow_dispatch":
      if args.tag:
        # Manual release of existing tag; use for example when release
        # workflow fails to trigger wheel upload.
        version = args.tag.lstrip("v")
        should_publish = "true"
      elif args.ref == "refs/heads/main":
        # For dev releases
        version = get_next_dev_version(args.package)
        should_publish = "true"

    case _:
      # PRs should not publish
      should_publish = "false"

  if args.gha:
    # Writing to GITHUB_OUTPUT if available
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
      with open(output_file, "a") as f:
        f.write(f"version={version}\n")
        f.write(f"should_publish={should_publish}\n")

  print(f"version={version}")
  print(f"should_publish={should_publish}")


if __name__ == "__main__":
  main()
