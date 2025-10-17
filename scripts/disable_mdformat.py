"""A script to ensure markdown files ignored by Google's markdown formatter.

Google's internal formatter doesn't support yaml frontmatter, and has different
conventions than the OSS formatter.

Puts the command at the end of each file.
"""

import sys
import subprocess
from pathlib import Path

# --- Configuration ---
LINE_TO_ADD = "<!-- mdformat global-off -->"
FILE_GLOB_PATTERN = "*.md"
# ---------------------


def find_git_root():
  """
  Finds the root directory of the current git repository.
  Exits if 'git' is not found or this is not a git repo.
  """
  try:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return result.stdout.strip()
  except FileNotFoundError:
    print("Error: 'git' command not found.", file=sys.stderr)
    sys.exit(1)
  except subprocess.CalledProcessError:
    print("Error: Could not find git repository root.", file=sys.stderr)
    print("Are you sure you are inside a git repository?", file=sys.stderr)
    sys.exit(1)


def get_non_ignored_files(root_dir, glob_pattern):
  """
  Uses 'git ls-files' to get a list of all files in the repo
  that match the glob, while respecting .gitignore.
  """
  cmd = [
      "git",
      "ls-files",
      "--cached",
      "--others",
      "--exclude-standard",
      "--",
      glob_pattern,
  ]

  print(f"Finding non-ignored '{glob_pattern}' files...")

  try:
    result = subprocess.run(
        cmd,
        cwd=root_dir,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    file_list = [line for line in result.stdout.splitlines() if line]
    return file_list
  except subprocess.CalledProcessError as e:
    print("Error: 'git ls-files' failed.", file=sys.stderr)
    print(f"Details: {e.stderr}", file=sys.stderr)
    sys.exit(1)


def add_footer_to_files(root_dir, relative_file_list):
  """
  Iterates over the list of relative file paths, checks if the
  comment exists anywhere, and appends it if not.
  """
  count = 0
  root_path = Path(root_dir)

  print(f"Scanning {len(relative_file_list)} files...")

  for rel_path in relative_file_list:
    file_path = root_path / rel_path

    if not file_path.is_file():
      print(f"Skipping (not a file): {rel_path}")
      continue

    try:
      # Check if the line exists anywhere in the file
      content = file_path.read_text(encoding="utf-8")
      if LINE_TO_ADD in content:
        # print(f"Skipping (already present): {rel_path}")
        continue

      # If we're here, the line was not found.
      # Open in 'a' (append) mode to add to the end.
      print(f"Adding footer to: {rel_path}")
      with file_path.open("a", encoding="utf-8") as f:
        if content and not content.endswith("\n"):
          f.write("\n")
        f.write(f"{LINE_TO_ADD}\n")

      count += 1

    except UnicodeDecodeError:
      print(f"Skipping (not utf-8): {rel_path}")
    except Exception as e:
      print(f"Error processing {rel_path}: {e}", file=sys.stderr)

  print(f"\nDone. Footer added to {count} files.")
  if count > 0:
    sys.exit(1)


if __name__ == "__main__":
  git_root = find_git_root()
  file_list = get_non_ignored_files(git_root, FILE_GLOB_PATTERN)

  if file_list:
    add_footer_to_files(git_root, file_list)
  else:
    print(f"No non-ignored '{FILE_GLOB_PATTERN}' files found.")
