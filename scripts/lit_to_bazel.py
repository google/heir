from collections import deque
import os
import pathlib

import fire


# a sentinel for a bash pipe
PIPE = "|"
OUT_REDIRECT = ">"
IN_REDIRECT = "<"
RUN_PREFIX = "// RUN:"


def strip_run_prefix(line):
  if RUN_PREFIX in line:
    return line.split(RUN_PREFIX)[1]
  return line


def convert_to_run_commands(run_lines):
  run_lines = deque(run_lines)
  cmds = []
  current_command = ""
  while run_lines:
    line = run_lines.popleft()
    if RUN_PREFIX not in line:
      continue

    line = strip_run_prefix(line)

    if "|" in line:
      first, second = line.split("|", maxsplit=1)
      current_command += " " + first.strip()
      cmds.append(current_command.strip())
      current_command = ""
      cmds.append(PIPE)
      run_lines.appendleft(RUN_PREFIX + " " + second.strip())
      continue

    # redirecting to a file implicitly ends the command on that line
    if OUT_REDIRECT in line or IN_REDIRECT in line:
      cmds.append(line.strip())
      current_command = ""
      continue

    if line.strip().endswith("\\"):
      current_command += " " + line.replace("\\", "").strip()
      continue

    current_command += line
    cmds.append(current_command.strip())
    current_command = ""

  return cmds


def lit_to_bazel(
    lit_test_file: str,
    git_root: str = "",
):
  """A helper CLI that converts MLIR test files to bazel run commands.

  Args:
    lit_test_file: The lit test file that should be converted to a bazel run
      command.
  """

  if not git_root:
    git_root = pathlib.Path(__file__).parent.parent
    if not os.path.isdir(git_root / ".git"):
      raise RuntimeError(f"Could not find git root, looked at {git_root}")
  # if git root is manually specified, just trust it

  if not lit_test_file:
    raise ValueError("lit_test_file must be provided")

  if not os.path.isfile(lit_test_file):
    raise ValueError("Unable to find lit_test_file '%s'" % lit_test_file)

  run_lines = []
  with open(lit_test_file, "r") as f:
    for line in f:
      if "// RUN:" in line:
        run_lines.append(line)

  commands = convert_to_run_commands(run_lines)
  commands = [x for x in commands if "FileCheck" not in x]
  # remove consecutive and trailing pipes
  if commands[-1] == PIPE:
    commands.pop()
  deduped_commands = []
  for command in commands:
    if command == PIPE and deduped_commands[-1] == PIPE:
      continue
    deduped_commands.append(command)

  joined = " ".join(deduped_commands)
  # I would consider using bazel-bin/tools/heir-opt, but the yosys
  # requirement requires additional env vars to be set for the yosys and ABC
  # paths, which is not yet worth doing for this script.
  joined = joined.replace(
      "heir-opt",
      "bazel run --noallow_analysis_cache_discard //tools:heir-opt --",
  )
  joined = joined.replace(
      "heir-translate", f"{git_root}/bazel-bin/tools/heir-translate"
  )
  joined = joined.replace("%s", str(pathlib.Path(lit_test_file).absolute()))
  print(joined)


if __name__ == "__main__":
  fire.Fire(lit_to_bazel)
