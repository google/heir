"""Library for converting lit tests to bazel run commands."""

import collections
import os
import pathlib
import shutil
import subprocess

# a sentinel for a bash pipe
PIPE = "|"
OUT_REDIRECT = ">"
IN_REDIRECT = "<"
RUN_PREFIX = "// RUN:"

DEFAULT_TOOL_PREFIX = "bazel run --noallow_analysis_cache_discard //tools"
DEFAULT_BASE_PATH = ""
# fmt: off
DEFAULT_HEIR_TRANSLATE_PREFIX = "{git_root}/bazel-bin/tools/heir-translate"
# fmt: on


def strip_run_prefix(line):
  if RUN_PREFIX in line:
    return line.split(RUN_PREFIX)[1]
  return line


def convert_to_run_commands(run_lines):
  """Converts RUN lines to a list of commands."""
  run_lines = collections.deque(run_lines)
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


def normalize_lit_test_file_arg(
    lit_test_file: str, base_path: str = DEFAULT_BASE_PATH
) -> str:
  """Normalizes the lit test file argument."""
  # Convert a bazel test target into a file path.
  # Bazel test targets look like
  #
  #   //tests/path/to/dir:target.mlir.test
  #   tests/path/to/dir:target.mlir.test
  #
  # and are converted to
  #
  #   test/path/to/dir/target.mlir
  #
  if lit_test_file.startswith("//") or lit_test_file.endswith(".mlir.test"):
    print(f"Converting bazel test target {lit_test_file} to file path")

  if lit_test_file.startswith("//"):
    lit_test_file = lit_test_file[2:]
  if lit_test_file.endswith(".mlir.test"):
    lit_test_file = lit_test_file[:-5]

  components = lit_test_file.split(":")
  if len(components) > 1:
    lit_test_file = components[0] + "/" + components[1]

  if not os.path.exists(lit_test_file):
    # Try relative to base_path if we are at workspace root
    alt_path = os.path.join(base_path, lit_test_file)
    if os.path.exists(alt_path):
      lit_test_file = alt_path
    else:
      # Try removing base_path/ prefix if we are in that dir
      prefix = base_path + "/"
      if lit_test_file.startswith(prefix):
        alt_path = lit_test_file[len(prefix) :]
        if os.path.exists(alt_path):
          lit_test_file = alt_path

  return lit_test_file


def get_command_without_bazel_prefix(lit_test_file) -> str:
  """Gets the command without the bazel prefix."""
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
  return joined


def lit_to_bazel(
    lit_test_file: str,
    git_root: str = "",
    run: bool = False,
    debug_dir: str = "/tmp/mlir",
    tool_prefix: str = DEFAULT_TOOL_PREFIX,
    heir_translate_prefix: str = DEFAULT_HEIR_TRANSLATE_PREFIX,
    base_path: str = DEFAULT_BASE_PATH,
):
  """A helper CLI that converts MLIR test files to bazel run commands.

  Args:
    lit_test_file: The lit test file that should be converted to a bazel run
      command.
    git_root: The git root directory.
    run: Whether to run the command.
    debug_dir: The debug directory.
    tool_prefix: The prefix for the heir-opt tool.
    heir_translate_prefix: The prefix for the heir-translate tool.
    base_path: The base path for tests.
  """
  if not lit_test_file:
    raise ValueError("lit_test_file must be provided")

  lit_test_file = normalize_lit_test_file_arg(
      lit_test_file, base_path=base_path
  )

  if not os.path.isfile(lit_test_file):
    raise ValueError("Unable to find lit_test_file '%s'" % lit_test_file)

  command = get_command_without_bazel_prefix(lit_test_file)

  command = command.replace(
      "heir-opt",
      f"{tool_prefix}:heir-opt --",
  )

  command = command.replace(
      "heir-translate",
      heir_translate_prefix.format(git_root=git_root),
  )

  command = command.replace("%s", str(pathlib.Path(lit_test_file).absolute()))

  if run:
    # delete the debug dir if it exists
    if os.path.isdir(debug_dir):
      shutil.rmtree(debug_dir)
    command += (
        f" --mlir-print-ir-after-all --mlir-print-ir-tree-dir={debug_dir}"
    )
    print(command)

    # run the command
    subprocess.run(command, shell=True, check=True)
  else:
    print(command)
