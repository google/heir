import subprocess
import os

import fire

from scripts.lit_to_bazel import (
    get_command_without_bazel_prefix,
    normalize_lit_test_file_arg,
)


def is_command_executable(command):
  """Checks if a given command is executable on the system's PATH."""
  path_env = os.environ.get("PATH")
  treat_as_literal = os.path.sep in command or not path_env
  if os.path.altsep:
    treat_as_literal = treat_as_literal or os.path.altsep in command

  if treat_as_literal:
    return os.path.isfile(command) and os.access(command, os.X_OK)

  for directory in path_env.split(os.pathsep):
    full_path = os.path.join(directory, command)
    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
      return True
  return False


def get_doctest_input(lit_test_file) -> str:
  """Return the mlir part of the given file, omitting lit/filecheck stuff."""
  mlir_lines = []
  with open(lit_test_file, "r") as f:
    for line in f:
      if "// RUN" in line or "// CHECK" in line:
        continue
      mlir_lines.append(line)

  return "".join(mlir_lines).strip()


def lit_to_doc(
    lit_test_file: str,
    git_root: str = "",
):
  """A helper CLI that converts MLIR test files to bazel run commands.

  Args:
    lit_test_file: The lit test file that should be converted to a bazel run
      command.
  """
  lit_test_file = normalize_lit_test_file_arg(lit_test_file)
  mlir_input = get_doctest_input(lit_test_file)
  command = get_command_without_bazel_prefix(lit_test_file)
  command = command.replace("%s", lit_test_file)

  if not is_command_executable("heir-opt"):
    command = command.replace("heir-opt", "bazel-bin/tools/heir-opt")

  result = subprocess.run(
      command, shell=True, capture_output=True, text=True, check=True
  )
  mlir_output = result.stdout.strip()

  if not is_command_executable("heir-opt"):
    command = command.replace("bazel-bin/tools/heir-opt", "heir-opt")

  return f"""#### Example

Command: `{command}`

Input:

```mlir
{mlir_input}
```

Output:

```mlir
{mlir_output}
```
"""


if __name__ == "__main__":
  fire.Fire(lit_to_doc)
