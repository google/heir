"""Classes for running heir-opt and heir-translate."""

import os
from pathlib import Path
import subprocess


class CLIBackend:

  def __init__(self, binary_path: str | Path):
    self.binary_path = binary_path

  def run_binary(self, options, input) -> str:
    """Run the binary on the input.

    Args:
        options: The options to pass to the binary.
        input: The input to pass to the binary.

    Returns:
        The stdout of the executed process.
    """
    completed_process = subprocess.run(
        [os.path.abspath(self.binary_path)] + options,
        input=input,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed_process.returncode != 0:
      message = f"Error running {self.binary_path}. "
      message += "stderr was:\n\n" + completed_process.stderr + "\n\n"
      message += "input was:\n\n" + input + "\n\n"
      message += "options were:\n\n" + " ".join([str(x) for x in options])
      raise ValueError(message)

    return completed_process.stdout

  def run_binary_stderr(self, options, input) -> tuple[str, str]:
    """Run the binary on the input.

    Args:
        options: The options to pass to the binary.
        input: The input to pass to the binary.

    Returns:
        The stdout and stderr of the executed process.
    """
    completed_process = subprocess.run(
        [os.path.abspath(self.binary_path)] + options,
        input=input,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed_process.returncode != 0:
      message = f"Error running {self.binary_path}. "
      message += "stderr was:\n\n" + completed_process.stderr + "\n\n"
      message += "input was:\n\n" + input + "\n\n"
      message += "options were:\n\n" + " ".join([str(x) for x in options])
      raise ValueError(message)

    return completed_process.stdout, completed_process.stderr


class HeirOptBackend(CLIBackend):

  def __init__(self, binary_path="heir-opt"):
    """Initialize heir-opt with a path to the heir-opt binary.

    If not specified, will assume heir-opt is on the path.
    """
    super().__init__(binary_path)


class HeirTranslateBackend(CLIBackend):

  def __init__(self, binary_path="heir-translate"):
    """Initialize heir-translate with a path to the heir-translate binary.

    If not specified, will assume heir-translate is on the path.
    """
    super().__init__(binary_path)
