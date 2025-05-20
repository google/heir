from pathlib import Path
from typing import Optional
import importlib.resources
import importlib.util
import os
import pathlib
import sysconfig

from heir.interfaces import CompilationResult, EncValue

from colorama import Fore, Style


def find_above(dirname: str) -> Path | None:
  path = pathlib.Path(__file__).resolve()
  matching = [p for p in path.parents if (p / dirname).exists()]
  return matching[-1] if matching else None


def get_repo_root() -> Path | None:
  default = find_above("bazel-bin")
  found = os.getenv("HEIR_REPO_ROOT_MARKER")
  return Path(found) if found else default


def strip_and_verify_eval_arg_consistency(
    compilation_result: CompilationResult, *args, **kwargs
):
  stripped_args = []
  for i, arg in enumerate(args):
    if i in compilation_result.secret_args:
      if not isinstance(arg, EncValue):
        raise ValueError(f"Expected EncValue for argument {i}, got {type(arg)}")
      # check that the name matches:
      if not arg.identifier == compilation_result.arg_names[i]:
        raise ValueError(
            "Expected EncValue for identifier"
            f" {compilation_result.arg_names[i]}, got EncValue for"
            f" {arg.identifier}"
        )
      # strip the identifier
      stripped_args.append(arg.value)
    else:
      if isinstance(arg, EncValue):
        raise ValueError(
            f"Expected non-EncValue for argument {i}, "
            f"got EncValue for {arg.identifier}"
        )
      stripped_args.append(arg)

  # How to deal with kwargs?
  if kwargs:
    raise NotImplementedError(
        "HEIR's Python Frontend currently doesn't support passing values as"
        " keyword arguments."
    )

  return stripped_args, kwargs


def get_module_origin(module_name: str) -> Optional[str]:
  """Return the origin of the module as a path."""
  spec = importlib.util.find_spec(module_name)
  if spec is None:
    raise ValueError(f"Module '{module_name}' not found.")
  return spec.origin


def is_pip_installed() -> bool:
  """Return true if heir is installed via pip."""
  try:
    module_path = get_module_origin("heir") or ""
    # purelib gives the environment-specific location of site-packages
    # (for venv) or dist-packages (for system-wide installation) of
    # non-builtin modules.
    return module_path.startswith(sysconfig.get_paths()["purelib"])
  except ModuleNotFoundError:
    return False


class BackendWarning:

  def __init__(self, name: str, message: str):
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + f"HEIR Warning ({name}): {message}"
        + Style.RESET_ALL
    )
