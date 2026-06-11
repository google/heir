from pathlib import Path
from typing import Optional
import importlib.resources
import importlib.util
import os
import pathlib
import sysconfig
import sys

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


def is_pip_installed(debug: bool = False) -> bool:
  """Return true if heir is installed via pip."""
  debug_active = debug or os.environ.get("OPENFHE_DEBUG") == "1"
  try:
    origin = get_module_origin("heir")
    if debug_active:
      print(
          "HEIRpy Debug (common): get_module_origin('heir') returned:"
          f" {origin}",
          file=sys.stderr,
      )
    if not origin:
      if debug_active:
        print(
            "HEIRpy Debug (common): is_pip_installed result: False (no origin)",
            file=sys.stderr,
        )
      return False
    module_path = Path(origin)
    purelib = Path(sysconfig.get_paths()["purelib"])
    platlib = Path(sysconfig.get_paths()["platlib"])

    res = module_path.is_relative_to(purelib) or module_path.is_relative_to(
        platlib
    )

    if debug_active:
      print(
          f"HEIRpy Debug (common): Module path: {module_path}", file=sys.stderr
      )
      print(f"HEIRpy Debug (common): purelib: {purelib}", file=sys.stderr)
      print(f"HEIRpy Debug (common): platlib: {platlib}", file=sys.stderr)
      print(
          f"HEIRpy Debug (common): is_pip_installed result: {res}",
          file=sys.stderr,
      )

    return res
  except (ModuleNotFoundError, ValueError) as e:
    if debug_active:
      print(
          f"HEIRpy Debug (common): is_pip_installed caught exception: {e}",
          file=sys.stderr,
      )
      print(
          "HEIRpy Debug (common): is_pip_installed result: False",
          file=sys.stderr,
      )
    return False


class BackendWarning:

  def __init__(self, name: str, message: str):
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + f"HEIR Warning ({name}): {message}"
        + Style.RESET_ALL
    )
