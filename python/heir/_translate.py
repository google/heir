"""Helpers for running heir-translate translations (code emitters) in-process."""

from typing import Any, Dict, Union

from . import ir
from ._mlir_libs import _heir


class TranslationError(RuntimeError):
  """Raised when a translation fails; the message contains the diagnostics."""


def translations() -> Dict[str, str]:
  """Returns a mapping from translation name to its description."""
  return dict(_heir.translations())


def _format_flag(name: str, value: Any) -> str:
  flag = "--" + name.replace("_", "-")
  if value is True:
    return flag
  if value is False:
    return flag + "=false"
  return f"{flag}={value}"


def translate(
    op: Union[ir.Module, ir.Operation, ir.OpView],
    translation: str,
    *flags: str,
    **options: Any,
) -> str:
  """Runs a heir-translate translation and returns its output.

  The input is not modified (the translation runs on a copy).

  Example:

    heir.translate(module, "emit-openfhe-pke",
                   openfhe_include_type="install-relative")

  Args:
    op: the module (or operation) to translate.
    translation: the translation name, e.g. "emit-openfhe-pke" (a leading "--"
      is accepted). See `heir.translations()` for the available ones.
    *flags: raw heir-translate flags, e.g. "--weights-file=weights.bin".
    **options: heir-translate options as keyword arguments; underscores are
      converted to dashes, and `True` passes the flag without a value.

  Returns:
    The translated output.

  Raises:
    TranslationError: if the translation or its options are invalid, or the
      translation fails.
  """
  if isinstance(op, ir.Module):
    op = op.operation
  elif isinstance(op, ir.OpView):
    op = op.operation
  args = list(flags) + [_format_flag(k, v) for k, v in options.items()]
  try:
    return _heir.translate(op, translation.lstrip("-"), args)
  except RuntimeError as e:
    raise TranslationError(str(e)) from None
