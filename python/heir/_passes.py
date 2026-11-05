"""Helpers for running heir-opt passes and pipelines in-process."""

from typing import Any, Union

from . import ir
from . import passmanager


def _format_option_value(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (list, tuple)):
    return ",".join(_format_option_value(v) for v in value)
  text = str(value)
  if any(c in text for c in " {},"):
    return '"' + text.replace('"', '\\"') + '"'
  return text


class Pass:
  """A heir-opt pass or pass pipeline, with options.

  Option names may use underscores in place of dashes. For example,

    Pass("mlir-to-bgv", ciphertext_degree=1024, encryption_technique_extended=True)

  is equivalent to the heir-opt flag

    --mlir-to-bgv="ciphertext-degree=1024 encryption-technique-extended=true"
  """

  def __init__(self, name: str, /, **options: Any):
    self.name = name
    self.options = options

  def __str__(self) -> str:
    if not self.options:
      return self.name
    opts = " ".join(
        f"{key.replace('_', '-')}={_format_option_value(value)}"
        for key, value in self.options.items()
    )
    return f"{self.name}{{{opts}}}"

  def __repr__(self) -> str:
    args = [repr(self.name)] + [f"{k}={v!r}" for k, v in self.options.items()]
    return f"Pass({', '.join(args)})"


PassLike = Union[str, Pass]


def _normalize(p: PassLike) -> str:
  """Accepts a Pass, a textual pass (`name{opts}`), or a heir-opt flag."""
  if isinstance(p, Pass):
    return str(p)
  text = p.strip()
  if text.startswith("-"):
    # heir-opt flag syntax: --name or --name=opts or --name="opts"
    text = text.lstrip("-")
    if "=" in text and ("{" not in text or text.index("=") < text.index("{")):
      name, opts = text.split("=", 1)
      opts = opts.strip()
      if len(opts) >= 2 and opts[0] == opts[-1] and opts[0] in "'\"":
        opts = opts[1:-1]
      text = f"{name}{{{opts}}}"
  return text


def pipeline_string(*passes: PassLike) -> str:
  """Returns the textual pass pipeline (anchored on builtin.module)."""
  if len(passes) == 1 and isinstance(passes[0], str):
    text = passes[0].strip()
    if text.startswith("builtin.module("):
      return text
  return "builtin.module(" + ",".join(_normalize(p) for p in passes) + ")"


def run(
    module: ir.Module,
    *passes: PassLike,
    verify: bool = True,
    print_ir_after_all: bool = False,
    enable_timing: bool = False,
) -> ir.Module:
  """Runs passes (or pipelines) on a module in place, like heir-opt.

  Each pass can be a `Pass`, a textual pass like "mlir-to-bgv{ciphertext-degree=8}",
  or a heir-opt flag like "--mlir-to-bgv=ciphertext-degree=8". A single fully
  specified pipeline "builtin.module(...)" is also accepted.

  Example:

    heir.run(module, "mlir-to-bgv", heir.Pass("scheme-to-openfhe"))

  Args:
    module: the module to transform.
    *passes: the passes to run, in order.
    verify: whether to verify the IR after each pass.
    print_ir_after_all: whether to print the IR after each pass (to stderr).
    enable_timing: whether to print pass timing information (to stderr).

  Returns:
    The (mutated) module, for chaining.

  Raises:
    ValueError: if the pipeline cannot be parsed (e.g., an unknown pass).
    heir.ir.MLIRError: if a pass fails; contains the diagnostics.
  """
  if not passes:
    raise ValueError("expected at least one pass")
  with module.context:
    pm = passmanager.PassManager.parse(pipeline_string(*passes))
  pm.enable_verifier(verify)
  if print_ir_after_all:
    pm.enable_ir_printing()
  if enable_timing:
    pm.enable_timing()
  pm.run(module.operation)
  return module
