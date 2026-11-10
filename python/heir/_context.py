"""Helpers for creating contexts and modules."""

import contextlib
import os
from typing import Iterator, Optional, Union

from . import ir


def _current_context() -> Optional[ir.Context]:
  try:
    return ir.Context.current
  except ValueError:
    return None


def _current_location() -> Optional[ir.Location]:
  try:
    return ir.Location.current
  except ValueError:
    return None


@contextlib.contextmanager
def context(
    *, allow_unregistered_dialects: bool = False
) -> Iterator[ir.Context]:
  """Creates and enters a context in which all HEIR dialects are available.

  Also enters an unknown `Location`, so that ops can be built without passing
  `loc=` explicitly.

  Args:
    allow_unregistered_dialects: whether to allow ops from unknown dialects.

  Yields:
    The new context.
  """
  with ir.Context() as ctx, ir.Location.unknown(ctx):
    ctx.allow_unregistered_dialects = allow_unregistered_dialects
    yield ctx


@contextlib.contextmanager
def module() -> Iterator[ir.Module]:
  """Creates an empty module and sets the insertion point to its body.

  Uses the current context if there is one, otherwise enters a new one (see
  `heir.context`).

  Example:

    with heir.module() as m:
      @func.FuncOp.from_py_func(heir.types.secret(heir.types.i32()))
      def double(x):
        ...

  Yields:
    The new module.
  """
  with contextlib.ExitStack() as stack:
    if _current_context() is None:
      stack.enter_context(context())
    elif _current_location() is None:
      stack.enter_context(ir.Location.unknown())
    m = ir.Module.create()
    stack.enter_context(ir.InsertionPoint(m.body))
    yield m


def parse(source: str, *, context: Optional[ir.Context] = None) -> ir.Module:
  """Parses MLIR assembly into a module (in the current context by default)."""
  return ir.Module.parse(source, context=context)


def parse_file(
    path: Union[str, os.PathLike[str]],
    *,
    context: Optional[ir.Context] = None,
) -> ir.Module:
  """Parses an MLIR file into a module (in the current context by default)."""
  with open(path, "r") as f:
    return ir.Module.parse(f.read(), context=context)
