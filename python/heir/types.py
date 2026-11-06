"""Constructors for common MLIR and HEIR types.

All functions use the current context unless `context` is given.

Example:

  with heir.context():
    t = heir.types.secret(heir.types.tensor([1024], heir.types.i16()))
    # !secret.secret<tensor<1024xi16>>
"""

from typing import Optional, Sequence

from . import ir

__all__ = [
    "bf16",
    "f16",
    "f32",
    "f64",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "index",
    "integer",
    "is_secret",
    "mod_arith",
    "secret",
    "secret_value_type",
    "tensor",
]

Context = Optional[ir.Context]


def integer(width: int, *, context: Context = None) -> ir.IntegerType:
  """A signless integer type of the given bit width, e.g. i32."""
  return ir.IntegerType.get_signless(width, context=context)


def i1(*, context: Context = None) -> ir.IntegerType:
  return integer(1, context=context)


def i8(*, context: Context = None) -> ir.IntegerType:
  return integer(8, context=context)


def i16(*, context: Context = None) -> ir.IntegerType:
  return integer(16, context=context)


def i32(*, context: Context = None) -> ir.IntegerType:
  return integer(32, context=context)


def i64(*, context: Context = None) -> ir.IntegerType:
  return integer(64, context=context)


def index(*, context: Context = None) -> ir.IndexType:
  return ir.IndexType.get(context=context)


def bf16(*, context: Context = None) -> ir.BF16Type:
  return ir.BF16Type.get(context=context)


def f16(*, context: Context = None) -> ir.F16Type:
  return ir.F16Type.get(context=context)


def f32(*, context: Context = None) -> ir.F32Type:
  return ir.F32Type.get(context=context)


def f64(*, context: Context = None) -> ir.F64Type:
  return ir.F64Type.get(context=context)


def tensor(shape: Sequence[int], element_type: ir.Type) -> ir.RankedTensorType:
  """A ranked tensor type, e.g. tensor([4, 4], f32()) is tensor<4x4xf32>."""
  return ir.RankedTensorType.get(list(shape), element_type)


def secret(value_type: ir.Type) -> ir.Type:
  """A `!secret.secret<value_type>` type, marking `value_type` as encrypted."""
  return ir.Type.parse(f"!secret.secret<{value_type}>", value_type.context)


def is_secret(type_: ir.Type) -> bool:
  """Returns whether `type_` is a `!secret.secret<...>` type."""
  return str(type_).startswith("!secret.secret<")


def secret_value_type(type_: ir.Type) -> ir.Type:
  """Returns `T` for a `!secret.secret<T>` type."""
  text = str(type_)
  if not is_secret(type_):
    raise ValueError(f"not a secret type: {text}")
  return ir.Type.parse(text[len("!secret.secret<") : -1], type_.context)


def mod_arith(
    modulus: int,
    storage_type: Optional[ir.IntegerType] = None,
    *,
    context: Context = None,
) -> ir.Type:
  """A `!mod_arith.int<modulus : storage_type>` type (integers mod `modulus`).

  Args:
    modulus: the modulus.
    storage_type: the integer type used to store values. Defaults to the
      smallest power-of-two width (at least 8) that fits `modulus` with a spare
      bit.
    context: the context to use (defaults to the current context).
  """
  if modulus < 2:
    raise ValueError(f"modulus must be at least 2, got {modulus}")
  if storage_type is None:
    width = 8
    while width < modulus.bit_length() + 1:
      width *= 2
    storage_type = integer(width, context=context)
  return ir.Type.parse(
      f"!mod_arith.int<{modulus} : {storage_type}>",
      context if context is not None else storage_type.context,
  )
