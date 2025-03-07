"""Defines Python type annotations for MLIR types."""

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar, TypeVarTuple, get_args, get_origin

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


operator_error_message = (
    "MLIRTypeAnnotation should only be used for annotations."
)


class MLIRTypeAnnotation(ABC):

  @staticmethod
  @abstractmethod
  def numba_str():
    raise NotImplementedError(
        "No numba type exists for a generic MLIRTypeAnnotation"
    )

  def __add__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __sub__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __mul__(self, other) -> Self:
    raise RuntimeError(operator_error_message)


class Secret(Generic[T], MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    raise NotImplementedError("No numba type exists for a generic Secret")


class Tensor(Generic[*Ts], MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    raise NotImplementedError("No numba type exists for a generic Tensor")


class F32(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_str():
    return "float32"


class F64(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_str():
    return "float64"


class I1(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "bool"


class I4(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "int4"


class I8(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "int8"


class I16(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "int16"


class I32(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "int32"


class I64(MLIRTypeAnnotation):

  @staticmethod
  def numba_str():
    return "int64"


# Helper functions


def to_numba_str(type) -> str:
  if get_origin(type) == Secret:
    raise TypeError(
        "Secret type should not appear inside another type annotation."
    )

  if get_origin(type) == Tensor:
    args = get_args(type)
    inner_type = args[-1]
    if get_origin(inner_type) == Tensor:
      raise TypeError("Nested Tensors are not yet supported.")
    # Cf. https://numba.pydata.org/numba-doc/dev/reference/types.html#arrays
    return f"{to_numba_str(inner_type)}[{','.join([':'] * (len(args) - 1))}]"

  if issubclass(type, MLIRTypeAnnotation):
    return type.numba_str()

  raise TypeError(f"Unsupported type annotation: {type}, {get_origin(type)}")


def parse_annotations(annotations):
  if not annotations:
    raise TypeError("Function is missing type annotations.")
  signature = ""
  secret_args = []
  rettype = None
  for idx, (name, arg_type) in enumerate(annotations.items()):
    if name == "return":
      # A user may not annotate the return type as secret
      rettype = to_numba_str(arg_type)
      continue
    if get_origin(arg_type) == Secret:
      assert len(get_args(arg_type)) == 1
      numba_arg = to_numba_str(get_args(arg_type)[0])
      if name == "return":
        rettype = numba_arg
        continue
      secret_args.append(idx)
      signature += f"{numba_arg},"
    else:
      signature += f"{to_numba_str(arg_type)},"
  return signature, secret_args, rettype
