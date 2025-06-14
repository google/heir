"""Defines Python type annotations for MLIR types."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TypeVarTuple, get_args, get_origin, Optional
from numba.core.types import Type as NumbaType
from numba.core.types import boolean, int8, int16, int32, int64, float32, float64
from numba.extending import type_callable

T = TypeVar("T")
Ts = TypeVarTuple("Ts")

# List of all MLIR types we define here, for use in other parts of the compiler
MLIR_TYPES = []  # populated via MLIRType's __init_subclass__


def check_for_value(a: "MLIRType"):
  if not hasattr(a, "value"):
    raise RuntimeError(
        "Trying to use an operator on an MLIRType without a value."
    )


class MLIRType(ABC):

  def __init__(self, value: Optional[int] = None):
    # MLIRType subclasses are used in two ways:
    #
    # 1. As an explicit cast, in which case the result of the cast is a value
    #    that can be further operated on.
    # 2. As a type, in which case there is no explicit value and the class
    #    represents a standalone type.
    #
    # (2) is useful for match/case when the program is being analyzed for its
    # types. (1) is useful when allowing a program typed with heir to also run
    # as standard Python code.
    if value is not None:
      self.value = value

  def __int__(self):
    check_for_value(self)
    return int(self.value)

  def __index__(self):
    check_for_value(self)
    return int(self.value)

  def __str__(self):
    check_for_value(self)
    return str(self.value)

  def __repr__(self):
    check_for_value(self)
    return str(self.value)

  def __eq__(self, other):
    check_for_value(self)
    if isinstance(other, MLIRType):
      check_for_value(other)
      return self.value == other.value
    return self.value == other

  def __ne__(self, other):
    return not self.__eq__(other)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    MLIR_TYPES.append(cls)

  @staticmethod
  @abstractmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic MLIRType")

  @staticmethod
  @abstractmethod
  def mlir_type() -> str:
    raise NotImplementedError("No mlir type exists for a generic MLIRType")

  def __add__(self, other):
    check_for_value(self)
    return self.value + other

  def __radd__(self, other):
    check_for_value(self)
    return other + self.value

  def __sub__(self, other):
    check_for_value(self)
    return self.value - other

  def __rsub__(self, other):
    check_for_value(self)
    return other - self.value

  def __mul__(self, other):
    check_for_value(self)
    return self.value * other

  def __rmul__(self, other):
    check_for_value(self)
    return other * self.value

  def __rshift__(self, other):
    check_for_value(self)
    return self.value >> other

  def __rrshift__(self, other):
    check_for_value(self)
    return other >> self.value

  def __lshift__(self, other):
    check_for_value(self)
    return self.value << other

  def __rlshift__(self, other):
    check_for_value(self)
    return other << self.value


class Secret(Generic[T], MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Secret")

  @staticmethod
  def mlir_type() -> str:
    raise NotImplementedError("No mlir type exists for a generic Secret")


class Tensor(Generic[*Ts], MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Tensor")

  @staticmethod
  def mlir_type() -> str:
    raise NotImplementedError("No mlir type exists for a generic Tensor")


class F32(MLIRType):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float32

  @staticmethod
  def mlir_type() -> str:
    return "f32"


class F64(MLIRType):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float64

  @staticmethod
  def mlir_type() -> str:
    return "f64"


class I1(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return boolean

  @staticmethod
  def mlir_type() -> str:
    return "i1"


class I8(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int8

  @staticmethod
  def mlir_type() -> str:
    return "i8"


class I16(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int16

  @staticmethod
  def mlir_type() -> str:
    return "i16"


class I32(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int32

  @staticmethod
  def mlir_type() -> str:
    return "i32"


class I64(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int64

  @staticmethod
  def mlir_type() -> str:
    return "i64"


# Register the types defined above with Numba
for typ in [I8, I16, I32, I64, I1, F32, F64]:

  @type_callable(typ)
  def build_typer_function(context, typ=typ):
    return lambda value: typ.numba_type()


# Helper functions


def to_numba_type(type: type) -> NumbaType:
  if get_origin(type) == Secret:
    raise TypeError(
        "Secret type should not appear inside another type annotation."
    )

  if get_origin(type) == Tensor:
    args = get_args(type)
    if len(args) != 2:
      raise TypeError(
          "Tensor should contain exactly two elements: a shape list and a"
          f" type, but found {type}"
      )
    shape = args[0]
    inner_type = args[1]
    if get_origin(inner_type) == Tensor:
      raise TypeError("Nested Tensors are not yet supported.")
    # This is slightly cursed, as numba constructs array types via slice syntax
    # Cf. https://numba.pydata.org/numba-doc/dev/reference/types.html#arrays
    ty = to_numba_type(inner_type)[(slice(None),) * len(shape)]
    # We augment the type object with `shape` for the actual sizes
    ty.shape = shape  # type: ignore
    return ty

  if issubclass(type, MLIRType):
    return type.numba_type()

  raise TypeError(f"Unsupported type annotation: {type}, {get_origin(type)}")


def parse_annotations(
    annotations,
) -> tuple[list[NumbaType], list[int], NumbaType | None]:
  """Converts a python type annotation to a list of numba types.
  Args:
    annotations: A dictionary of type annotations, e.g. func.__annotations__
  Returns:
    A tuple of (args, secret_args, rettype) where:
    - args: a list of numba types for the function arguments
    - secret_args: a list of indices of secret arguments
    - rettype: the numba type of the return value
  """
  if not annotations:
    raise TypeError("Function is missing type annotations.")
  args: list[NumbaType] = []
  secret_args: list[int] = []
  rettype = None
  for idx, (name, arg_type) in enumerate(annotations.items()):
    if name == "return":
      # A user may not annotate the return type as secret
      rettype = to_numba_type(arg_type)
      continue
    if get_origin(arg_type) == Secret:
      assert len(get_args(arg_type)) == 1
      numba_arg = to_numba_type(get_args(arg_type)[0])
      if name == "return":
        rettype = numba_arg
        continue
      secret_args.append(idx)
      args.append(numba_arg)
    else:
      args.append(to_numba_type(arg_type))
  return args, secret_args, rettype
