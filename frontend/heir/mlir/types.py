"""Defines Python type annotations for MLIR types."""

from typing import TypeVar, TypeVarTuple, Generic, get_args, get_origin, _GenericAlias

T = TypeVar('T')
Ts = TypeVarTuple("Ts")

class MLIRTypeAnnotation:
  def numba_str():
    raise NotImplementedError("No numba type exists for a generic MLIRTypeAnnotation")

class Secret(Generic[T], MLIRTypeAnnotation):
  def numba_str():
    raise NotImplementedError("No numba type exists for a generic Secret")

class Tensor(Generic[*Ts], MLIRTypeAnnotation):
  def numba_str():
    raise NotImplementedError("No numba type exists for a generic Tensor")

class F32(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable
  def numba_str():
    return "float32"

class F64(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable
  def numba_str():
    return "float64"

class I1(MLIRTypeAnnotation):
  def numba_str():
    return "bool"

class I4(MLIRTypeAnnotation):
  def numba_str():
    return "int4"

class I8(MLIRTypeAnnotation):
  def numba_str():
    return "int8"

class I16(MLIRTypeAnnotation):
  def numba_str():
    return "int16"

class I32(MLIRTypeAnnotation):
  def numba_str():
    return "int32"
class I64(MLIRTypeAnnotation):
 def numba_str():
   return "int64"



# Helper functions

def to_numba_str(type) -> str:
  if(get_origin(type) == Secret):
    raise TypeError("Secret type should not appear inside another type annotation.")

  if(get_origin(type) == Tensor):
    args = get_args(type)
    inner_type = args[len(args) - 1]
    if(get_origin(inner_type) == Tensor):
      raise TypeError("Nested Tensors are not yet supported.")
    return f"{to_numba_str(inner_type)}[{','.join([':'] * (len(args) - 1))}]"

  if(issubclass(type, MLIRTypeAnnotation)):
    return type.numba_str()

  raise TypeError(f"Unsupported type annotation: {type}")


def parse_annotations(annotations):
  if (not annotations):
    raise TypeError("Function is missing type annotations.")
  signature = ""
  secret_args = []
  for idx, (_, arg_type) in enumerate(annotations.items()):
      if get_origin(arg_type) == Secret:
          secret_args.append(idx)
          assert(len(get_args(arg_type)) == 1)
          signature+= f"{to_numba_str(get_args(arg_type)[0])},"
      else:
        signature += f"{to_numba_str(arg_type)},"
  return signature, secret_args
