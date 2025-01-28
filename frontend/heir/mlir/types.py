from typing import TypeVar, TypeVarTuple, Generic
from numba import types
from numba.extending import typeof_impl, as_numba_type, type_callable

# (Mostly Dummy) Classes for Type Annotations

T = TypeVar('T')
Ts = TypeVarTuple("Ts")

class MLIRTypeAnnotation:
  def numba_str():
    raise NotImplementedError("No numba type exists for MLIRTypeAnnotation")


class Secret(Generic[T], MLIRTypeAnnotation):
  def numba_str():
    raise NotImplementedError("No numba type exists for Secret")

class Tensor(Generic[*Ts], MLIRTypeAnnotation):
  def numba_str():
    raise NotImplementedError("No numba type exists for Tensor")

class F32(MLIRTypeAnnotation):
  def numba_str():
    return "float32"

class F64(MLIRTypeAnnotation):
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


# # Numba Types
# class SecretType(types.Type):
#   def __init__(self):
#     super(SecretType, self).__init__(name="Secret")

# secret_type = SecretType()

# @typeof_impl.register(Secret)
# def typeof_index(val, c):
#   return secret_type

# as_numba_type.register(Secret, SecretType)
