"""The compilation pipeline."""

from heir_py.mlir_emitter import TextualMlirEmitter
from numba.core.bytecode import ByteCode, FunctionIdentity
from numba.core.interpreter import Interpreter


def run_compiler(function):
  func_id = FunctionIdentity.from_function(function)
  bytecode = ByteCode(func_id)
  ssa_ir = Interpreter(func_id).interpret(bytecode)
  mlir_textual = TextualMlirEmitter(ssa_ir).emit()

  # FIXME: compile the MLIR and generated bindings,
  # import the module, and return the callbacks
  return mlir_textual
