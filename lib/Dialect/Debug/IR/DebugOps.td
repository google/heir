#ifndef LIB_DIALECT_DEBUG_IR_DEBUGOPS_TD_
#define LIB_DIALECT_DEBUG_IR_DEBUGOPS_TD_

include "DebugDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Debug_Op<string mnemonic, list<Trait> traits = []> :
    Op<Debug_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::debug";
}

def Debug_ValidateOp : Debug_Op<"validate", []> {
  let summary = "Validates an SSA value.";
  let description = [{
    The `debug.validate` operation is a high-level placeholder for validating
    an SSA value. This is transformed via `*-add-debug-port` passes to a function
    call to an externally defined function that may then decrypt and validate
    the operand.

    The mandatory `name` attribute gives a unique identifier for the validation
    instance, and this is used to connect intermediate values of a plaintext
    execution of a program to the corresponding program points of the
    HEIR-compiled program.

    An optional `metadata` attribute may contain an arbitrary JSON blob, to
    be passed to the function call, which is intended to contain metadata like
    the plaintext execution result that the called function can use to compare
    with the decrypted ciphertext.
  }];

  let arguments = (ins
    AnyType:$input,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$metadata
  );

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif  // LIB_DIALECT_DEBUG_IR_DEBUGOPS_TD_
