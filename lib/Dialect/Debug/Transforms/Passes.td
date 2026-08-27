#ifndef LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DebugInsertLinalgCheckpoints : Pass<"debug-insert-linalg-checkpoints", "mlir::ModuleOp"> {
  let summary = "Insert debug checkpoints at secret-dependent Linalg operations";
  let description = [{
    Inserts a `debug.validate` operation for each secret function input and for
    each secret-dependent result of a Linalg operation. This instruments the
    tensor-level model before HEIR expands one model operation into many
    cryptographic operations, producing useful semantic checkpoints without
    the noise of a callback after every FHE operation.

    Cleartext-only Linalg operations, such as weight transposes and output
    initializers, are omitted. Existing validations make the pass idempotent.
  }];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"\"\"", "Name of the function to instrument; empty instruments all functions">,
    Option<"includeInputs", "include-inputs", "bool",
           /*default=*/"true", "Whether to checkpoint secret function inputs">,
  ];
  let dependentDialects = ["mlir::heir::debug::DebugDialect"];
}

def DebugValidateNames : Pass<"debug-validate-names", "mlir::ModuleOp"> {
  let summary = "Validates that debug.validate names are unique";
  let description = [{
    This pass walks the IR and ensures that each `debug.validate` operation
    has a unique `name` attribute. If any duplicates are found, the pass fails.
  }];
  let dependentDialects = ["mlir::heir::debug::DebugDialect"];
}

#endif  // LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_
