#ifndef LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DebugValidateNames : Pass<"debug-validate-names", "mlir::ModuleOp"> {
  let summary = "Validates that debug.validate names are unique";
  let description = [{
    This pass walks the IR and ensures that each `debug.validate` operation
    has a unique `name` attribute. If any duplicates are found, the pass fails.
  }];
  let dependentDialects = ["mlir::heir::debug::DebugDialect"];
}

#endif  // LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_TD_
