#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def JaxiteCkksParameterSelection : Pass<"jaxite-ckks-parameter-selection", "mlir::ModuleOp"> {
  let summary = "Selects parameters for Jaxite CKKS backend";
  let description = [{
    This pass selects parameters for the Jaxite CKKS backend and annotates them on the module.

    (* example filepath=tests/Dialect/JaxiteWord/Transforms/doctest.mlir *)
  }];
  let dependentDialects = ["mlir::heir::jaxiteword::JaxiteWordDialect"];
}

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
