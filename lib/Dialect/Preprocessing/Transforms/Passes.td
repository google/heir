#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ValidatePreprocessing : Pass<"validate-preprocessing", "::mlir::ModuleOp"> {
  let summary = "Validate well-formedness of preprocessing dialect ops";
  let description = [{
    Verifies exactly one StoreOp and exactly one LoadOp per site_id across the module,
    asserting index arity match and element type correspondence.

    This helper is meant to validate that the output of `split-preprocessing`
    satisfies appropriate invariants.
  }];
  let dependentDialects = ["mlir::heir::preprocessing::PreprocessingDialect"];
}

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_TD_
