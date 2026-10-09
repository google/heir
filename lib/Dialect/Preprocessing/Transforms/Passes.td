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

def UseDynamicResourceDir : Pass<"preprocessing-use-dynamic-resource-dir", "::mlir::ModuleOp"> {
  let summary = "Pass the resource directory into functions that load resources";
  let description = [{
    Every function that executes `preprocessing.load_resource`, directly or
    through a call, gains a trailing `!preprocessing.resource_dir` argument.
    The loads take their directory from it and call sites forward the
    caller's. A public function's caller then names the directory once at
    runtime instead of the generated code baking it in.
  }];
  let dependentDialects = [
    "mlir::heir::preprocessing::PreprocessingDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_TD_
