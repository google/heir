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
    Adds `!preprocessing.resource_dir` arguments to `preprocessing.load_resource` ops,
    threading it through the enclosing func with a trailing func arg.

    This allows one to use `preprocessing.load_resource` without forcing
    relative paths from the executing binary to the loaded resource.

    (* example filepath=tests/Dialect/Preprocessing/Transforms/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::preprocessing::PreprocessingDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_TD_
