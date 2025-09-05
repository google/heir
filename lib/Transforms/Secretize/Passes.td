#ifndef LIB_TRANSFORMS_SECRETIZE_PASSES_TD_
#define LIB_TRANSFORMS_SECRETIZE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def Secretize : Pass<"secretize", "ModuleOp"> {
  let summary = "Adds secret argument attributes to entry function";

  let description = [{
    Helper pass that adds a secret.secret attribute argument to each function argument.
    By default, the pass applies to all functions in the module.
    This may be overridden with the option -function=func_name to apply to a single function only.
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];

  let options = [
    Option<"function", "function", "std::string", "\"\"", "function to add secret annotations to">
  ];
}

def WrapGeneric : Pass<"wrap-generic", "ModuleOp"> {
  let summary = "Wraps regions using secret args in secret.generic bodies";

  let description = [{
    This pass converts functions (`func.func`) with `{secret.secret}` annotated
    arguments to use `!secret.secret<...>` types and wraps the function body in
    a `secret.generic` region. The output type is also converted to
    `!secret.secret<...>`.

    (* example filepath=tests/Transforms/wrap_generic/doctest.mlir *)
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SECRETIZE_PASSES_TD_
