#ifndef INCLUDE_TRANSFORMS_SECRETIZE_SECRETIZE_TD_
#define INCLUDE_TRANSFORMS_SECRETIZE_SECRETIZE_TD_

include "mlir/Pass/PassBase.td"

def Secretize : Pass<"secretize", "ModuleOp"> {
  let summary = "Adds secret argument attributes to entry function";

  let description = [{
    Adds a secret.secret attribute argument to each argument in the entry
    function of an MLIR module. By default, the function is `main`. This may be
    overridden with the option -entry-function=top_level_func.
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];

  let options = [
    Option<"entryFunction", "entry-function", "std::string", "\"main\"", "entry function of the module">
  ];
}

#endif  // INCLUDE_TRANSFORMS_SECRETIZE_SECRETIZE_TD_
