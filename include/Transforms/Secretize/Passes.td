#ifndef INCLUDE_TRANSFORMS_SECRETIZE_PASSES_TD_
#define INCLUDE_TRANSFORMS_SECRETIZE_PASSES_TD_

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

def WrapGeneric : Pass<"wrap-generic", "ModuleOp"> {
  let summary = "Wraps regions using secret args in secret.generic bodies";

  let description = [{
    This pass wraps function regions of `func.func` that use secret arguments in
    `secret.generic` bodies.

    Secret arguments are annotated using a `secret.secret` argument attribute.
    This pass converts these to secret types and then inserts a `secret.generic`
    body to hold the functions region. The output type is also converted to a
    secret.

    Example input:
    ```mlir
      func.func @main(%arg0: i32 {secret.secret}) -> i32 {
        %0 = arith.constant 100 : i32
        %1 = arith.addi %0, %arg0 : i32
        return %1 : i32
      }
    ```

    Output:
    ```mlir
      func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
        %0 = secret.generic ins(%arg0 : !secret.secret<i32>) {
        ^bb0(%arg1: i32):
          %1 = arith.constant 100 : i32
          %2 = arith.addi %0, %arg1 : i32
          secret.yield %2 : i32
        } -> !secret.secret<i32>
        return %0 : !secret.secret<i32>
      }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif  // INCLUDE_TRANSFORMS_SECRETIZE_PASSES_TD_
