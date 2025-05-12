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
    This pass converts functions (`func.func`) with `{secret.secret}` annotated arguments
    to use `!secret.secret<...>` types and wraps the function body in a `secret.generic` region.
    The output type is also converted to `!secret.secret<...>`.

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
        %0 = secret.generic(%arg0 : !secret.secret<i32>) {
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

#endif  // LIB_TRANSFORMS_SECRETIZE_PASSES_TD_
