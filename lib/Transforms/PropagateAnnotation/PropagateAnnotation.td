#ifndef LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
#define LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_

include "mlir/Pass/PassBase.td"

def PropagateAnnotation : Pass<"propagate-annotation"> {
  let summary = "Propagate annotation from operation to subsequent operations";
  let description = [{
    This pass propagates the attribute from one operation to subsequent operations
    if these operations does not have the attribute already.

    Example: with `--propagate-annotation=attr-name=test.attr`

    ```mlir
    func.func @foo(%arg0: i16 {test.attr = 1}) -> i16 {
      %0 = arith.muli %arg0, %arg0 : i16
      %1 = mgmt.relinearize %0 : i16
      return %1 : i16
    }
    ```

    the above IR becomes

    ```mlir
    func.func @foo(%arg0: i16 {test.attr = 1 : i64}) -> i16 {
      %0 = arith.muli %arg0, %arg0 {test.attr = 1 : i64} : i16
      %1 = mgmt.relinearize %0 {test.attr = 1 : i64} : i16
      return {test.attr = 1 : i64} %1 : i16
    }
    ```
  }];

  let options = [
    Option<"attrName", "attr-name", "std::string", /*default=*/"\"\"",
           "The attribute name to propagate with.">,
    Option<"reverse", "reverse", "bool", /*default=*/"false",
           "Whether to propagate in reverse">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
