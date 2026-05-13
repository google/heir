#ifndef LIB_INTERFACE_HOISTINGINTERFACES_TD_
#define LIB_INTERFACE_HOISTINGINTERFACES_TD_

include "mlir/IR/OpBase.td"

def LayoutConversionHoistableOpInterface : OpInterface<"LayoutConversionHoistableOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that abstracts the common information required for the
    `layout-optimization` pass to evaluate the feasibility and cost of hoisting
    a `convert_layout` op through a given operation.
  }];

  let methods = [
    InterfaceMethod<
      "Get hoisters for the op.",
      "std::vector<::mlir::heir::Hoister>",
      "getHoisters",
      (ins "::mlir::heir::tensor_ext::ConvertLayoutOp":$convertLayoutOp)
    >,
  ];


}

#endif  // LIB_INTERFACE_HOISTINGINTERFACES_TD_
