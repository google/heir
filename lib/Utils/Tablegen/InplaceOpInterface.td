#ifndef LIB_UTILS_TABLEGEN_INPLACEOPINTERFACE_TD_
#define LIB_UTILS_TABLEGEN_INPLACEOPINTERFACE_TD_

include "mlir/IR/Interfaces.td"

def InplaceOpInterface: OpInterface<"InplaceOpInterface"> {
  let cppNamespace = "::mlir::heir";

  let description = [{
    Interface for ops to to tell which operand is the same as the result.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Return the inplace operand for this op.",
      /*retTy=*/"int",
      /*methodName=*/"getInplaceOperandIndex"
    >,
  ];
}

#endif  // LIB_UTILS_TABLEGEN_INPLACEOPINTERFACE_TD_
