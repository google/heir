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

  // TODO(#2317): figure out how to get OpInterface verifier to run
  // let verify = [{
  //   ::mlir::heir::KernelName kernelName = ::mlir::heir::KernelName::Trivial;
  //   auto attrName = ::mlir::heir::secret::SecretDialect::kKernelAttrName;
  //   if ($_op->hasAttr(attrName)) {
  //     auto attr = $_op->getAttrOfType<::mlir::heir::secret::KernelAttr>(attrName);
  //     kernelName = attr.getName();
  //   }
  //   if (!::mlir::heir::isSupportedKernel($_op, kernelName)) {
  //     return $_op->emitOpError()
  //            << "has unsupported kernel ''";
  //   }
  //   return success();
  // }];
}

#endif  // LIB_INTERFACE_HOISTINGINTERFACES_TD_
