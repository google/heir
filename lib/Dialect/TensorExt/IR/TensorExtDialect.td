#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

include "mlir/IR/DialectBase.td"

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let description = [{
    The `tensor_ext` dialect contains operations on plaintext tensors that
    correspond to the computation model of certain FHE schemes, but are
    unlikely to be upstreamed to MLIR due to their specificity to FHE.
  }];

  let cppNamespace = "::mlir::heir::tensor_ext";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
