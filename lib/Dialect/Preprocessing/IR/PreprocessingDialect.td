#ifndef LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGDIALECT_TD_
#define LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Preprocessing_Dialect : Dialect {
  let name = "preprocessing";
  let description = [{
    The `preprocessing` dialect defines operations and types for
    representing sever-side plaintext preprocessing. The dialect
    is meant as a structural intermediate dialect to support the
    `split-preprocessing` pass.
  }];

  let cppNamespace = "::mlir::heir::preprocessing";
  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGDIALECT_TD_
