#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLDIALECT_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TfheRustBool_Dialect : Dialect {
  let name = "tfhe_rust_bool";

  let description = [{
    The `tfhe_rust_bool` dialect is an exit dialect for generating rust code against the tfhe-rs library API,
    using the boolean parameter set.

    See https://github.com/zama-ai/tfhe-rs
  }];

  let cppNamespace = "::mlir::heir::tfhe_rust_bool";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLDIALECT_TD_
