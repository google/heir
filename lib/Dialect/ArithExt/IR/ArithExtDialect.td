#ifndef LIB_DIALECT_ARITHEXT_IR_ARITHEXTDIALECT_TD_
#define LIB_DIALECT_ARITHEXT_IR_ARITHEXTDIALECT_TD_

include "mlir/IR/DialectBase.td"

def ArithExt_Dialect : Dialect {
  let name = "arith_ext";
  let description = [{
    The `arith_ext` dialect contains operations used with polynomial arithmetic,
    but are unlikely to be upstreamed to MLIR due to their specificity to FHE.
  }];

  let cppNamespace = "::mlir::heir::arith_ext";
  let dependentDialects = [
    "arith::ArithDialect",
  ];
}

#endif  // LIB_DIALECT_ARITHEXT_IR_ARITHEXTDIALECT_TD_
