#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Pass/PassBase.td"

def ArithToModArith : Pass<"arith-to-mod-arith", "ModuleOp"> {
  let summary = "Lower standard `arith` to `mod-arith`.";

  let description = [{
    This pass lowers the `arith` dialect to their `mod-arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
