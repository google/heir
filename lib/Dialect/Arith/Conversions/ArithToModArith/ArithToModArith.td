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

    This pass will transform `arith` operations to the mod-arith dialect, where
    the find-mac pass can be used to convert consecutive multiply addition
    operations into a single operation. In a later pass, these large precision
    MAC operations (typically 64 or 32-bit) will be lowered into small
    precision (8 or 4b) operations that can be mapped to CGGI operations.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];

  let options = [
    Option<"modulus", "modulus", "int64_t",
           /*default=*/"0", "Modulus to use for the mod-arith dialect."
           " If not specified, the pass will use the natural modulus for that integer type">,
  ];
}

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
