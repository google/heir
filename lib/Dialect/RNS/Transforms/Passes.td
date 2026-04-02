#ifndef LIB_DIALECT_RNS_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_RNS_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def LowerConvertBasis : Pass<"rns-lower-convert-basis"> {
  let summary = "Lower rns.convert_basis to explicit RNS and ModArith ops";
  let description = [{
    This pass replaces `rns.convert_basis` with an implementation based on
    mixed-radix representation over the target RNS basis.
    (* example filepath=tests/Dialect/RNS/Transforms/lower_convert_basis.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::heir::rns::RNSDialect",
  ];
}

#endif  // LIB_DIALECT_RNS_TRANSFORMS_PASSES_TD_
