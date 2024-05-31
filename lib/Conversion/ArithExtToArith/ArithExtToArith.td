#ifndef LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_
#define LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_

include "lib/DRR/Utils.td"
include "lib/Dialect/ArithExt/IR/ArithExtOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ArithExtToArith : Pass<"arith-ext-to-arith", "ModuleOp"> {
  let summary = "Lower `arith_ext` to standard `arith`.";

  let description = [{
    This pass lowers the `arith_ext` dialect to their `arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::arith_ext::ArithExtDialect",
  ];
}

// Using DRR to generate the lowering patterns for specific operations

defvar DefGE = ConstantEnumCase<Arith_CmpIPredicateAttr, "uge">;

def ConvertSubIfGE : Pattern<
  (ArithExt_SubIfGEOp $x, $y),
  [
    (Arith_SubIOp:$subOp $x, $y, DefOverflow),
    (Arith_CmpIOp:$cmpOp DefGE, $x, $y),
    (SelectOp $cmpOp, $subOp, $x)
  ]
>;

#endif  // LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_
