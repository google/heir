#ifndef LIB_TRANSFORMS_RELUCANONICALIZATIONS_REWRITES_TD_
#define LIB_TRANSFORMS_RELUCANONICALIZATIONS_REWRITES_TD_

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"

// select(a >= c, a, c) = max(a, c)
def SelectGreaterThanEqualFloat : Pat<
  (SelectOp
    (Arith_CmpFOp $pred, $a, $c, $flags), $a, $c),
  (Arith_MaximumFOp $a, $c, $flags),
  [(Constraint<
            CPred<"$0.getValue() == arith::CmpFPredicate::UGT || "
                  "$0.getValue() == arith::CmpFPredicate::UGE">> $pred)]>;

// select(a >= c, a, c) = max(a, c)
def SelectGreaterThanEqualUnsigned : Pat<
  (SelectOp
    (Arith_CmpIOp $pred, $a, $c), $a, $c),
  (Arith_MaxUIOp $a, $c),
  [(Constraint<
            CPred<"$0.getValue() == arith::CmpIPredicate::ugt || "
                  "$0.getValue() == arith::CmpIPredicate::uge">> $pred)]>;

// select(a >= c, a, c) = max(a, c)
def SelectGreaterThanEqualSigned : Pat<
  (SelectOp
    (Arith_CmpIOp $pred, $a, $c), $a, $c),
  (Arith_MaxSIOp $a, $c),
  [(Constraint<
            CPred<"$0.getValue() == arith::CmpIPredicate::sgt || "
                  "$0.getValue() == arith::CmpIPredicate::sge">> $pred)]>;

#endif // LIB_TRANSFORMS_RELUCANONICALIZATIONS_REWRITES_TD_
