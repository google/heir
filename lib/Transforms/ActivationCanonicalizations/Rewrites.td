#ifndef LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_REWRITES_TD_
#define LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_REWRITES_TD_

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/IR/PatternBase.td"
include "lib/Dialect/MathExt/IR/MathExtOps.td"

def IsOne : Constraint<CPred<"IsOne($0)">>;

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

// AddFOp is commutative and is canonicalized to have a constant operand be second.

// 1 / (1 + exp(-x)) = math_ext.sigmoid(x).
def Sigmoid : Pat<
  (Arith_DivFOp
    (Arith_ConstantOp $c1_1),
    (Arith_AddFOp
      (Math_ExpOp (Arith_NegFOp $x, $_), $_),
      (Arith_ConstantOp $c1_2), $_), $_),
  (MathExt_SigmoidOp $x),
  [(IsOne $c1_1), (IsOne $c1_2)]>;

#endif // LIB_TRANSFORMS_ACTIVATIONCANONICALIZATIONS_REWRITES_TD_
