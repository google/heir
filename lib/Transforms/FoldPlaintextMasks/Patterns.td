
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"


def IsIntegerOneHotVector : Constraint<CPred<"isOneHotVector($0)">>;
def IntersectMasks : NativeCodeCall<"intersectMasks($0, $1)">;

// IntersectMasks simplifies repeated applications of a plaintext mask
// By combining the masks.
foreach ArithOp = [Arith_MulIOp, Arith_MulFOp] in {
  def InsertsectMasks_#ArithOp#_Lhs1 : Pat<
    (ArithOp $mask1, (ArithOp $mask2, $value, $flags1), $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;

  def InsertsectMasks_#ArithOp#_Lhs2 : Pat<
    (ArithOp $mask1, (ArithOp $value, $mask2, $flags1), $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;

  def InsertsectMasks_#ArithOp#_Lhs3 : Pat<
    (ArithOp $value, (ArithOp $mask1, $mask2, $flags1), $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;

  def InsertsectMasks_#ArithOp#_Rhs1 : Pat<
    (ArithOp (ArithOp $mask2, $value, $flags1), $mask1, $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;

  def InsertsectMasks_#ArithOp#_Rhs2 : Pat<
    (ArithOp (ArithOp $value, $mask2, $flags1), $mask1, $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;

  def InsertsectMasks_#ArithOp#_Rhs3 : Pat<
    (ArithOp (ArithOp $mask1, $mask2, $flags1), $value, $flags2),
    (ArithOp $value, (Arith_ConstantOp (IntersectMasks $mask1, $mask2)), $flags1),
    [
      (IsIntegerOneHotVector $mask1),
      (IsIntegerOneHotVector $mask2),
    ]
  >;
}
