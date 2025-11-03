#ifndef LIB_TRANSFORMS_APPLYFOLDERS_PATTERNS_TD_
#define LIB_TRANSFORMS_APPLYFOLDERS_PATTERNS_TD_

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"
include "lib/Utils/DRR/Utils.td"

def AnyZero : AttrConstraint<
    CPred<"::mlir::matchPattern($_self, m_AnyZeroFloat())">,
    "is int or float zero">;

// a + 0.0 = a
def AddFloatingPointZero : Pat<
  (Arith_AddFOp $a,
    (ConstantLikeMatcher AnyZero:$value), $anyAttr),
  (replaceWithValue $a)>;

// add(empty, a) = a
def AddEmptyTensor : Pat<
  (Arith_AddFOp $a,
    (Tensor_EmptyOp $b), $anyAttr),
  (replaceWithValue $a)>;

#endif // LIB_TRANSFORMS_APPLYFOLDERS_PATTERNS_TD_
