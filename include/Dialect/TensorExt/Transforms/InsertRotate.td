#ifndef INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTPATTERNS_TD_
#define INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTPATTERNS_TD_

include "include/DRR/Utils.td"
include "include/Dialect/TensorExt/IR/TensorExtOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"

// Get the target_slot attribute from an op, if it exists, or else
// return a zero index attribute.
def GetTargetSlotAttr : NativeCodeCall<
      "$0.getOwner()->hasAttr(\"target_slot\")"
      " ? llvm::cast<mlir::IntegerAttr>($0.getOwner()->getAttr(\"target_slot\"))"
      " : $_builder.getIndexAttr(0)">;

// The patterns in this file are intended to align with the automatic-SIMD
// batching heuristics from the HECO project. See section 4.4 of
// https://arxiv.org/abs/2202.01649 and the hir2hir passes in
// https://github.com/MarbleHE/HECO/blob/main/src/Passes/hir2hir/

// Match an arith op that extracts scalar values from two tensors, and replace
// it with rotations to align slots and apply the same op in SIMD. Other
// patterns in this file will find better alignment of adjacent rotations, and
// canonicalization patterns will remove duplicated rotations.
foreach ArithOp = [Arith_AddIOp, Arith_SubIOp, Arith_MulIOp] in {
  def InsertRotations_#ArithOp : Pattern<
    (ArithOp:$arithOp
      (Tensor_ExtractOp $t1, (variadic $i1)),
      (Tensor_ExtractOp $t2, (variadic $i2)),
      $overflow),
    [
      (TensorExt_RotateOp:$r1 $t1,
          (Arith_SubIOp $i1, (Arith_ConstantOp (GetTargetSlotAttr $arithOp)), DefOverflow)),
      (TensorExt_RotateOp:$r2 $t2,
          (Arith_SubIOp $i2, (Arith_ConstantOp (GetTargetSlotAttr $arithOp)), DefOverflow)),
      (ArithOp:$opResult $r1, $r2, $overflow),
      (Tensor_ExtractOp
        $opResult,
        (MakeSingleResultVariadic
          (Arith_ConstantOp (GetTargetSlotAttr $arithOp)))),
    ]
  >;
}


// Pre-align the first op's operands to the index that the result is
// used for in a subsequent op. This is used to simplify the IR
// primarily when there is no specific slot target selected for an op. In
// that case, the above pattern will still replace extractions with
// rotations, and the simplifications will occur by replacing triples
// of rotations with pairs.
// TODO(#514): handle OuterOp with two different InnerOps on the LHS and RHS
foreach InnerOp = [Arith_AddIOp, Arith_SubIOp, Arith_MulIOp] in {
  foreach OuterOp = [Arith_AddIOp, Arith_SubIOp, Arith_MulIOp] in {
    // Left associated grouping handles (add (add (rotate t1 i1) (rotate t2 i2)) (rotate t3 i3))
    def AlignRotations_LeftAssociated_Inner_#InnerOp#_Outer_#OuterOp : Pattern<
      (OuterOp
        (InnerOp (TensorExt_RotateOp $t1, $i1), (TensorExt_RotateOp $t2, $i2), $ovf1),
        (TensorExt_RotateOp $t3, $i3),
        $ovf2),
      [
        (TensorExt_RotateOp:$r1 $t1, (Arith_SubIOp $i1, $i3, DefOverflow)),
        (TensorExt_RotateOp:$r2 $t2, (Arith_SubIOp $i2, $i3, DefOverflow)),
        (InnerOp:$addResult $r1, $r2, $ovf1),
        (OuterOp:$output $addResult, $t3, $ovf2),
        // Downstream ops are not updated by this pass, so we need to preserve the original
        // rotation and then clean it up in a separate canonicalization pattern.
        (TensorExt_RotateOp $output, $i3),
      ]
    >;

    // Right associated grouping handles (add (rotate t1 i1) (add (rotate t2 i2) (rotate t3 i3)))
    def AlignRotations_RightAssociated_Inner_#InnerOp#_Outer_#OuterOp : Pattern<
      (OuterOp
        (TensorExt_RotateOp $t3, $i3),
        (InnerOp (TensorExt_RotateOp $t1, $i1), (TensorExt_RotateOp $t2, $i2), $ovf1),
        $ovf2),
      [
        (TensorExt_RotateOp:$r1 $t1, (Arith_SubIOp $i1, $i3, DefOverflow)),
        (TensorExt_RotateOp:$r2 $t2, (Arith_SubIOp $i2, $i3, DefOverflow)),
        (InnerOp:$addResult $r1, $r2, $ovf1),
        (OuterOp:$output $addResult, $t3, $ovf2),
        // Downstream ops are not updated by this pass, so we need to preserve the original
        // rotation and then clean it up in a separate canonicalization pattern.
        (TensorExt_RotateOp $output, $i3),
      ]
    >;
  }
}

#endif  // INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTPATTERNS_TD_
