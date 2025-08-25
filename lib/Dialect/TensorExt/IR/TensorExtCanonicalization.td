#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_

include "TensorExtOps.td"
include "lib/Utils/DRR/Utils.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"

def OutOfBoundsOfTensorDim :
  Constraint<
    CPred<
      "llvm::cast<mlir::IntegerAttr>($0).getValue().getSExtValue() < 0 "
      "|| llvm::cast<mlir::IntegerAttr>($0).getValue().getSExtValue() > "
      "llvm::cast<mlir::ShapedType>($1.getType()).getShape()[0]"
    >
  >;

def IsOneDimTensor : Constraint<
  CPred<
    "llvm::cast<mlir::ShapedType>($0.getType()).getRank() == 1"
  >
>;

def DenseElementsAttrIsSplatPred
    : CPred<"llvm::cast<DenseElementsAttr>($_self).isSplat()">;

// A constant tensor that has elements with the same value.
def SplatElementsAttr : ElementsAttrBase<DenseElementsAttrIsSplatPred,
  "splat tensor attr">;

// rotate dense<x>, %t -> dense<x>
def RotateDenseConstant : Pat<
  (TensorExt_RotateOp
    (Arith_ConstantOp:$constantValue SplatElementsAttr:$constantAttr), $shiftAmount),
  (replaceWithValue $constantValue)
>;

// rotate %t, 0 -> %t
def DropZeroRotation : Pat<
  (TensorExt_RotateOp $denseConstant, (ConstantLikeMatcher APIntAttr:$c0)),
  (replaceWithValue $denseConstant),
  [(IsZeroIntAttr $c0)]
>;

// Currently commented out because it doesn't work for multi-dimensional tensors.
// Will be uncommented and fixed by Asra's PR. Commenting this out causes various
// other tests to fail.
// rotate %t, x -> rotate %t, x mod size
def NormalizeRotationIndex : Pat<
  (TensorExt_RotateOp $tensor, (Arith_ConstantOp:$shiftOp APIntAttr:$shiftAmount)),
  (TensorExt_RotateOp $tensor,
      (Arith_RemUIOp
        $shiftOp,
        // Only works for 1D tensors: index is taken modulo the tensor length,
        // i.e., dim 0
        (CreateIndexCastOp
          (Tensor_DimOp $tensor, (Arith_ConstantOp ConstantAttr<IndexAttr, "0">)),
          $shiftOp))
      ),
  [(OutOfBoundsOfTensorDim $shiftAmount, $tensor),
   (IsOneDimTensor $tensor)]
>;

// %0 = rotate %t, x
// %1 = rotate %0, y
// ---> rotate %t (x+y)
def CombineSequentialRotates : Pat<
  (TensorExt_RotateOp
    (TensorExt_RotateOp $tensor, (Arith_ConstantOp:$xOp APIntAttr:$x)),
    (Arith_ConstantOp:$yOp APIntAttr:$y)),
  (TensorExt_RotateOp $tensor, (Arith_AddIOp $xOp, $yOp, DefOverflow)),
  []
>;

// A rotation followed by extraction can be extracted directly from the
// original tensor.
def RotatePlusExtractToIndexedExtract : Pat<
  (Tensor_ExtractOp
    (TensorExt_RotateOp $tensor, $shift),
    (variadic $index)),
  (Tensor_ExtractOp
    $tensor,
    (MakeSingleResultVariadic (Arith_AddIOp $shift, $index, DefOverflow)))
>;

foreach ArithOp = [Arith_AddIOp, Arith_SubIOp, Arith_MulIOp, Arith_AddFOp, Arith_SubFOp, Arith_MulFOp] in {
  // Rotating two tensors by the same amount can be converted to a single
  // post-rotation. This can result in eliminating either the rotation (because
  // it can be combined with a later rotation) or the arith op itself, if it is
  // is identical to an existing arith op applied before the rotation.
  def FactorParallelRotationsThroughOp_#ArithOp : Pat<
    (ArithOp
      (TensorExt_RotateOp $t1, $i),
      (TensorExt_RotateOp $t2, $i),
      $ovf),
    (TensorExt_RotateOp (ArithOp $t1, $t2, $ovf), $i),
    [], [], (addBenefit 3)
  >;

  // %0 = rotate %t1, x
  // %1 = rotate %t2, y
  // %2 = add %0, %1
  // %3 = rotate %2, z
  // --->
  // %0 = rotate %t1, x + z
  // %1 = rotate %t2, y + z
  // %2 = add %0, %1
  def MergeRotationsSandwichingOp_#ArithOp : Pat<
    (TensorExt_RotateOp
      (ArithOp
        (TensorExt_RotateOp $t1, $s1),
        (TensorExt_RotateOp $t2, $s2),
        $ovf),
      $s3),
    (ArithOp
      (TensorExt_RotateOp $t1, (Arith_AddIOp $s3, $s1, DefOverflow)),
      (TensorExt_RotateOp $t2, (Arith_AddIOp $s3, $s2, DefOverflow)),
      $ovf),
    [], [], (addBenefit 2)
  >;

  // These two patterns don't eliminate any rotations, but they move rotations
  // earlier in the IR, which can help to identify common subexpressions to
  // eliminate and trigger other patterns.
  // %0 = rotate %t1, x
  // %2 = add %0, %t2
  // %1 = rotate %2, y
  // --->
  // %0 = rotate %t1, x + y
  // %1 = rotate %t2, y
  // %2 = add %0, %1
  def MergeEquivalentRotationsEarlierLHS_#ArithOp : Pat<
    (TensorExt_RotateOp
      (ArithOp:$arithOp
        (TensorExt_RotateOp $t1, $s1),
        $t2,
        $ovf),
      $s2),
    (ArithOp
      (TensorExt_RotateOp $t1, (Arith_AddIOp $s2, $s1, DefOverflow)),
      (TensorExt_RotateOp $t2, $s2),
      $ovf),
    // HasOneUse is important, because otherwise we would just be adding an
    // additional rotation.
    //
    // E.g.,
    //
    //  %1 = arith.muli %arg2, %arg3
    //  %2 = tensor_ext.rotate %1, %c4
    //  %3 = arith.addi %1, %2
    //  %4 = tensor_ext.rotate %3, %c2
    //  %5 = arith.addi %3, %4
    //  %6 = tensor_ext.rotate %5, %c1
    //  %7 = arith.addi %5, %6
    //
    // %4 would match and insert
    //
    //  %r1 = tensor_ext.rotate %1, %c6
    //  %r2 = tensor_ext.rotate %1, %c2
    //  %r3 = arith.addi %r1, %r2
    //
    // But the original addi cannot be removed, and so the rotation it depends
    // on also cannot be removed.
    [(HasOneUse:$arithOp)]
  >;
  def MergeEquivalentRotationsEarlierRHS_#ArithOp : Pat<
    (TensorExt_RotateOp
      (ArithOp:$arithOp
        $t2,
        (TensorExt_RotateOp $t1, $s1),
        $ovf),
      $s2),
    (ArithOp
      (TensorExt_RotateOp $t2, $s2),
      (TensorExt_RotateOp $t1, (Arith_AddIOp $s2, $s1, DefOverflow)),
      $ovf),
    [(HasOneUse:$arithOp)]
  >;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_
