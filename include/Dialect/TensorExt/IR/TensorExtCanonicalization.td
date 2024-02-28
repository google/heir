#ifndef INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_
#define INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_

include "TensorExtOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"

// TODO(#515): refactor these helpers to a common file with InsertRotate.td
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

def MakeSingleResultVariadic: NativeCodeCall<"{ $0 }">;

def CreateIndexCastOp : NativeCodeCall<
      "$_builder.create<arith::IndexCastOp>($0.getLoc(), $1.getType(), $0)">;

def IsZero :
    Constraint<
      CPred<"llvm::cast<mlir::IntegerAttr>($0).getValue().isZero()">>;

def OutOfBoundsOfTensorDim :
  Constraint<
    CPred<
      "llvm::cast<mlir::IntegerAttr>($0).getValue().getSExtValue() < 0 "
      "|| llvm::cast<mlir::IntegerAttr>($0).getValue().getSExtValue() > "
      "llvm::cast<mlir::ShapedType>($1.getType()).getShape()[0]"
    >
  >;

// rotate %t, 0 -> %t
def DropZeroRotation : Pat<
  (TensorExt_RotateOp $tensor, (ConstantLikeMatcher APIntAttr:$c0)),
  (replaceWithValue $tensor),
  [(IsZero $c0)]
>;

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
  [(OutOfBoundsOfTensorDim $shiftAmount, $tensor)]
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

// Rotating two tensors by the same amount can be converted to a single
// post-rotation. This can result in eliminating either the rotation (because
// it can be combined with a later rotation) or the arith op itself, if it is
// is identical to an existing arith op applied before the rotation.
foreach ArithOp = [Arith_AddIOp, Arith_SubIOp, Arith_MulIOp] in {
  def FactorParallelRotationsThroughOp_#ArithOp : Pat<
    (ArithOp
      (TensorExt_RotateOp $t1, $i),
      (TensorExt_RotateOp $t2, $i),
      $ovf),
    (TensorExt_RotateOp (ArithOp $t1, $t2, $ovf), $i)
  >;
}

#endif  // INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTCANONICALIZATION_TD_
