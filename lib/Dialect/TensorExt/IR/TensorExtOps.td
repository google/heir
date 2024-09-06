#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_

include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class TensorExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::tensor_ext";
}

def TensorExt_RotateOp : TensorExt_Op<"rotate", [Pure, AllTypesMatch<["tensor", "output"]>]> {
  let summary = "Rotate a tensor some number of indices left.";
  let description = [{
    This op represents a left-rotation of a tensor by given number of indices.
    Negative shift values are interpreted as right-rotations.

    This corresponds to the `rotate` operation in arithmetic FHE schemes like
    BGV.

    This operation's current behavior allows rotating multi-dimensional tensors
    by rotating along the tensor's only non-unit dimension. This assumes the
    tensor is packed along the non-unit dimension.

    // In the future, the op will be adjusted to support rotations of general
    // multi-dimensional tensors with a vector of rotation indices for each
    // dimension. The lowering will implement the correct operations to rotate
    // the tensor along the indices given its packing.

    Examples:

    ```mlir
    %0 = ... : tensor<16xi32>
    %c7 = arith.constant 7 : i32
    %1 = tensor_ext.rotate %0, %c7 : tensor<16xi32>, i32
    ```
  }];

  let arguments = (ins AnyTensor:$tensor, SignlessIntegerLike:$shift);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($tensor)) `,` type($shift)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
