#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_

include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"
include "lib/Dialect/TensorExt/IR/TensorExtAttributes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class TensorExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::tensor_ext";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
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

  let arguments = (ins AnyRankedTensor:$tensor, SignlessIntegerOrIndexLike:$shift);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($tensor)) `,` type($shift)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def PermutationLike : AnyAttrOf<[
  // A permutation defined by a compact affine map with one dimension and one result expression.
  // E.g., affine_map<(d0) -> ((d0 - 4) mod 64)> compactly represents a
  // permutation defined by a single shift.
  Builtin_AffineMapAttr,
  // A permutation defined explicitly by mapping i -> a[i] for indices i of the array attr.
  // E.g., An array [3, 2, 0, 1] defines the permutation
  //
  //      0 -> 3, 1 -> 2, 2 -> 0, 3 -> 1
  //
  AnyI64ElementsAttr,
]>;

def TensorExt_PermuteOp : TensorExt_Op<"permute", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "Permute a tensor by a static permutation.";
  let description = [{
    This op represents a permutation of a tensor.

    This is lowered from a `convert_layout` op, and is implemented in terms of
    `rotate` operations.
  }];

  let arguments = (ins AnyRankedTensor:$input, PermutationLike:$permutation);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "operands attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def TensorExt_ConvertLayoutOp : TensorExt_Op<"convert_layout", [Pure, AllTypesMatch<["value", "output"]>]> {
  let summary = "Convert from one layout to another.";
  let description = [{
    This op represents the conversion of a value from one packed layout to
    another. This is implemented via a "shift network" of ciphertext rotations,
    plaintext masks (ciphertext-plaintext multiplications), and additions.

    This op is inserted by layout selection passes.
  }];

  let assemblyFormat = "operands attr-dict `:` type($output)";
  let arguments = (ins AnyType:$value, TensorExt_LayoutAttr:$from_layout, TensorExt_LayoutAttr:$to_layout);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def TensorExt_AssignLayoutOp : TensorExt_Op<"assign_layout", [Pure, AllTypesMatch<["value", "output"]>]> {
  let summary = "Assign a layout to a plaintext tensor or scalar.";
  let description = [{
    This op allows the ingestion of a plaintext value into the layout system.
    For example, ops like `linalg.reduce`, require a tensor input to represent
    initial values. These will generally be created by an `arith.constant` or
    `tensor.empty` op, which does not have secret results. Lowerings will
    convert this to a packed plaintext, so that the subsequent ops can be
    lowered as ciphertext-plaintext ops.

    This op is inserted by layout selection passes.
  }];

  let assemblyFormat = "operands attr-dict `:` type($output)";
  let arguments = (ins AnyType:$value, TensorExt_LayoutAttr:$layout);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
}


#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
