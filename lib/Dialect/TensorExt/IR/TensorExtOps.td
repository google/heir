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
  // A permutation defined explicitly by mapping (ct, slot) -> (ct, slot)
  AnyI64ElementsAttr,
  // A layout relation, representing a mapping (ct, slot) -> (ct, slot)
  // TODO(#2047): remove the other options for PermutationLike
  TensorExt_NewLayoutAttr,
]>;

// TODO(#2047): rename PermuteOp to make it clear it supports more general mappings
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

// One-of attr for layout and new layout
def LayoutLike : AnyAttrOf<[
  TensorExt_LayoutAttr,
  TensorExt_NewLayoutAttr,
]>;

def TensorExt_ConvertLayoutOp : TensorExt_Op<"convert_layout", [Pure, AllTypesMatch<["value", "output"]>]> {
  let summary = "Convert from one layout to another.";
  let description = [{
    This op represents the conversion of a value from one packed layout to
    another. This is implemented via a "shift network" of ciphertext rotations,
    plaintext masks (ciphertext-plaintext multiplications), and additions.

    This op is inserted by layout selection passes.
  }];

  let assemblyFormat = "operands attr-dict `:` type($output)";
  let arguments = (ins AnyType:$value, LayoutLike:$from_layout, LayoutLike:$to_layout);
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
  let arguments = (ins AnyType:$value, LayoutLike:$layout);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
}

def TensorExt_UnpackOp : TensorExt_Op<"unpack", [Pure]> {
  let summary = "Unpack data from a ciphertext-semantic tensor.";
  let description = [{
    This op extracts the underlying cleartext data from a ciphertext-semantic
    tensor.
  }];

  let arguments = (ins AnyType:$value, LayoutLike:$layout);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
}

def TensorExt_RotateAndReduceOp : TensorExt_Op<"rotate_and_reduce",[Pure, AllTypesMatch<["tensor", "output"]>]> {
  let summary = "Performs a reduction of a periodically rotated tensor.";
  let description = [{
    This op reduces products of a plaintext with a periodically rotated
    tensor.

    In almost full generality, the reduction performed is

    \[
      \sum_{i \in [0, n]} p(P, T*i) \cdot rotate(v, T*i)
    \]

    where $f$ is a function, $p(P, T*i)$ is a function of a plaintext $P$ and
    $rotate(v, T*i)$ is a rotation of the ciphertext $v$ with period $T$. The
    operation takes as input the ciphertext vector $v$, the period $T$, the
    number of reductions $n$, and a tensor of plaintext values `[p(P, 0), p(P,
    T), ..., p(P, T*(n-1))]`.

    This can be used to implement a matrix vector product that uses a
    Halevi-Shoup diagonalization of the plaintext matrix. In this case, the
    reduction is

    \[
      \sum_{i \in [0, n]} P(i) \cdot rotate(v, i)
    \]

    where $P(i)$ is the $i$th diagonal of the plaintext matrix and the period
    $T$ is $1$.

    An accumulation of the ciphertext slots is also handled via this operation
    by omitting the plaintext $p(P, T*i)$ argument and using a period of 1 with
    `n = |v|` so that the reduction is simply a sum of all rotation of the
    ciphertext.

    If `reduceOp` is set to an MLIR operation name (e.g., `arith.mulf`), then
    the reduction operation is modified to use that operation instead of a sum.
    The chosen op must be one of `arith.muli`, `arith.mulf`, `arith.addi`,
    or `arith.addf`.

    Efficient lowerings of this operation can use the Baby-Step / Giant-Step
    approach from [Faster Homomorphic Linear Transformations in
    HElib](https://eprint.iacr.org/2018/244.pdf) to reduce the number of
    ciphertext rotations.
  }];

  let arguments = (
    ins AnyRankedTensor:$tensor,
    Optional<AnyRankedTensor>:$plaintexts,
    IndexAttr:$period,
    IndexAttr:$steps,
    OptionalAttr<Builtin_StringAttr>:$reduceOp
  );
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
        "Value":$tensor, "Value":$plaintexts, "int64_t":$period, "int64_t":$steps,
        "::llvm::StringRef":$reduceOp), [{
      return build(
        $_builder,
        $_state,
        tensor,
        plaintexts,
        $_builder.getIndexAttr(period),
        $_builder.getIndexAttr(steps),
        $_builder.getStringAttr(reduceOp)
      );
    }]>,
    OpBuilder<(ins
        "Value":$tensor, "Value":$plaintexts, "int64_t":$period, "int64_t":$steps,
        "std::optional<::mlir::StringAttr>":$reduceOp), [{
        ::llvm::SmallVector<::mlir::NamedAttribute, 3> attrs = {
          $_builder.getNamedAttr(
            "period", $_builder.getIndexAttr(period)),
          $_builder.getNamedAttr(
            "steps", $_builder.getIndexAttr(steps)),
        };
        if (reduceOp.has_value()) {
          attrs.push_back($_builder.getNamedAttr("reduceOp", reduceOp.value()));
        }
      return build(
        $_builder,
        $_state,
        tensor.getType(),
        ValueRange{tensor, plaintexts},
        attrs
      );
    }]>,

    // Builder for empty plaintexts
    OpBuilder<(ins
        "Value":$tensor, "int64_t":$period, "int64_t":$steps,
        "::llvm::StringRef":$reduceOp), [{
      return build(
        $_builder,
        $_state,
        tensor.getType(),
        ValueRange{tensor},
        {
          $_builder.getNamedAttr(
            "period", $_builder.getIndexAttr(period)),
          $_builder.getNamedAttr(
            "steps", $_builder.getIndexAttr(steps)),
          $_builder.getNamedAttr(
            "reduceOp", $_builder.getStringAttr(reduceOp))
        }
      );
    }]>
  ];

  // TODO(#2134): Add canonicalization patterns
}


#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
