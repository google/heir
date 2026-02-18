#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/TensorExt/IR/TensorExtAttributes.td"
include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"
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
    BGV. This op currently only supports 1D rotations of the last axis of a
    tensor. A `tensor<4x64xi32>` is interpreted as 4 ciphertexts each with 64
    slots, and a rotation on a value of this type rotates each ciphertext by
    the given amount.

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

def MappingLike : AnyAttrOf<[
  // A list of tuples [a, b, c, d] representing an explicit map (ct, slot) ->
  // (ct, slot) defined by f(a, b) = (c, d). This mapping is primarily used
  // for testing implement-shift-network, as not all mappings can be achieved
  // as a consequence of higher-level layout conversions.
  AnyI64ElementsAttr,
  // A layout relation with domain and range space equal to (ct, slot).
  TensorExt_LayoutAttr,
]>;

def TensorExt_RemapOp : TensorExt_Op<"remap", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "Reorganize the entries of a tensor by a static mapping.";
  let description = [{
    This op represents a remapping of entries of a tensor.

    This op is primarily inserted as a lowered form of `convert_layout` op, to
    represent a ciphertext slot-repacking operation. However, it can more
    generally express any partial mapping from source tensor indices to target
    indices within a tensor. Unmapped entries from the input tensor are
    unmodified.

    This operation differs from a `tensor.gather/scatter` operation in that it
    can replicate values from the source tensor or omit values in the
    destination tensor. It differs from a "lane shuffle" in that the mapping
    need not be a permutation. It differs from a "swizzle" in that it cannot
    change the dimension of the result tensor.

    In the slot form of a ciphertext, this op is lowered to a "shift network"
    of ciphertext-plaintext mask, rotate, and sum operations by the
    `implement-shift-network` pass.
  }];

  let arguments = (ins AnyRankedTensor:$input, MappingLike:$permutation);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "operands attr-dict `:` type($input)";
  let hasVerifier = 1;
}

// Forces ops to use a general Attribute and dyn_cast to the specific kind of
// layout they support.
def LayoutLike : AnyAttrOf<[
  // A list of tuples [a, b, c, d] representing an explicit map (ct, slot) ->
  // (ct, slot) defined by f(a, b) = (c, d).
  AnyI64ElementsAttr,
  TensorExt_LayoutAttr,
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

def TensorExt_RotateAndReduceOp : TensorExt_Op<"rotate_and_reduce", [
  Pure,
  AllTypesMatch<["tensor", "output"]>,
  DeclareOpInterfaceMethods<PlaintextOperandInterface>
]> {
  let summary = "Performs a reduction of a periodically rotated tensor.";
  let description = [{
    This op reduces products of a plaintext with a periodically rotated
    tensor.

    In almost full generality, the reduction performed is

    $$ \sum_{i \in [0, n]} p(P, iT) \cdot rotate(v, iT) $$

    where $f$ is a function, $p(P, iT)$ is a function of a plaintext $P$ and
    $rotate(v, iT)$ is a rotation of the ciphertext $v$ with period $T$. The
    operation takes as input the ciphertext vector $v$, the period $T$, the
    number of reductions $n$, and a tensor of plaintext values

    `[p(P, 0), p(P, T), ..., p(P, (n-1)T)]`

    This can be used to implement a matrix vector product that uses a
    Halevi-Shoup diagonalization of the plaintext matrix. In this case, the
    reduction is

    $$ \sum_{i \in [0, n]} P(i) \cdot rotate(v, i) $$

    where $P(i)$ is the $i$th diagonal of the plaintext matrix and the period
    $T$ is $1$.

    An accumulation of the ciphertext slots is also handled via this operation
    by omitting the plaintext $p(P, Ti)$ argument and using a period of 1 with
    `n = |v|` so that the reduction is simply a sum of all rotations of the
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

  let extraClassDeclaration = [{
    // PlaintextOperandInterface method
    ::llvm::SmallVector<unsigned> maybePlaintextOperands() {
      return {1};
    }
  }];

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
