<!-- mdformat off(yaml frontmatter) -->
---
title: Secret
weight: 9
---
<!-- mdformat on -->

The [`secret` dialect](https://heir.dev/docs/dialects/secret/) contains types
and operations to represent generic computations on secret data. It is intended
to be a high-level entry point for the HEIR compiler, agnostic of any particular
FHE scheme.

Most prior FHE compiler projects design their IR around a specific FHE scheme,
and provide dedicated IR types for the secret analogues of existing data types,
and/or dedicated operations on secret data types. For example, the Concrete
compiler has `!FHE.eint<32>` for an encrypted 32-bit integer, and `add_eint` and
similar ops. HECO has `!fhe.secret<T>` that models a generic secret type, but
similarly defines `fhe.add` and `fhe.multiply`, and other projects are similar.

The problem with this approach is that it is difficult to incorporate the apply
upstream canonicalization and optimization passes to these ops. For example, the
`arith` dialect in MLIR has
[canonicalization patterns](https://sourcegraph.com/github.com/llvm/llvm-project@0ab3f160c4bff1c7d57c046b95ab8c5035ae986f/-/blob/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td)
that must be replicated to apply to FHE analogues. One of the goals of HEIR is
to reuse as much upstream infrastructure as possible, and so this led us to
design the `secret` dialect to have both generic types and generic computations.
Thus, the `secret` dialect has two main parts: a `secret<T>` type that wraps any
other MLIR type `T`, and a `secret.generic` op that lifts any computation on
cleartext to the "corresponding" computation on secret data types.

## Overview with BGV-style lowering pipeline

Here is an example of a program that uses `secret` to lift a dot product
computation:

```mlir
func.func @dot_product(
    %arg0: !secret.secret<tensor<8xi16>>,
    %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<i16> {
  %c0_i16 = arith.constant 0 : i16
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %1 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %arg2[%arg4] : tensor<8xi16>
      %extracted_0 = tensor.extract %arg3[%arg4] : tensor<8xi16>
      %2 = arith.muli %extracted, %extracted_0 : i16
      %3 = arith.addi %arg5, %2 : i16
      affine.yield %3 : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

The operands to the `generic` op are the secret data types, and the op contains
a single region, whose block arguments are the corresponding cleartext data
values. Then the region is free to perform any computation, and the values
passed to `secret.yield` are lifted back to `secret` types. Note that
`secret.generic` is not isolated from its enclosing scope, so one may refer to
cleartext SSA values without adding them as generic operands and block
arguments.

Clearly `secret.generic` does not actually do anything. It is not decrypting
data. It is merely describing the operation that one wishes to apply to the
secret data in more familiar terms. It is a structural operation, primarily used
to demarcate which operations involve secret operands and have secret results,
and group them for later optimization. The benefit of this is that one can write
optimization passes on types and ops that are not aware of `secret`, and they
will naturally match on the bodies of `generic` ops.

For example, here is what the above dot product computation looks like after
applying the `-cse -canonicalize -heir-simd-vectorizer` passes, the
implementations of which do not depend on `secret` or `generic`.

```mlir
func.func @dot_product(
    %arg0: !secret.secret<tensor<8xi16>>,
    %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<i16> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %1 = arith.muli %arg2, %arg3 : tensor<8xi16>
    %2 = tensor_ext.rotate %1, %c4 : tensor<8xi16>, index
    %3 = arith.addi %1, %2 : tensor<8xi16>
    %4 = tensor_ext.rotate %3, %c2 : tensor<8xi16>, index
    %5 = arith.addi %3, %4 : tensor<8xi16>
    %6 = tensor_ext.rotate %5, %c1 : tensor<8xi16>, index
    %7 = arith.addi %5, %6 : tensor<8xi16>
    %extracted = tensor.extract %7[%c7] : tensor<8xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

The canonicalization patterns for `secret.generic` apply a variety of
simplifications, such as:

-   Removing any unused or non-secret arguments and return values.
-   Hoisting operations in the body of a `generic` that only depend on cleartext
    values to the enclosing scope.
-   Removing any `generic` ops that use no secrets at all.

These can be used together with the
[`secret-distribute-generic` pass](https://heir.dev/docs/passes/secretpasses/#-secret-distribute-generic)
to split an IR that contains a large `generic` op into `generic` ops that
contain a single op, which can then be lowered to a particular FHE scheme
dialect with dedicated ops. This makes lowering easier because it gives direct
access to the secret version of each type that is used as input to an individual
op.

As an example, a single-op secret might look like this (taken from the larger
example below. Note the use of a cleartext from the enclosing scope, and the
proximity of the secret type to the op to be lowered.

```mlir
  %c2 = arith.constant 2 : index
  %3 = secret.generic ins(%2 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>):
    %8 = tensor_ext.rotate %arg2, %c2 : tensor<8xi16>, index
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
```

For a larger example, applying `--secret-distribute-generic --canonicalize` to
the IR above:

```mlir
func.func @dot_product(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<i16> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %8 = arith.muli %arg2, %arg3 : tensor<8xi16>
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %1 = secret.generic ins(%0 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>):
    %8 = tensor_ext.rotate %arg2, %c4 : tensor<8xi16>, index
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %2 = secret.generic ins(%0, %1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %8 = arith.addi %arg2, %arg3 : tensor<8xi16>
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %3 = secret.generic ins(%2 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>):
    %8 = tensor_ext.rotate %arg2, %c2 : tensor<8xi16>, index
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %4 = secret.generic ins(%2, %3 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %8 = arith.addi %arg2, %arg3 : tensor<8xi16>
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %5 = secret.generic ins(%4 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>):
    %8 = tensor_ext.rotate %arg2, %c1 : tensor<8xi16>, index
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %6 = secret.generic ins(%4, %5 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %8 = arith.addi %arg2, %arg3 : tensor<8xi16>
    secret.yield %8 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  %7 = secret.generic ins(%6 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>):
    %extracted = tensor.extract %arg2[%c7] : tensor<8xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %7 : !secret.secret<i16>
}
```

And then lowering it to `bgv` with `--secret-to-bgv="poly-mod-degree=8"` (the
pass option matches the tensor size, but it is an unrealistic FHE polynomial
degree used here just for demonstration purposes). Note type annotations on ops
are omitted for brevity.

```mlir
#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#params = #lwe.rlwe_params<ring = <cmod=463187969, ideal=#_polynomial.polynomial<1 + x**8>>>
!ty1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=tensor<8xi16>>
!ty2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i16>

func.func @dot_product(%arg0: !ty1, %arg1: !ty1) -> !ty2 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = bgv.mul %arg0, %arg1
  %1 = bgv.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>}
  %2 = bgv.rotate %1, %c4
  %3 = bgv.add %1, %2
  %4 = bgv.rotate %3, %c2
  %5 = bgv.add %3, %4
  %6 = bgv.rotate %5, %c1
  %7 = bgv.add %5, %6
  %8 = bgv.extract %7, %c7
  return %8
}
```

## Differences for CGGI-style pipeline

The `tosa-to-boolean-tfhe` and related pipelines add a few additional steps. The
main goal here is to apply a hardware circuit optimizer to blocks of standard
MLIR code (inside `secret.generic` ops) which converts the computation to an
optimized boolean circuit with a desired set of gates. Only then is
`-secret-distribute-generic` applied to split the ops up and lower them to the
`cggi` dialect. In particular, because passing an IR through the circuit
optimizer requires unrolling all loops, one useful thing you might want to do is
to optimize only the *body* of a for loop nest.

To accomplish this, we have two additional mechanisms. One is the pass option
`ops-to-distribute` for `-secret-distribute-generic`, which allows the user to
specify a list of ops that `generic` should be split across, and all others left
alone. Specifying `affine.for` here will pass `generic` through the `affine.for`
loop, but leave its body intact. This can also be used with the `-unroll-factor`
option to the `-yosys-optimizer` pass to partially unroll a loop nest and pass
the partially-unrolled body through the circuit optimizer.

The other mechanism is the `secret.separator` op, which is a purely structural
op that demarcates the boundary of a subset of a block that should be jointly
optimized in the circuit optimizer.

For example, the following `tosa` ops lower to multiple linalg instructions, and
hence multiple for loops, that we want to pass to a circuit optimizer as a unit.
The `secret.separator` ops surrounding the op are preserved through the
lowering.

```mlir
func.func @main(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x16xi32> {
  secret.separator
  %4 = "tosa.const"() {value = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> : tensor<16xi32>} : () -> tensor<16xi32>
  %5 = "tosa.const"() {value = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> : tensor<16x1xi8>} : () -> tensor<16x1xi8>
  %6 = "tosa.fully_connected"(%arg0, %5, %4) {quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>} : (tensor<1x1xi8>, tensor<16x1xi8>, tensor<16xi32>) -> tensor<1x16xi32>
  secret.separator
  return %6 : tensor<1x16xi32>
}
```

After running `--tosa-to-boolean-tfhe` and dumping the IR after the linalg ops
are lowered to loops, we can see the `secret.separator` ops enclose the lowered
ops, with the exception of some pure ops that are speculatively executed.

```mlir
func.func @main(%arg0: memref<1x1xi8, strided<[?, ?], offset: ?>> {secret.secret}) -> memref<1x16xi32> {
  %c-128_i32 = arith.constant -128 : i32
  %0 = memref.get_global @__constant_16xi32 : memref<16xi32>
  %1 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
  secret.separator
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      %2 = affine.load %1[%arg2, %arg1] : memref<16x1xi8>
      affine.store %2, %alloc[%arg1, %arg2] : memref<1x16xi8>
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      %2 = affine.load %0[%arg2] : memref<16xi32>
      affine.store %2, %alloc_0[%arg1, %arg2] : memref<1x16xi32>
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 1 {
        %2 = affine.load %arg0[%arg1, %arg3] : memref<1x1xi8, strided<[?, ?], offset: ?>>
        %3 = affine.load %alloc[%arg3, %arg2] : memref<1x16xi8>
        %4 = affine.load %alloc_0[%arg1, %arg2] : memref<1x16xi32>
        %5 = arith.extsi %2 : i8 to i32
        %6 = arith.subi %5, %c-128_i32 : i32
        %7 = arith.extsi %3 : i8 to i32
        %8 = arith.muli %6, %7 : i32
        %9 = arith.addi %4, %8 : i32
        affine.store %9, %alloc_0[%arg1, %arg2] : memref<1x16xi32>
      }
    }
  }
  secret.separator
  memref.dealloc %alloc : memref<1x16xi8>
  return %alloc_0 : memref<1x16xi32>
}
```

We decided to use the `separator` op over a few alternatives:

-   Grouping by `secret.generic`: these `tosa` ops must be bufferized, but
    `secret` types cannot participate in bufferization (see the Limitations
    section).
-   Grouping by basic blocks: `secret.generic` is a single-block op with a yield
    terminator, and grouping by blocks would require us to change this.
-   Grouping by regions: SSA values generated by a region are not visible to the
    enclosing scope, so we would need to have the region-bearing op return
    values, which is tedious to organize.
-   Attaching attributes to ops that should be grouped together: this would not
    be preserved by upstream lowerings and optimization passes.

## `generic` operands

`secret.generic` takes any SSA values as legal operands. They may be `secret`
types or non-secret. Canonicalizing `secret.generic` removes non-secret operands
and leaves them to be referenced via the enclosing scope (`secret.generic` is
not `IsolatedFromAbove`).

This may be unintuitive, as one might expect that only secret types are valid
arguments to `secret.generic`, and that a verifier might assert non-secret args
are not present.

However, we allow non-secret operands because it provides a convenient scope
encapsulation mechanism, which is useful for the `--yosys-optimizer` pass that
runs a circuit optimizer on individual `secret.generic` ops and needs to have
access to all SSA values used as inputs. The following passes are related to
this functionality:

- `secret-capture-generic-ambient-scope`
- `secret-generic-absorb-constants`
- `secret-extract-generic-body`

Due to the canonicalization rules for `secret.generic`, anyone using these
passes as an IR organization mechanism must be sure not to canonicalize before
accomplishing the intended task.

## Limitations

### Bufferization

Secret types cannot participate in bufferization passes. In particular,
`-one-shot-bufferize` hard-codes the notion of tensor and memref types, and so
it cannot currently operate on `secret<tensor<...>>` or `secret<memref<...>>`
types, which prevents us from implementing a bufferization interface for
`secret.generic`. This was part of the motivation to introduce
`secret.separator`, because `tosa` ops like a fully connected neural network
layer lower to multiple linalg ops, and these ops need to be bufferized before
they can be lowered further. However, we want to keep the lowered ops grouped
together for circuit optimization (e.g., fusing transposes and constant weights
into the optimized layer), but because of this limitation, we can't simply wrap
the `tosa` ops in a `secret.generic` (bufferization would fail).
