---
title: Ciphertext Packing System
weight: 9
---

This document describes HEIR's ciphertext packing system, including:

- A notation and internal representation of a ciphertext packing, which we call
  a _layout_.
- An abstraction layer to associate SSA values with layouts and manipulate and
  analyze them before a program is converted to concrete FHE operations.
- A variety of layouts and kernels from the FHE literature.
- A layout and kernel optimizer based on the
  [Fhelipe compiler](https://github.com/fhelipe-compiler/fhelipe).
- A layout conversion implementation of the
  [Vos-Vos-Erkin graph coloring algorithm](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20).

For background on what ciphertext packing is and its role in homomorphic
encryption, see
[this introductory blog post](https://www.jeremykun.com/2024/09/06/packing-matrix-vector-multiplication-in-fhe/).
The short version of that blog post is that the SIMD-style HE computational
model requires implementing linear-algebraic operations in terms of elementwise
additions, multiplications, and cyclic rotations of large-dimensional vectors
(with some exceptions like the
[Park-Gentry matrix-multiplication kernel](https://eprint.iacr.org/2025/448)).

Practical programs require many such operations, and the task of the compiler is
to jointly choose ciphertext packings and operation kernels so as to minimize
overall program latency. In this document we will call the joint process of
optimizing layouts and kernels by the name "layout optimization." In FHE
programs, runtime primarily comes from the quantity of rotation and bootstrap
operations, the latter of which is in turn approximated by multiplicative depth.
Metrics like memory requirements may also be constrained, but for most of this
document latency is the primary concern.

HEIR's design goal is to be an extensible HE compiler framework, we aim to
support a variety of layout optimizers and multiple layout representations. As
such, we separate the design of the layout representation from the details of
the layout optimizer, and implement lowerings for certain ops that can be reused
across optimizers.

This document will begin by describing the layout representation, move on to the
common, reusable components for working with that representation, and then
finally describe one layout optimizer implemented in HEIR based on Fhelipe.

## Layout representation

A _layout_ is a description of how cleartext data is organized within a list of
ciphertexts. In general, a layout is a partial function mapping from the index
set of a list of ciphertext _slots_ to the index set of a cleartext tensor. The
function describes which cleartext data value is stored at which ciphertext
slot.

A layout is _partial_ because not all ciphertext slots need to be used, and the
function uses ciphertext slots as the domain and cleartext indices as the
codomain because cleartext values may be replicated among multiple slots, but a
slot can store at most one cleartext value.

HEIR restricts the above definition of a layout as follows:

- The partial function must be expressible as a _Presburger relation_, which
  will be defined in detail below.
- Unmapped ciphertext slots always contain zero.\[^1\]

We claim that this subset of possible layouts is a superset of all layouts that
have been used in the FHE literature to date. For example, both the layout
notation of Fhelipe and the TileTensors of HeLayers are defined in terms of
specific parameterized quasi-affine formulas.

Next we define a Presburger relation, then move on to examples.

### Quasi-affine formulas and Presburger relations

**Definition:** A _quasi-affine_ formula is a multivariate formula built from
the following operations:

- Integer literals
- Integer-valued variables
- addition and subtraction
- multiplication by an integer constant
- floor- and ceiling-rounded division by a nonzero integer constant
- modulus by a nonzero integer constant

Using the BNF grammar from the
[MLIR website](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions),
we can also define it as

```
affine-expr ::= `(` affine-expr `)`
              | affine-expr `+` affine-expr
              | affine-expr `-` affine-expr
              | `-`? integer-literal `*` affine-expr
              | affine-expr `ceildiv` integer-literal
              | affine-expr `floordiv` integer-literal
              | affine-expr `mod` integer-literal
              | `-`affine-expr
              | bare-id
              | `-`? integer-literal
```

**Definition:** Let $d, r \\in \\mathbb{Z}\_{\\geq 0}$ represent a number of
domain and range dimensions, respectively. A _Presburger relation_ is a binary
relation over $\\mathbb{Z}^{d} \\times \\mathbb{Z}^{r}$ that can be expressed as
the solution to a set of equality and inequality constraints defined using
quasi-affine formulas.

We will use the Integer Set Library (ISL) notation to describe Presburger
relations. For an introduction to the ISL notation and library, see
[this article](https://jeremykun.com/2025/10/07/isl-a-primer/). For a
comprehensive reference, see
[the ISL manual](https://libisl.sourceforge.io/manual.pdf).

**Example 1:** Given a data vector of type `tensor<8xi32>` and a ciphertext with
32 slots, a layout that repeats the tensor cyclically is given as:

```
{
    [d] -> [ct, slot] :
    0 <= d < 8
    and ct = 0
    and 0 <= slot < 32
    and (d - slot) mod 8 = 0
}
```

From Example 1, we note that in HEIR the domain of a layout always aligns with
the shape of the domain tensor, and the range of a layout is always a 2D tensor
whose first dimension denotes the ciphertext index and whose second index is the
slot within that ciphertext.

**Example 2:** Given a data matrix of type `tensor<8x8xi32>` and 8 ciphertexts
with 32 slots each, the following layout implements the standard Halevi-Shoup
diagonal layout.

```
{
    [row, col] -> [ct, slot] :
    0 <= row < 8
    and 0 <= col < 8
    and 0 <= ct < 8
    and 0 <= slot < 32
    and (row - col + ct) mod 8 = 0
    and (row - slot) mod 32 = 0
}
```

Note, this layout implements a diagonal packing, and further replicates each
diagonal cyclically within a ciphertext.

### Layout attributes

Layouts are represented in HEIR via the `tensor_ext.layout` attribute. Its
argument includes a string using the ISL notation above. For example

```mlir
#tensor_layout = #tensor_ext.layout<
    "{ [i0] -> [ct, slot] : (slot - i0) mod 8 = 0 and ct = 0 and 1023 >= slot >= 0 and 7 >= i0 >= 0 }"
>
```

Generally, layout attributes are associated with an SSA value by being attached
to the op that owns the SSA value. In MLIR, which op owns the value has two
cases:

- For an op result, the layout attribute is stored on the op.
- For a block argument, the layout attribute is stored on the op owning the
  block, using the `OperandAndResultAttrInterface` to give a consistent API for
  accessing the attribute.

These two differences are handled properly by a helper library,
`lib/Utils/AttributeUtils.h`, which exposes setters and getters for layout
attributes. As of 2025-10-01, the system does not provide a way to handle ops
with multiple regions or multi-block regions.

For example:

```mlir
%1 = arith.addi %0, %1 {tensor_ext.layout = #layout_attr} : tensor<512xf32>
```

## Data-semantic and ciphertext-semantic tensors

In HEIR, before lowering to scheme ops, we distinguish between types in two
regimes:

- _Data-semantic tensors_, which are scalars and tensors that represent
  cleartext data values, largely unchanged from the original input program.
- _Ciphertext-semantic tensors_, which are rank-2 tensors that represent packed
  cleartext values in ciphertexts.

The task of analyzing an IR to determine which layouts and kernels to use
happens in the data-semantic regime. In these passes, chosen layouts are
persisted between passes as attributes on ops (see
[Layout attributes](#layout-attributes) above), and data types are unchanged.

In this regime, there are three special `tensor_ext` ops that are no-ops on
data-semantic type, but are designed to manipulate the layout attributes. These
ops are:

- `tensor_ext.assign_layout`, which takes a data-semantic value and a layout
  attribute, and produces the same data-semantic type. This is an "entry point"
  into the layout system and lowers to a loop that packs the data according to
  the layout.
- `tensor_ext.convert_layout`, which makes an explicit conversion between a
  data-semantic value's current layout and a new layout. Typically this lowers
  to a shift network.
- `tensor_ext.unpack`, which clears the layout attribute on a data-semantic
  value, and serves as an exit point from the layout system. This lowers to a
  loop which extracts the packed cleartext data back into user data.

A layout optimizer is expected to insert `assign_layout` ops for any server-side
cleartexts that need to be packed at runtime.

By contrast, in the ciphertext-semantic regime, all secret values are rank-2
tensors whose first axis indexes ciphertexts and whose second axis indexes slots
within ciphertexts. These tensors are subject to the constraints of the SIMD FHE
computational model (elementwise adds, muls, and structured rotations), though
the type system does not enforce this until `secret-to-<scheme>` lowerings,
which would fail if encountering an op that cannot be implemented in FHE.

We preserve the use of the `tensor` type here, rather than create new types, so
that we can reuse MLIR infrastructure. For example, if we were to use a new
tensor-like type for ciphertext-semantic tensors, we would not be able to use
`arith.addi` anymore, and would have to reimplement folding and canonicalization
patterns from MLIR in HEIR. In the future we hope MLIR will relax these
constraints via interfaces and traits, and at that point we could consider a
specialized type.

Before going on, we note that ciphertext-semantic tensors are agnostic to how
the "slots" are encoded in the underlying FHE scheme. In particular, slots could
correspond to evaluation points of an RNS polynomial, i.e., to "NTT form" slots.
But they could also correspond to the coefficients of an RNS polynomial in
coefficient form. As of 2025-10-01, HEIR's Fhelipe-inspired pipeline
materializes slots as NTT-form slots in all cases, but this is not required by
the layout system, and future layout optimizers may take into account
conversions between NTT and coefficient form as part of a layout conversion
step.

## HEIR's Fhelipe-inspired layout optimizer

### Pipeline overview

The `mlir-to-<scheme>` pipeline involves the following passes that manipulate
layouts:

- `layout-propagation`
- `fold-convert-layout-into-assign-layout`
- `layout-optimization`
- `convert-to-ciphertext-semantics`
- `implement-rotate-and-reduce`
- `add-client-interface`

The two passes that are closest to Fhelipe's design are `layout-propagation` and
`layout-optimization`. The former sets up initial default layouts for all values
and default kernels for all ops that need them, and propagates them forward,
inserting layout conversion ops as needed to resolve layout mismatches. The
latter does a backwards pass, jointly choosing more optimal kernels and
attempting to hoist layout conversions earlier in the IR. If layout conversions
are hoisted all the way to function arguments then they are "free" because they
can be merged into client preprocessing.

Next we will outline the responsibility of each pass in detail. The
documentation page for each of these passes is linked in each section, and
contains doctests as examples that are kept in sync with the implementation of
the pass.

**FIXME: add doctests to all of these passes**

### `layout-propagation`

**FIXME: rename new-layout-propagation to match this**

The [`layout-propagation`](/docs/passes/#-layout-propagation) pass runs a
forward pass through the IR to assign default layouts to each SSA value that
needs one, and a default kernel to each operation that needs one.

For each secret-typed function argument, no layout can be inferred, so a default
layout is assigned. The default layout for scalars is to repeat the scalar in
every slot of a single ciphertext. The default layout for tensors is a row-major
layout into as many ciphertexts as are needed to fit the tensor.

Then layouts are propagated forward through the IR. For each op, a default
kernel is chosen, and if the layouts of the operands are already set and agree,
the result layout is inferred according to the kernel.

If the layouts are not compatible with the default kernel, a `convert_layout` op
is inserted to force compatibility. If one or more operands has a layout that is
not set (which can happen if the operand is a cleartext value known to the
server), then a compatible layout is chosen and an `assign_layout` op is
inserted to persist this information for later passes.

**FIXME: how are things chosen when the tensor sizes are not a power of two?**

### `fold-convert-layout-into-assign-layout`

Because `layout-propagation` may have inserted some redundant conversions,
sequences of `assign_layout` followed by `convert_layout` are folded together
into combined `assign_layout` ops.

**FIXME: should we just add this pattern at the end of `layout-propagation`?**

### `layout-optimization`

The [`layout-optimization`](/docs/passes/#-layout-optimization) pass has two
main goals: to choose better kernels for ops, and to try to eliminate
`convert_layout` ops. It does this by running a backward pass through the IR. If
it encounters an op that is followed by a `convert_layout` op, it attempts to
hoist the `convert_layout` through the op to its arguments.

In doing this, it must consider:

- Changing the kernel of the op, and the cost of implementing the kernel. E.g.,
  a new kernel may be better for the new layout of the operands.
- Whether the new layout of op results still need to be converted, and the new
  cost of these conversions. E.g., if the op result has multiple uses, or the op
  result had multiple layout conversions, only one of which is hoisted.
- The new cost of operand layout conversions. E.g., if a layout conversion is
  hoisted to one operand, it may require other operands to be converted to
  remain compatible.

In all of the above, the "cost" includes an estimate of the latency of a kernel,
an estimate of the latency of a layout conversion, as well as the knowledge that
some layout conversions may be free or cheaper because of their context in the
IR.

The cost of a kernel is hard-coded in a table.

**FIXME: kernels are currently considered free, cf. #1888**

The cost of a layout conversion is estimated by simulating what the
`implement-shift-network` would do if it ran on a layout conversion. And
`layout-optimization` includes analyses that allow it to determine a folded cost
for layout conversions that occur after other layout conversions, as well as the
free cost of layout conversions that occur at function arguments, after
`assign_layout` ops, or separated from these by ops that do not modify a layout.

**FIXME: extend this pass to also pick the min kernel for an op even if it
doesn't have a trailing layout conversion.**\*

After the backward pass, any remaining `convert_layout` ops at the top of a
function are hoisted into function arguments and folded into existing layout
attributes.

### `convert-to-ciphertext-semantics`

The
[`convert-to-ciphertext-semantics`](/docs/passes/#-convert-to-ciphertext-semantics)
pass has two responsibilities that must happen at the same time:

- Converting all data-semantic values to ciphertext-semantic values with
  corresponding types.
- Implementing FHE kernels for all ops as chosen by earlier passes.

After this pass is complete, the IR must be in the ciphertext-semantic regime
and all operations on secret-typed values must be constrained by the SIMD FHE
computational model.

In particular, this pass implements `assign_layout` as an explicit loop that
packs cleartext data into ciphertext slots according to the layout attribute. It
also implements `convert_layout` as a shift network, which is a sequence of
plaintext masks and rotations that can arbitrarily (albeit expensively) shuffle
data in slots. This step can be isolated via the
[`implement-shift-network`](/docs/passes/#-implement-shift-network) pass, but
the functionality is inlined in this pass since it must happen at the same time
as type conversion.

When converting function arguments, any secret-typed argument is assigned a new
attribute called `tensor_ext.original_type`, which records the original
data-semantic type of the argument as well as the layout used for its packing.
This is used later by the `add-client-interface` pass to generate client-side
encryption and decryption helper functions.

### `implement-rotate-and-reduce`

Some kernels rely on a baby-step giant-step optimization, and defer the
implementation of that operation so that canonicalization patterns can optimize
them. Instead they emit a `tensor_ext.rotate_and_reduce` op. The
[`implement-rotate-and-reduce`](/docs/passes/#-implement-rotate-and-reduce) pass
implements this op using baby-step giant-step, or other approaches that are
relevant to special cases.

**FIXME: canonicalizer is not actually run between the two passes above**

### `add-client-interface`

The [`add-client-interface`](/docs/passes/#-add-client-interface) pass inserts
additional functions that can be used by the client to encrypt and decrypt data
according to the layouts chosen by the layout optimizer.

It fetches the `original_type` attribute on function arguments, and generates an
encryption helper function for each secret argument, and a decryption helper
function for each secret return type.

These helper functions use `secret.conceal` and `secret.reveal` for
scheme-agnostic encryption and decryption, but eagerly implement the packing
logic as a loop, equivalently to how `assign_layout` is lowered in
`convert-to-ciphertext-semantics`, and adding an analogous lowering for
`tensor_ext.unpack`.

## Reusable components for working with layouts

### Lowering data-semantic ops with FHE kernels

Any layout optimizer will eventually need to convert data-semantic values to
ciphertext-semantic tensors. In doing this, all kernels need to be implemented
in one pass at the same time that the types are converted.

The `convert-to-ciphertext-semantics` pass implements this conversion without
making any decisions about which layouts or kernels to use. In particular, for
ops that have multiple supported kernels, it picks the kernel to use based on
the `kernel` attribute on the op (cf. `secret::SecretDialect::kKernelAttrName`).

In this way, we decouple the decision of which layout and kernel to use (the
optimizer's job) from the implementation of that kernel (the lowering's job).
Ideally all layout optimizer pipelines can reuse this pass, which avoids the
common pitfalls associated with writing dialect conversion passes. New kernels,
similarly, can be primarily implemented as described in the next section.

Finally, if a new optimizer or layout notation is introduced into HEIR, it
should ultimately be converted to use the same `tensor_ext.layout` attribute so
that it can reuse the lowerings of ops like `tensor_ext.assign_layout` and
`tensor_ext.unpack`.

### Testing kernels and layouts

Writing kernels can be tricky, so HEIR provides a simplified framework for
implementing kernels which allows them to be unit-tested in isolation, while the
lowering to MLIR is handled automatically by a common library.

The implementation library is called `ArithmeticDag`. Some initial
implementations are in `lib/Kernel/KernelImplementation.h`, and example unit
tests are in `lib/Kernel/*Test.cpp`. Then a class called
`IRMaterializingVisitor` walks the DAG and generates MLIR code.

Similarly, `lib/Utils/Layout/Evaluate.h` provides helper functions to
materialize layouts on test data-semantic tensors, which can be combined with
`ArithmeticDag` to unit-test a layout and kernel combination without ever
touching MLIR.

### Manipulating layouts

The directory `lib/Utils/Layout` contains a variety of helper code for
manipulating layout relations, including:

- Constructing or testing for common kinds of layouts, such as row-major,
  diagonal, and layouts related to particular machine learning ops like
  convolution.
- Generating explicit loops that iterate over the space of points in a layout,
  which is used to generate packing and unpacking code.
- Helpers for hoisting layout conversions through ops.

These are implemented using two APIs: one is the Fast Presburger Library (FPL),
which is part of MLIR and includes useful operations like composing relations
and projecting out dimensions. The other is the Integer Set Library (ISL), which
is a more fully-featured library that supports code generation and advanced
analyses and simplification routines. As we represent layouts as ISL strings, we
include a two-way interoperability layer that converts between ISL and FPL
representations of the same Presburger relation.

## A case study: the Orion kernel

**FIXME: write this section**

## FAQ

### Can users define kernels without modifying the compiler?

**No** (as of 2025-10-01). However, a kernel DSL is **in scope** for HEIR. Reach
out if you'd like to be involved in the design.

\[^1\]: This may be relaxed in the future with additional static analyses that
can determine that some slots are never read.
