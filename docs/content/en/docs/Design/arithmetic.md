## Representing FHE SIMD Operations
<!-- mdformat off(yaml frontmatter) -->
---
title: Arithmetic FHE Pipeline
weight: 30
---
<!-- mdformat on -->

Below, we describe the process of compiling programs for Ring-LWE-based FHE schemes, such as BGV/BFV or CKKS.


## Input Programs
<!-- TODO (#629): Once a design document for frontends/inputs exists, the following paragraph should migrate there, as it's not arithmetic-FHE specific -->
In general, HEIR expects input programs to fulfill a few criteria:
* **Code to be translated must be in a single module (`builtin.module`)**
  (though this can frequently be omitted in textual IR as the parser auto-inserts it).
* **Code to be translated must be contained in functions (`func.func`).**
* **Code to be translated uses HEIR's system of [Secret](https://heir.dev/docs/design/secret) annotations.**
* **Code to be translated must be free of (secret)-data-dependent code flow**
  (though HEIR offers transformations that automate this translation.)
* **All code to be translated must be side effect free.**
  As a result, only types with value semantics are permissible (e.g., `tensor` is suitable, but `memref` is not.)
* **Arithmetic operations must not overflow**
  (otherwise, equivalence between the input program and the homomorphic computation is not guaranteed).

### Arithmetic Restrictions
In addition, while the boolean FHE approach supports a wide variety of `arith` operations natively,
arithmetic FHE only supports a small subset of `arith` operations.
Except for addition, subtraction and multiplication, which correspond to the native homomorphic operations,
general `arith` operations, even when supported,
might generate significant amounts of FHE operations and result in an infeasibly complex program.

### SIMD Paradigm
We refer to [SIMD Optimizations](https://heir.dev/docs/design/simd/)
for a discussion on how to express FHE SIMD operations
and the optimizations HEIR offers around this.

> **Warning**
> The current HEIR passes frequently support only one-dimensional tensors,
> and might either throw errors or produce incorrect code
> for higher-dimensional inputs.
> As a workaround, one could reshape all multi-dimensional tensors into one-dimensional tensors,
> but MLIR/HEIR currently do not provide a pass to automate this process.

### Other Limitations
While it is technically possible to compile code that already contains both client- and server-side computations,
or to compile two separate client and server programs while ensuring they "match up" correctly,
this is an advanced and, at the current time, unsupported use case.
In the following, we therefore assume that the program contains only the server-side computation,
and will demonstrate how HEIR can automatically generate suitable client-side encrypt/decrypt functions.


### Compilation Flow
Starting with a high-level program that fulfills the input program requirements,
we describe the compilation flow below.
We assume the program has already been transformed to the SIMD paradigm,
either manually or through the `-heir-simd-vectorizer` pipeline
(See [SIMD Optimizations](https://heir.dev/docs/design/simd/)).

1. **Secret Annotation & Wrapping**
   See also the [Secret](https://heir.dev/docs/design/secret) Design documentation,
   In the input IR, function arguments that should be secret are annotated with `{{secret.secret}}`,
   either manually or via the `entry-function=...` option of the `--secretize` pass
   (this option is also present on pipelines that internally use the `secretize` pass).
   <!-- TODO: I think we should change this, and make the secretize step an explicit step,
              to allow better reuse of the mlir-to-... pipelines for frontends/advanced inputs -->
    These function arguments are then converted to `secret.secret<original-type>` types,
    and the function body wrapped in a `secret.generic` operation.

1. **Distribute Secret**
   In order to determine which operations actually need to be translated to FHE,
   and which can remain (e.g., computation of indices in the clear, helper code, etc),
   the `--distribute-generic` pass transforms the code so that all operations
   that must be translated to FHE are isolated into their own `secret.generic` wrapper.

1. **Parameter Selection**
   HEIR can do basic automatic parameter selection based on the MultDepth of a program.
   However, at the moment, this is available only when targeting OpenFHE
   and uses the OpenFHE parameter selection internally.

   At the moment, parameter selection for other targets must be done manually,
   with the desired polynomial degree and coefficient moduli passed in as pass options
   (`poly-mod-degree=...` and `coefficient-mod-bits=...`).
   <!-- TODO: this is insufficient for real-world parameter sets.
              We should support passing in exact moduli,
              both for bignum and RNS based settings -->

1. **Translation to FHE Scheme**
   Each operation is translated to the native operations of the FHE scheme.
   As mentioned above, this pass requires that, at the very least, the polynomial degree is specified,
   as this is necessary to determine how `tensor` types are translated to ciphertexts.
   As mentioned in the introduction, non-arithmetic operations (e.g., `arith.select` or `arith.cmpi`),
   where supported, will produce significant amounts of FHE operations for each such operation.
   This can quickly make a program infeasible to run under FHE.

1. **Noise Management**
   As part of the basic translation, HEIR naively inserts relinearizations after each multiplication.
   <!-- TODO: And I guess we should also issue a modswitch or rescale? -->
   <!-- TODO: Can the type system even handle RNS, where the type changes? -->


1. **Translation to LWE**
  Each `bgv` operation is then translated to the common (R)LWE abstraction,
  which models common components/gadgets used in lattice cryptography.

1. **Translation to Polynomial**
