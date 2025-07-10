---
title: Design
weight: 50
---

HEIR is a compiler toolchain that allows the compilation of high-level programs
to equivalent programs that operate on encrypted data.

HEIR is built in the [MLIR](https://mlir.llvm.org/) framework.

HEIR defines dialects at various layers of abstraction, from high-level
scheme-agnostic operations on secret types to low-level polynomial arithmetic.
The diagram below shows some of the core HEIR dialects, and the compilation flow
is generally from the top of the diagram downward.

<img style="display:block; margin-left:75px; width:500px;" src="/images/dialect-diagram.png" />

The pages in this section describe the design of various subcomponents of HEIR.
