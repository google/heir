---
title: 'HEIR: Homomorphic Encryption Intermediate Representation'
linkTitle: Home
menu: {main: {weight: 1}}
weight: 1

cascade:
  - type: blog
    # Comment this to make blog appear in the main sidebar nav.
    # It shows all blog posts expanded, and is too long.
    toc_root: true
    _target:
      path: /blog/**
  - type: docs
    _target:
      path: /**
---

## What is HEIR?

HEIR is a compiler toolchain for
[fully homomorphic encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption)
(FHE). We aim to standardize a set of intermediate representations related to
FHE, which compiler engineers, hardware designers, and cryptography researchers
can build upon to accelerate the research and development of production-strength
privacy-first software systems.

HEIR is built in the [MLIR](https://mlir.llvm.org/) framework.

For an overview of the project's goals, see
[our talk at FHE.org](https://www.youtube.com/watch?v=kqDFdKUTNA4).

To see the dialects and possible flows, take a look at the diagram below: {{%
figure src="/images/dialects.svg" link="/images/dialects.svg" %}}

## Project Goals

- Provide MLIR dialects for all modern FHE schemes.
- Design MLIR dialects that appropriately abstract across the many flavors of
  related schemes.
- Design lower-level dialects for optimizing underlying abstract-algebraic
  operations (e.g., modular polynomial arithmetic).
- Provide hardware accelerator designers an easy path to integrate, so that a
  wide variety of FHE programs, optimizations, and parameter choices can be
  compared across accelerators.
- Provide a platform for research into novel FHE optimizations.
- Provide a platform for benchmarking.
- Provide integrations with multiple front-end languages, such as ClangIR and
  TensorFlow.

## Partners

TODO(b/287634511): add list of partner companies and universities

The HEIR codebase and documentation are maintained by Google.
