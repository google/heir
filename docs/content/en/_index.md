---
title: "HEIR: Homomorphic Encryption Intermediate Representation"
linkTitle: Home
menu: {main: {weight: 1}}
weight: 1

cascade:
  - type: "blog"
    # uncomment this to make blog disappear from the sidebar
    # toc_root: true
    _target:
      path: "/blog/**"
  - type: "docs"
    _target:
      path: "/**"
---

## What is HEIR?

HEIR is a compiler toolchain for [fully homomorphic
encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption) (FHE).
We aim to standardize a set of intermediate representations related to FHE,
which compiler engineers, hardware designers, and cryptography researchers
can build upon to accelerate the research and development
of production-strength privacy-first software systems.

HEIR is built in the [MLIR](https://mlir.llvm.org/) framework.

## Project Goals

- Provide MLIR dialects for all modern FHE schemes.
- Design MLIR dialects that appropriately abstract across the many flavors of related schemes.
- Design lower-level dialects for optimizing underlying abstract-algebraic operations (e.g., modular polynomial arithmetic).
- Provide hardware accelerator designers an easy path to integrate, so that a wide variety of FHE programs, optimizations, and parameter choices can be compared across accelerators.
- Provide a platform for research into novel FHE optimizations.
- Provide integrations with multiple front-end languages, such as ClangIR and TensorFlow.

## Partners

HEIR is an industry-wide collaboration with many stakeholders, including
(in alphabetical order):

- CryptoLab
- Duke University
- ETH Zurich
- Google
- Intel
- KU Leuven
- NVIDIA
- Niobium Microsystems
- Optalysys
- Royal Holloway
- Samsung
- Zama

The HEIR codebase and documentation are maintained by Google.
