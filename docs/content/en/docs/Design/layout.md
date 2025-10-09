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

## Reusable components for working with layouts

## HEIR's Fhelipe-inspired layout optimizer
