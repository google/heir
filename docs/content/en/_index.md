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

<img style="float:right;" src="/images/heir_logo_256x256.png" />

## What is HEIR?

HEIR is a compiler toolchain for
[fully homomorphic encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption)
(FHE). We aim to be the industry-standard compiler for FHE. Application
developers, compiler engineers, hardware designers, and cryptography researchers
can build upon HEIR to accelerate the research and development of
production-strength privacy-first software systems.

## Why HEIR?

For application developers, HEIR aims to provide a simple entrypoint to start
working with FHE. Write a program in Python, annotate the types to mark which
are secret, and HEIR will compile the rest.

For hardware designers, HEIR provides multiple layers of abstraction at which to
integrate code generation. This allows HEIR to support code generation for
hardware accelerators that implement high-level FHE operations like bootstrap,
as well as accelerators that operate at lower-level polynomial arithmetic.

For cryptography researchers, HEIR provides a convenient platform for research.
HEIR provides the compiler infrastructure and implements standard optimizations
from the literature. A researcher can focus on their novel optimization, and use
HEIR for its benchmarking, example programs, and comparisons to alternative
approaches. See [research with HEIR](/docs/research_with_heir/) for research
built on HEIR, and tips for doing research with HEIR.

## Project Goals

For an overview of the project, see
[our talk at FHE.org](https://www.youtube.com/watch?v=kqDFdKUTNA4).

- Compile high level programs to encrypted equivalents.
- Support all modern FHE schemes.
- Support code generation for FHE hardware accelerators, including GPU, TPU,
  FPGA, and custom ASICs.
- Support code generation for standard FHE libraries, such as OpenFHE and
  Lattigo.
- Support front-end languages for ease of development, such as Python and Torch.
- Design lower-level dialects for optimizing underlying abstract-algebraic
  operations (e.g., RNS polynomial arithmetic).
- Provide a platform for research into novel FHE optimizations.
- Provide a platform for benchmarking.

## Disclaimers

The HEIR codebase and documentation are maintained by Google.

Logo design by [Edward Chen](https://edwjchen.com/) and
[Erin Park](https://www.yerinstudio.com/).
