# Polygeist

In lieu of actually integrating Polygeist into HEIR (they have incompatibly
pinned upstream LLVM hashes), run polygeist as a separate tool and check in both
the C++ and output MLIR sources into this directory.

```bash
Polygeist/build/bin/cgeist \
  '-function=*' \
  -raise-scf-to-affine <... more passes ...> \
  -S -O3 \
  input_cc_file.cpp > test_file.mlir
```
