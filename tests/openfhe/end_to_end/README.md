# End to end OpenFHE codegen tests

These tests exercise OpenFHE codegen for the
[OpenFHE](https://github.com/openfheorg/openfhe-development) backend library,
including compiling the generated C++ source and running the resulting binary.

OpenFHE is added as a project-level dependency (unlike the `tfhe-rs` end-to-end
tests) and built from source.
