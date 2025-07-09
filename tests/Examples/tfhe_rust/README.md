# End to end Rust codegen tests

These tests exercise Rust codegen for the
[tfhe-rs](https://github.com/zama-ai/tfhe-rs) backend library, including
compiling the generated Rust source and running the resulting binary.

The tests are generated using a custom Bazel macro found in
[test.bzl](tests/Examples/tfhe_rust/test.bzl). `tfhe-rs` is added a
project-level dependency, and it's version is pinned in
[MODULE.bazel](MODULE.bazel).
