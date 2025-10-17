# End to end lattigo codegen tests

These tests exercise Lattigo codegen for the
[Lattigo](https://github.com/tuneinsight/lattigo) backend library, including
compiling the generated golang source and running the resulting binary.

Lattigo is added as a bazel project-level dependency (unlike the `tfhe-rs`
end-to-end tests) and built from source.

<!-- mdformat global-off -->
