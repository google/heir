# End to end Rust codegen tests - Boolean FPGA

These tests exercise Rust codegen for the
[tfhe-rs](https://github.com/zama-ai/tfhe-rs) backend library, including
compiling the generated Rust source and running the resulting binary. This sets
tests are specifically of the boolean plaintexts, accompanying COSIC-KU Leuven
version of the library, and the
[FPT-accelerator](https://eprint.iacr.org/2022/1635).

> :warning: Not possible to run these tests without the COSIC extension of
> TFHE-rs and FPT-accelerator

To avoid introducing these large dependencies into the entire project, these
tests are manual, and require the system they're running on to have
[Cargo](https://doc.rust-lang.org/cargo/index.html) installed. During the test,
cargo will fetch and build the required dependencies, and `Cargo.toml` in this
directory effectively pins the version of `tfhe` supported.

Use the following command to run the tests in this directory, where the default
Cargo home `$HOME/.cargo` may need to be replaced by your custom `$CARGO_HOME`,
if you overrode the default option when installing Cargo.

```bash
  bazel query "filter('.mlir.test$', //tests/tfhe_rust_bool/end_to_end_fpga/...)" \
  | xargs bazel test --sandbox_writable_path=$HOME/.cargo "$@"
```

Manually generate the Rust code for the CGGI lowering:

```bash
  bazel run //tools:heir-opt -- -cse --straight-line-vectorize --cggi-to-tfhe-rust-bool -cse $(pwd)/tests/cggi_to_tfhe_rust_bool/add_bool.mlir | bazel run //tools:heir-translate -- --emit-tfhe-rust-bool
```

The `manual` tag is added to the targets in this directory to ensure that they
are not run when someone runs a glob test like `bazel test //...`.

If you don't do this correctly, you will see an error like this:

```
# .---command stderr------------
# |     Updating crates.io index
# |  Downloading crates ...
# |   Downloaded memoffset v0.9.0
# | error: failed to download replaced source registry `crates-io`
# |
# | Caused by:
# |   failed to open `/home/you/.cargo/registry/cache/index.crates.io-6f17d22bba15001f/memoffset-0.9.0.crate`
# |
# | Caused by:
# |   Read-only file system (os error 30)
# `-----------------------------
```
