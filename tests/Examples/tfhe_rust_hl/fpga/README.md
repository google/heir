# End to end Rust codegen tests - High Level API

These tests generate Rust code for the
[tfhe-rs](https://github.com/zama-ai/tfhe-rs) backend library, including
compiling the generated Rust source and running the resulting binary on Belfort
FPGAs. These target the integer plaintexts and the accompanying library.

To avoid introducing these large dependencies into the entire project, tests are
manual, and require the system they're running on to have
[Cargo](https://doc.rust-lang.org/cargo/index.html) installed. During the test,
cargo will fetch and build the required dependencies.

## Prerequisites

Install the heir toolchain from the nightly release:

```bash
wget https://github.com/google/heir/releases/download/nightly/heir-translate
chmod +x heir-translate
```

## FPGA Backend

The FPGA backend is built using AWS EC2 F2 instances. They are configured with
the Belfort Amazon Machine Image (AMI). Instructions for building the AMI can be
found on [Belfort's Github](https://github.com/belfortlabs/hello-fpga).

Cargo.toml depends on a `tfhe-rs` fork that contains the set-up for the FPGA
backend. The following 3 lines of code were added to `main.rs` to connect to the
FPGA:

```rust
let mut fpga_key = BelfortServerKey::from(&server_key);
fpga_key.connect();
set_server_key(fpga_key.clone());
```

## Running the example

To transpile the .mlir file to rust code, run the following command:

```bash
cd heir/tests/Examples/tfhe_rust_hl/fpga
~/heir-translate ./arith.mlir --emit-tfhe-rust-hl > ./src/fn_under_test.rs
```

Running the example is done by replacing input 1 with an integer and executing
the following command:

```bash
cargo run --release -- {$INPUT_1}
```

<!-- mdformat global-off -->
