---
title: Pipelines
weight: 60
---

## `heir-opt`

### `--heir-simd-vectorizer`

Run scheme-agnostic passes to convert FHE programs that operate on scalar types
to equivalent programs that operate on vectors.

This pass is intended to process FHE programs that are known to be good for
SIMD, but a specific FHE scheme has not yet been chosen. It expects to handle
`arith` ops operating on `tensor` types (with or without `secret.generic`).

The pass unrolls all loops, then applies a series of passes that convert scalar
operations on tensor elements to SIMD operations on full tensors. This uses the
FHE computational model common to BGV, BFV, and CKKS, in which data is packed in
polynomial ciphertexts, interpreted as vectors of individual data elements, and
arithmetic can be applied across entire ciphertexts, with some limited support
for rotations via automorphisms of the underlying ring.

Along the way, this pipeline applies heuristic optimizations to minimize the
number of rotations needed, relying on the implicit cost model that rotations
are generally expensive. The specific set of passes can be found in
`tools/heir-opt.cpp` where the pipeline is defined.

### `--heir-tosa-to-arith`

Lowers a TOSA MLIR model to `func`, `arith`, and `memref`.

Lowers from TOSA through `linalg` and `affine`, and converts all tensors to
memrefs. Fully unrolls all loops, and forwards stores to subsequent loads
whenever possible. The output is suitable as an input to
`heir-translate --emit-verilog`. Retains `affine.load` and `affine.store` ops
that cannot be removed (e.g., reading from the input and writing to the output,
or loading from a memref with a variable index).

The pass pipeline assumes that the input is a valid TOSA MLIR model with
stripped quantized types. The
[iree-import-tflite](https://iree.dev/guides/ml-frameworks/tflite) tool can
lower a TFLite FlatBuffer to textual MLIR with `--output-format=mlir-ir`. See
[hello_world.tosa.mlir](https://github.com/google/heir/blob/main/tests/verilog/hello_world.tosa.mlir)
for an example.

### `--yosys-optimizer`

Uses Yosys to booleanize and optimize MLIR functions.

This pass pipeline requires inputs to be in standard MLIR (`arith`, `affine`,
`func`, `memref`). The pass imports the model to Yosys and runs passes to
booleanize the circuit and then uses ABC to perform optimizations. We use
standard LUT 3 cells. THe output of this pass includes `arith` constants and
`comb.truth_table` ops.

The pass requires that the environment variable `HEIR_ABC_BINARY` contains the
location of the ABC binary and that `HEIR_YOSYS_SCRIPTS_DIR` contains the
location of the Yosys' techlib files that are needed to execute the path.

This pass can be disabled by defining `HEIR_NO_YOSYS`; this will avoid Yosys
library and ABC binary compilation, and avoid registration of this pass.

### `--tosa-to-boolean-tfhe`

This is an experimental pipeline for end-to-end private inference.

Converts a TOSA MLIR model to tfhe_rust dialect defined by HEIR. It converts a
tosa model to optimized boolean circuit using Yosys ABC optimizations. The
resultant optimized boolean circuit in comb dialect is then converted to cggi
and then to tfhe_rust exit dialect. This pipeline can be used with
heir-translate --emit-tfhe-rust to generate code for
[`tfhe-rs`](https://docs.zama.ai/tfhe-rs) FHE library.

The pass requires that the environment variable `HEIR_ABC_BINARY` contains the
location of the ABC binary and that `HEIR_YOSYS_SCRIPTS_DIR` contains the
location of the Yosys' techlib files that are needed to execute the path.

## `heir-translate`

### `--emit-tfhe-rust`

Code generation for the [`tfhe-rs`](https://docs.zama.ai/tfhe-rs) FHE library.
The library is based on the CGGI cryptosystem, and so this pass is most useful
when paired with lowerings from the `cggi` dialect.

The version of `tfhe-rs` supported is defined in the
[end to end `tfhe_rust` tests](https://github.com/google/heir/tree/main/tests/tfhe_rust/end_to_end/Cargo.toml).

Example input:

```mlir
!sks = !tfhe_rust.server_key
!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

func.func @test_apply_lookup_table(%sks : !sks, %lut: !lut, %input : !eui3) -> !eui3 {
  %v1 = tfhe_rust.apply_lookup_table %sks, %input, %lut : (!sks, !eui3, !lut) -> !eui3
  %v2 = tfhe_rust.add %sks, %input, %v1 : (!sks, !eui3, !eui3) -> !eui3
  %c1 = arith.constant 1 : i8
  %v3 = tfhe_rust.scalar_left_shift %sks, %v2, %c1 : (!sks, !eui3, i8) -> !eui3
  %v4 = tfhe_rust.apply_lookup_table %sks, %v3, %lut : (!sks, !eui3, !lut) -> !eui3
  return %v4 : !eui3
}
```

Example output:

```rust
use tfhe::shortint::prelude::*;

pub fn test_apply_lookup_table(
  v9: &ServerKey,
  v10: &LookupTableOwned,
  v11: &Ciphertext,
) -> Ciphertext {
  let v4 = v9.apply_lookup_table(&v11, &v10);
  let v5 = v9.unchecked_add(&v11, &v4);
  let v6 = 1;
  let v7 = v9.scalar_left_shift(&v5, v6);
  let v8 = v9.apply_lookup_table(&v7, &v10);
  v8
}
```

Note, the chosen variable names are arbitrary, and the resulting program still
must be integrated with a larger Rust program.

### `--emit-verilog`

Code generation for verilog from `arith` and `memref`. Expects a single top
level `func.func` op as the entry point, which is converted to the output
verilog module.

Example input:

```mlir
module {
  func.func @main(%arg0: i8) -> (i8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %0 = arith.extsi %arg0 : i8 to i32
    %1 = arith.subi %0, %c1 : i32
    %2 = arith.muli %1, %c2 : i32
    %3 = arith.addi %2, %c3 : i32
    %4 = arith.cmpi sge, %2, %c0 : i32
    %5 = arith.select %4, %c1, %c2 : i32
    %6 = arith.shrsi %3, %c1 : i32
    %7 = arith.shrui %3, %c1 : i32
    %out = arith.trunci %6 : i32 to i8
    return %out : i8
  }
}
```

Output:

```verilog
module main(
  input wire signed [7:0] arg1,
  output wire signed [7:0] _out_
);
  wire signed [31:0] v2;
  wire signed [31:0] v3;
  wire signed [31:0] v4;
  wire signed [31:0] v5;
  wire signed [31:0] v6;
  wire signed [31:0] v7;
  wire signed [31:0] v8;
  wire signed [31:0] v9;
  wire v10;
  wire signed [31:0] v11;
  wire signed [31:0] v12;
  wire signed [31:0] v13;
  wire signed [7:0] v14;

  assign v2 = 0;
  assign v3 = 1;
  assign v4 = 2;
  assign v5 = 3;
  assign v6 = {{24{arg1[7]}}, arg1};
  assign v7 = v6 - v3;
  assign v8 = v7 * v4;
  assign v9 = v8 + v5;
  assign v10 = v8 >= v2;
  assign v11 = v10 ? v3 : v4;
  assign v12 = v9 >>> v3;
  assign v13 = v9 >> v3;
  assign v14 = v12[7:0];
  assign _out_ = v14;
endmodule
```

### `--emit-metadata`

Prints a json object describing the function signatures. Used for code
generation after `--emit-verilog`.

Example input:

```mlir
module {
  func.func @main(%arg0: memref<80xi8>) -> memref<1x3x2x1xi8> {
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
    return %alloc_0 : memref<1x3x2x1xi8>
  }
}
```

Example output:

```json
{
  "functions": [
    {
      "name": "main",
      "params": [
        {
          "index": 0,
          "type": {
            "memref": {
              "element_type": {
                "integer": {
                  "is_signed": false,
                  "width": 8
                }
              },
              "shape": [80]
            }
          }
        }
      ],
      "return_types": [{
        "memref": {
          "element_type": {
            "integer": {
              "is_signed": false,
              "width": 8
            }
          },
          "shape": [1, 3, 2, 1]
        }
      }]
    }
  ]
}
```
