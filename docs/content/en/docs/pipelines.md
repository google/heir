<!-- mdformat off(yaml frontmatter) -->
---
title: Pipelines
weight: 3
---
<!-- mdformat on -->

## `heir-opt`

### `--heir-tosa-to-arith`

Lowers a TOSA MLIR model to `func`, `arith`, and `memref`.

Lowers from TOSA through `linalg` and `affine`, and converts all tensors to
memrefs. Fully unrolls all loops, and forwards stores to subsequent loads
whenever possible. The output is suitable as an input to `heir-translate
--emit-verilog`. Retains `affine.load` and `affine.store` ops that cannot be
removed (e.g., reading from the input and writing to the output, or loading
from a memref with a variable index).

The pass pipeline assumes that the input is a valid TOSA MLIR model with
stripped quantized types. The
[iree-import-tflite](https://openxla.github.io/iree/getting-started/tflite/)
tool can lower a TFLite FlatBuffer to textual MLIR with
`--output-format=mlir-ir`. See
[hello_world.tosa.mlir](https://github.com/google/heir/blob/main/tests/hello_world.tosa.mlir)
for an example.

## `heir-translate`

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
