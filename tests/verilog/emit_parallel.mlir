// RUN: heir-translate --emit-verilog %s 2>&1 > %t1
// RUN: FileCheck %s < %t1

// The following checks for correctness, ensuring that the statements are
// synthesized correctly. As input, we need to concatenate two 16-bit integers.
//   458755 is 0000000000000111 0000000000000011 concatenated
//   The test truncates both integers and so we expect
//   00000111 00000011 which is x07 x03 in hex
// RUN: heir-translate --emit-verilog %s 2>&1 > %t1
// RUN: run_verilog \
// RUN:  --verilog_module %t1 \
// RUN:  --input='arg1=458755' \
// RUN:  > %t
// RUN: FileCheck %s --check-prefix=VAL < %t

// VAL: b'\x07\x03'

module {
  func.func @main(%arg0 : memref<1x2xi16>) -> memref<1x2xi8> {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2xi8>
    affine.parallel (%arg1) = (0) to (2) {
      %1 = affine.load %arg0[%c0, %arg1] : memref<1x2xi16>
      %2 = arith.trunci %1 : i16 to i8
      affine.store %2, %alloc[%c0, %arg1] : memref<1x2xi8>
    }
    return %alloc : memref<1x2xi8>
  }
}

// CHECK:      module main
// CHECK-NEXT:   input wire signed [31:0] [[IN:.*]],
// CHECK-NEXT:   output wire signed [15:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT: wire signed [15:0] [[V2:.*]];
// CHECK-NEXT: wire signed [15:0] [[V3:.*]];
// CHECK-NEXT: wire signed [7:0] [[V4:.*]];
// CHECK-EMPTY:
// CHECK-NEXT: genvar [[ARG1:.*]];
// CHECK-NEXT: generate
// CHECK: for ([[ARG1]] = 0; [[ARG1]] < 2; [[ARG1]] = [[ARG1]] + 1) begin
// CHECK-NEXT:   wire signed [15:0] [[V3:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V4:.*]];
// CHECK-NEXT:   assign [[V3]] = [[IN]]
// CHECK-NEXT:   assign [[V4]] = [[V3]][7:0];
// CHECK-NEXT:   assign [[V2]]
// CHECK-SAME:        = [[V4]]
// CHECK-NEXT: end
// CHECK-NEXT: endgenerate
// CHECK-NEXT: assign [[OUT]] = [[V2]];
// CHECK:      endmodule
