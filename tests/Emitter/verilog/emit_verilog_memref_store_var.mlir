// Test emit-verilog supporting translating symbolic affine loads into
// assignments into a wire of flattened bits.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%arg: i4) -> memref<4xi8> {
    %arg_index = arith.index_cast %arg : i4 to index
    %c0 = arith.constant 0 : i8
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi8>
    affine.store %c0, %alloc[%arg_index] : memref<4xi8>
    return %alloc : memref<4xi8>
  }
}

// CHECK:      module main
// CHECK-NEXT:   input wire signed [3:0] [[ARG:.*]],
// CHECK-NEXT:   output wire signed [31:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT: wire signed [31:0] [[V2:.*]];
// CHECK-NEXT: wire signed [7:0] [[V3:.*]];
// CHECK-NEXT: wire signed [31:0] [[V4:.*]];
// CHECK-EMPTY:
// CHECK-NEXT: assign [[V2]] = $unsigned([[ARG]]);
// CHECK-NEXT: assign [[V3]] = 0;
// CHECK:      assign [[V4]][7 + 8 * [[V2]] : 8 * [[V2]]] = [[V3]];
// CHECK-NEXT:  assign [[OUT]] = [[V4]];
// CHECK:      endmodule
