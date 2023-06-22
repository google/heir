// Test emit-verilog supporting translating memref allocations used as outputs
// into a wire of flattened bits.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%arg0: i8) -> memref<1x4xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4xi8>
    affine.store %arg0, %alloc[%c0, %c0] : memref<1x4xi8>
    %1 = arith.addi %arg0, %arg0 : i8
    affine.store %1, %alloc[%c0, %c1] : memref<1x4xi8>
    %2 = arith.addi %1, %arg0 : i8
    affine.store %2, %alloc[%c0, %c2] : memref<1x4xi8>
    %3 = arith.addi %2, %arg0 : i8
    affine.store %3, %alloc[%c0, %c3] : memref<1x4xi8>
    return %alloc : memref<1x4xi8>
  }
}

// CHECK:      module main
// CHECK-NEXT:   input wire signed [7:0] [[ARG:.*]],
// CHECK-NEXT:   output wire signed [31:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT: wire signed [31:0] [[V2:.*]];
// CHECK-NEXT: wire signed [7:0] [[V3:.*]];
// CHECK-NEXT: wire signed [7:0] [[V4:.*]];
// CHECK-NEXT: wire signed [7:0] [[V5:.*]];
// CHECK-EMPTY:
// CHECK-NEXT: assign [[V2]][7 : 0] = [[ARG]];
// CHECK-NEXT: assign [[V3]] = [[ARG]] + [[ARG]];
// CHECK-NEXT: assign [[V2]][15 : 8] = [[V3]];
// CHECK-NEXT: assign [[V4]] = [[V3]] + [[ARG]];
// CHECK-NEXT: assign [[V2]][23 : 16] = [[V4]];
// CHECK-NEXT: assign [[V5]] = [[V4]] + [[ARG]];
// CHECK-NEXT: assign [[V2]][31 : 24] = [[V5]];
// CHECK-NEXT: assign [[OUT]] = [[V2]];
// CHECK:      endmodule
