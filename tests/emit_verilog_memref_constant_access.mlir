// Test emit-verilog supports constant indexing into a multi-dimensional memref.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

// CHECK:      module main
module {
// CHECK-NEXT:   input wire signed [47:0] [[ARG:.*]],
// CHECK-NEXT:   output wire signed [7:0] [[OUT:.*]]
// CHECK-NEXT: );
  func.func @main(%marg: memref<2x3xi8>) -> (i8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c1_index = arith.constant 1 : index
// CHECK: assign v{{[0-9]+}} = [[ARG]][39 : 32];
    %arg0 = affine.load %marg[%c1_index, %c1_index] : memref<2x3xi8>
    %0 = arith.extsi %arg0 : i8 to i32
    %1 = arith.subi %0, %c1 : i32
    %2 = arith.muli %1, %c2 : i32
    %3 = arith.addi %2, %c3 : i32
    %4 = arith.cmpi sge, %2, %c0 : i32
    %5 = arith.select %4, %c1, %c2 : i32
    %6 = arith.shrsi %3, %c1 : i32
    %7 = arith.shrui %3, %c1 : i32
    %out = arith.trunci %6 : i32 to i8
// CHECK:  assign [[OUT]]
    return %out : i8
  }
// CHECK:      endmodule
}
