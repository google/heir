// Test emit-verilog supporting the minimal number of operations required to
// lower the hello_world.tflite model

// RUN: heir-translate %s --emit-verilog | FileCheck %s

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
    %7 = arith.trunci %6 : i32 to i8
    return %7 : i8
  }
}

// CHECK:      module main(
// CHECK-NEXT:   input wire [7:0] [[ARG:.*]],
// CHECK-NEXT:   output wire [7:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT:   wire [31:0] [[V2:.*]];
// CHECK-NEXT:   wire [31:0] [[V3:.*]];
// CHECK-NEXT:   wire [31:0] [[V4:.*]];
// CHECK-NEXT:   wire [31:0] [[V5:.*]];
// CHECK-NEXT:   wire [31:0] [[V6:.*]];
// CHECK-NEXT:   wire [31:0] [[V7:.*]];
// CHECK-NEXT:   wire [31:0] [[V8:.*]];
// CHECK-NEXT:   wire [31:0] [[V9:.*]];
// CHECK-NEXT:   wire [[V10:.*]];
// CHECK-NEXT:   wire [31:0] [[V11:.*]];
// CHECK-NEXT:   wire [31:0] [[V12:.*]];
// CHECK-NEXT:   wire [7:0] [[V13:.*]];
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: endmodule
