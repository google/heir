// Test emit-verilog supporting the minimal number of operations required to
// emit hello_world.tflite after it's lowered to arith via our other passes.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

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

// CHECK:      module main(
// CHECK-NEXT:   input wire signed [7:0] [[ARG:.*]],
// CHECK-NEXT:   output wire signed [7:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT:   wire signed [31:0] [[V2:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V3:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V4:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V5:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V6:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V7:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V8:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V9:.*]];
// CHECK-NEXT:   wire [[V10:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V11:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V12:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V13:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V14:.*]];
// CHECK-EMPTY:
// CHECK-NEXT:   assign [[V2]] = 0;
// CHECK-NEXT:   assign [[V3]] = 1;
// CHECK-NEXT:   assign [[V4]] = 2;
// CHECK-NEXT:   assign [[V5]] = 3;

// Double-braces means "regular expression" in FileCheck, so to match the
// leading double braces required for the sign extension syntax in verilog, we
// need this disgusting regular expression that matches two character classes
// [{] each consisting of a single opening brace: {{[{][{]}}
//
// CHECK-NEXT:   assign [[V6]] = {{[{][{]}}24{[[ARG]][7]}}, [[ARG]]};

// CHECK-NEXT:   assign [[V7]] = [[V6]] - [[V3]];
// CHECK-NEXT:   assign [[V8]] = [[V7]] * [[V4]];
// CHECK-NEXT:   assign [[V9]] = [[V8]] + [[V5]];
// CHECK-NEXT:   assign [[V10]] = [[V8]] >= [[V2]];
// CHECK-NEXT:   assign [[V11]] = [[V10]] ? [[V3]] : [[V4]];
// CHECK-NEXT:   assign [[V12]] = [[V9]] >>> [[V3]];
// CHECK-NEXT:   assign [[V13]] = [[V9]] >> [[V3]];
// CHECK-NEXT:   assign [[V14]] = [[V12]][7:0];
// CHECK-NEXT:   assign [[OUT]] = [[V14]];
// CHECK-NEXT: endmodule
