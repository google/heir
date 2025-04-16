// Test emit-verilog supports submodules.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

module {
  func.func @submod(%arg0: i8) -> (i8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %1 = arith.extsi %arg0 : i8 to i32
    %2 = arith.subi %1, %c1 : i32
    %3 = arith.muli %2, %c2 : i32
    %4 = arith.addi %3, %c3 : i32
    %out = arith.trunci %4 : i32 to i8
    return %out : i8
  }
  func.func @main(%arg0: ui8) -> (i8) {
    %0 =  builtin.unrealized_conversion_cast %arg0 : ui8 to i8
    %1 = call @submod(%0) : (i8) -> i8
    return %1 : i8
  }
}

// CHECK:      module submod(
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
// CHECK-NEXT:   wire signed [7:0] [[V10:.*]];
// CHECK-EMPTY:
// CHECK-NEXT:   assign [[V2]] = 0;
// CHECK-NEXT:   assign [[V3]] = 1;
// CHECK-NEXT:   assign [[V4]] = 2;
// CHECK-NEXT:   assign [[V5]] = 3;
// CHECK-NEXT:   assign [[V6]] = {{[{][{]}}24{[[ARG]][7]}}, [[ARG]]};
// CHECK-NEXT:   assign [[V7]] = [[V6]] - [[V3]];
// CHECK-NEXT:   assign [[V8]] = [[V7]] * [[V4]];
// CHECK-NEXT:   assign [[V9]] = [[V8]] + [[V5]];
// CHECK-NEXT:   assign [[V10]] = [[V9]][7:0];
// CHECK-NEXT:   assign [[OUT]] = [[V10]];
// CHECK-NEXT: endmodule

// CHECK:      module main(
// CHECK-NEXT:   input wire [7:0] [[ARG1:.*]],
// CHECK-NEXT:   output wire signed [7:0] [[OUT1:.*]]
// CHECK-NEXT: );
// CHECK-NEXT:   wire signed [7:0] [[V12:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V13:.*]];
// CHECK-EMPTY:
// CHECK-NEXT:   assign [[V12]] = $signed([[ARG1]]);
// CHECK-NEXT:   submod [[V13]]_call([[V12]], [[V13]], );
// CHECK-NEXT:   assign [[OUT1]] = [[V13]];
// CHECK-NEXT: endmodule
