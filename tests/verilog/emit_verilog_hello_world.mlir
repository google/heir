// Test emit-verilog supporting the minimal number of operations required to
// emit hello_world.tflite after it's lowered to arith via our other passes.

// RUN: heir-translate %s --emit-verilog > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%arg0: ui8) -> (i8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %0 =  builtin.unrealized_conversion_cast %arg0 : ui8 to i8
    %1 = arith.extsi %0 : i8 to i32
    %2 = arith.subi %1, %c1 : i32
    %3 = arith.muli %2, %c2 : i32
    %4 = arith.addi %3, %c3 : i32
    %5 = arith.cmpi sge, %3, %c0 : i32
    %6 = arith.select %5, %c1, %c2 : i32
    %7 = arith.shrsi %4, %c1 : i32
    %8 = arith.shrui %4, %c1 : i32
    %9 = arith.extui %0 : i8 to i32
    %10 = arith.andi %8, %c1 : i32
    %11 = arith.maxsi %8, %7 : i32
    %out = arith.trunci %7 : i32 to i8
    return %out : i8
  }
}

// CHECK:      module main(
// CHECK-NEXT:   input wire [7:0] [[ARG:.*]],
// CHECK-NEXT:   output wire signed [7:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT:   wire signed [31:0] [[C0:.*]];
// CHECK-NEXT:   wire signed [31:0] [[C1:.*]];
// CHECK-NEXT:   wire signed [31:0] [[C2:.*]];
// CHECK-NEXT:   wire signed [31:0] [[C3:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V0:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V1:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V2:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V3:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V4:.*]];
// CHECK-NEXT:   wire [[V5:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V6:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V7:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V8:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V9:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V10:.*]];
// CHECK-NEXT:   wire signed [31:0] [[V11:.*]];
// CHECK-NEXT:   wire signed [7:0] [[_OUT:.*]];
// CHECK-EMPTY:
// CHECK-NEXT:   assign [[C0]] = 0;
// CHECK-NEXT:   assign [[C1]] = 1;
// CHECK-NEXT:   assign [[C2]] = 2;
// CHECK-NEXT:   assign [[C3]] = 3;

// Double-braces means "regular expression" in FileCheck, so to match the
// leading double braces required for the sign extension syntax in verilog, we
// need this disgusting regular expression that matches two character classes
// [{] each consisting of a single opening brace: {{[{][{]}}
//
// CHECK-NEXT:   assign [[V0]] = $signed([[ARG]]);
// CHECK-NEXT:   assign [[V1]] = {{[{][{]}}24{[[V0]][7]}}, [[V0]]};

// CHECK-NEXT:   assign [[V2]] = [[V1]] - [[C1]];
// CHECK-NEXT:   assign [[V3]] = [[V2]] * [[C2]];
// CHECK-NEXT:   assign [[V4]] = [[V3]] + [[C3]];
// CHECK-NEXT:   assign [[V5]] = [[V3]] >= [[C0]];
// CHECK-NEXT:   assign [[V6]] = [[V5]] ? [[C1]] : [[C2]];
// CHECK-NEXT:   assign [[V7]] = [[V4]] >>> [[C1]];
// CHECK-NEXT:   assign [[V8]] = [[V4]] >> [[C1]];
// CHECK-NEXT:   assign [[V9]] = {{[{][{]}}24{1'b0}}, [[V0]]};
// CHECK-NEXT:   assign [[V10]] = [[V8]] & [[C1]];
// CHECK-NEXT:   assign [[V11]] = [[V8]] > [[V7]] ? [[V8]] : [[V7]];
// CHECK-NEXT:   assign [[_OUT]] = [[V7]][7:0];
// CHECK-NEXT:   assign [[OUT]] = [[_OUT]];
// CHECK-NEXT: endmodule

module {
  func.func @test_unrealized_conversion_cast(%arg0: ui8) -> (ui8) {
    %c1 = arith.constant 1 : i8
    %0 = builtin.unrealized_conversion_cast %arg0 : ui8 to i8
    %1 = arith.addi %0, %c1 : i8
    %2 = builtin.unrealized_conversion_cast %1 : i8 to ui8
    return %2 : ui8
  }
}

// CHECK:      module test_unrealized_conversion_cast(
// CHECK-NEXT:   input wire [7:0] [[ARG:.*]],
// CHECK-NEXT:   output wire [7:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT:   wire signed [7:0] [[C1:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V0:.*]];
// CHECK-NEXT:   wire signed [7:0] [[V1:.*]];
// CHECK-NEXT:   wire [7:0] [[V2:.*]];
// CHECK-EMPTY:
// CHECK-NEXT:   assign [[C1]] = 1;
// CHECK-NEXT:   assign [[V0]] = $signed([[ARG]]);
// CHECK-NEXT:   assign [[V1]] = [[V0]] + [[C1]];
// CHECK-NEXT:   assign [[V2]] = $unsigned([[V1]]);
// CHECK-NEXT:   assign [[OUT]] = [[V2]];
// CHECK-NEXT: endmodule
