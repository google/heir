// RUN: heir-translate --emit-verilog %s 2>&1 > %t1
// RUN: FileCheck %s < %t1

// The following checks for correctness, ensuring that the statements are
// synthesized correctly.
// RUN: heir-translate --emit-verilog %s 2>&1 > %t1
// RUN: run_verilog \
// RUN:  --verilog_module %t1 \
// RUN:  > %t
// RUN: FileCheck %s --check-prefix=VAL < %t

// VAL: b'\x80\x80\x80\x80\x80\x80'

module {
  func.func @main() -> memref<1x3x2x1xi8> {
    %c-128_i8 = arith.constant -128 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
    affine.parallel (%arg1) = (0) to (3) {
      affine.parallel (%arg2) = (0) to (2) {
        affine.store %c-128_i8, %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
      }
    }
    return %alloc : memref<1x3x2x1xi8>
  }
}

// CHECK:      module main
// CHECK-NEXT:   output wire signed [47:0] [[OUT:.*]]
// CHECK-NEXT: );
// CHECK-NEXT: wire signed [7:0] [[V1:.*]];
// CHECK-NEXT: wire signed [47:0] [[V2:.*]];
// CHECK-EMPTY:
// CHECK-NEXT: assign [[V1]] = -128;
// CHECK-NEXT: genvar [[ARG1:.*]];
// CHECK-NEXT: generate
// CHECK-NEXT: for ([[ARG1]] = 0; [[ARG1]] < 3; [[ARG1]] = [[ARG1]] + 1) begin
// CHECK-NEXT:   genvar [[ARG2:.*]];
// CHECK-NEXT:   generate
// CHECK-NEXT:   for ([[ARG2]] = 0; [[ARG2]] < 2; [[ARG2]] = [[ARG2]] + 1) begin
// CHECK-NEXT:      assign [[V2]]
// CHECK-SAME:        = [[V1]]
// CHECK-NEXT:   end
// CHECK-NEXT:   endgenerate
// CHECK-NEXT: end
// CHECK-NEXT: endgenerate
// CHECK-NEXT: assign [[OUT]] = [[V2]];
// CHECK:      endmodule
