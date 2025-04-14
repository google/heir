// RUN: heir-translate --emit-verilog %s | FileCheck %s

module {
  memref.global "private" constant @__constant_256xi8 : memref<256xi8> = dense<2> {alignment = 64 : i64}
  // CHECK: module variable_index
  // CHECK-NEXT:   input wire signed [7:0] [[ARG:.*]],
  func.func @variable_index(%arg : i8) -> i8 {
    %c128 = arith.constant 128 : index
    // CHECK: assign [[V0:.*]] = 2048'h
    %0 = memref.get_global @__constant_256xi8 : memref<256xi8>
    // CHECK: assign [[V1:.*]] = $unsigned([[ARG]]);
    %1 = arith.index_cast %arg : i8 to index
    // CHECK: assign [[V2:.*]] = [[V1]] + 128;
    %2 = arith.addi %1, %c128 : index
    // CHECK: assign [[V3:.*]] = [[V0]][7 + 8 * [[V2]] : 8 * [[V2]]]
    %3 = memref.load %0[%2] : memref<256xi8>
    // CHECK: end
    func.return %3 : i8
  }
}
