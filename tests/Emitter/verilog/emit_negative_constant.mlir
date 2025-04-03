// RUN: heir-translate --emit-verilog %s | FileCheck %s

module {
  memref.global "private" constant @__constant : memref<2xi8> = dense<[-1, -2]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_i32 : memref<1xi32> = dense<-7217> {alignment = 64 : i64}
  // CHECK: module negative_constant(
  func.func @negative_constant(%in: i8) -> (i32) {
    // CHECK: assign [[V0:.*]] = 16'hFEFF;
    // CHECK: assign [[V1:.*]] = 32'hFFFFE3CF;
    %cst = memref.get_global @__constant : memref<2xi8>
    %cst_1 = memref.get_global @__constant_i32 : memref<1xi32>

    %c0 = arith.constant 0 : index
    %1 = affine.load %cst[0] : memref<2xi8>
    %2 = arith.addi %in, %1 : i8
    %3 = arith.extsi %2 : i8 to i32
    %5 = affine.load %cst_1[0] : memref<1xi32>
    %4 = arith.addi %3, %5 : i32
    return %4 : i32
  }
}
