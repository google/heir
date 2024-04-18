// RUN: heir-translate --emit-verilog %s | FileCheck %s

module {
  memref.global "private" constant @__constant : memref<1xi8> = dense<-1> {alignment = 64 : i64}
  // CHECK-LABEL: module negative_constant(
  func.func @negative_constant(%in: i8) -> (i8) {
    // CHECK: assign [[V0:.*]] = 8'hFF;
    %cst = memref.get_global @__constant : memref<1xi8>
    %c0 = arith.constant 0 : index
    %1 = affine.load %cst[0] : memref<1xi8>
    %2 = arith.addi %in, %1 : i8
    return %2 : i8
  }
}
