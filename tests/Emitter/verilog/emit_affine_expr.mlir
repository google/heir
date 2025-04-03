// RUN: heir-translate --emit-verilog %s | FileCheck %s

module {
  memref.global "private" constant @__constant_2x2xi8 : memref<2x2xi8> = dense<"0xF41AED09"> {alignment = 64 : i64}
  // CHECK: module multidimensional
  func.func @multidimensional() -> memref<1x2xi8> {
    // CHECK: assign [[V0:.*]] = 32'h09ED1AF4;
    %0 = memref.get_global @__constant_2x2xi8 : memref<2x2xi8>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2xi8>
    // CHECK: for ([[ARG:.*]] = 0; [[ARG]] < 1; [[ARG]] = [[ARG]] + 1)
    affine.parallel (%arg2) = (0) to (1) {
      // CHECK: assign [[V1:.*]] = [[V0]][7 + 8 * (1 + 2 * ([[ARG]])) : 8 * (1 + 2 * ([[ARG]]))];
      %9 = affine.load %0[%arg2, 1] : memref<2x2xi8>
      // CHECK: assign [[ALLOC:.*]][7 + 8 * (1 + 2 * ([[ARG]])) : 8 * (1 + 2 * ([[ARG]]))] = [[V1]];
      affine.store %9, %alloc[%arg2, 1] : memref<1x2xi8>
    }
    // CHECK: end
    func.return %alloc : memref<1x2xi8>
  }
}
