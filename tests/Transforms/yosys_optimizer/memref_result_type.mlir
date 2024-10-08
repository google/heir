// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// This test ensures that the memref allocation occurring within the generic op
// has a shape extending the result shape. In this case the allocation is
// memref<1x1x8xi1> to match the result memref<1x1xi8>, instead of the flattened
// wire representation memref<8xi1>.

module attributes {tf_saved_model.semantics} {
  // CHECK: @main(%[[arg0:.*]]: [[SECRETI8:!secret.secret<i8>]]) -> [[SECRET1x1xi8:!secret.secret<memref<1x1xi8>>]] {
  // CHECK-NEXT: secret.cast %[[arg0]] : [[SECRETI8]] to [[SECRET8xi1:!secret.secret<memref<8xi1>>]]
  // CHECK-NEXT: secret.generic ins(%[[arg1:.*]] : [[SECRET8xi1]])
  // CHECK:         memref.alloc() : [[MEMREF1x1x8xi1:memref<1x1x8xi1>]]
  // CHECK:         %[[collapse:.*]] = memref.collapse_shape
  // CHECK-SAME:      into [[MEMREF8xi1:memref<8xi1>]]
  // CHECK:         secret.yield %[[collapse]] : [[MEMREF8xi1]]
  // CHECK:      secret.cast
  // CHECK-SAME:   [[SECRET8xi1]] to [[SECRET1x1xi8]]
  func.func @main(%arg0: !secret.secret<i8>) -> !secret.secret<memref<1x1xi8>> {
    %c22 = arith.constant 22 : i8
    %0 = secret.generic ins(%arg0 : !secret.secret<i8>) {
    ^bb0(%arg1: i8):
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
      %1 = arith.addi %arg1, %c22 : i8
      affine.store %1, %alloc[0, 0] : memref<1x1xi8>
      secret.yield %alloc : memref<1x1xi8>
    } -> !secret.secret<memref<1x1xi8>>
    return %0 : !secret.secret<memref<1x1xi8>>
  }
}
