// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

// This tests whether capturing affine indices into the scope of the generic
// works when the affine read or write is contained within a region of an
// operation inside the secret generic (the affine.parallel loop).

// CHECK: module
module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_16xi32 : memref<16xi8> = dense<[0, 0, -54, -55, -13, -15, -41, -84, 33, 0, 19, -55, 0, -69, 34, -72]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> {alignment = 64 : i64}
  // CHECK: main
  func.func @main(%arg0: !secret.secret<memref<1x1xi8, strided<[?, ?], offset: ?>>> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (!secret.secret<memref<1x16xi8>> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    // The global values are optimized away.
    // CHECK-NOT: memref.get_global
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi8>
    // CHECK: secret.generic
    %2 = secret.generic ins(%arg0 : !secret.secret<memref<1x1xi8, strided<[?, ?], offset: ?>>>) {
    ^bb0(%arg1: memref<1x1xi8, strided<[?, ?], offset: ?>>):
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      affine.parallel (%arg2) = (0) to (16) {
        %4 = affine.load %0[%arg2, %c0] : memref<16x1xi8>
        %5 = affine.load %1[%arg2] : memref<16xi8>
        %11 = arith.muli %4, %5 : i8
        affine.store %11, %alloc[%c0, %arg2] : memref<1x16xi8>
      }
      secret.yield %alloc : memref<1x16xi8>
    } -> (!secret.secret<memref<1x16xi8>>)
    secret.generic ins(%2 : !secret.secret<memref<1x16xi8>>) {
    ^bb0(%arg1: memref<1x16xi8>):
      memref.dealloc %arg1 : memref<1x16xi8>
      secret.yield
    }
    // CHECK: return
    return %2 : !secret.secret<memref<1x16xi8>>
  }
}
