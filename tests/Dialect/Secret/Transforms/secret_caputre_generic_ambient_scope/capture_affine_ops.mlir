// RUN: heir-opt --secret-capture-generic-ambient-scope %s | FileCheck %s

module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
// CHECK: main
  func.func @main(%arg0: !secret.secret<memref<1x1xi8>> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (!secret.secret<memref<1x16xi8>> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    // CHECK: %[[c0:.*]] = arith.constant
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    // CHECK: %[[mem:.*]] = secret.generic
    %5 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      secret.yield %alloc : memref<1x16xi8>
    } -> !secret.secret<memref<1x16xi8>>
    // CHECK: affine.for %[[i:.*]] = 0 to 1
    affine.for %arg1 = 0 to 1 {
      // CHECK-NEXT: affine.for %[[j:.*]] = 0 to 16
      affine.for %arg2 = 0 to 16 {
        // CHECK-NEXT: %[[val:.*]] = affine.load
        %20 = affine.load %0[%arg2, %arg1] : memref<16x1xi8>
        // CHECK-NEXT: secret.generic ins(%[[mem]], %[[val]], %[[i]], %[[j]] : !secret.secret<memref<1x16xi8>>, i8, index, index)
        secret.generic ins(%5 : !secret.secret<memref<1x16xi8>>) {
        ^bb0(%arg3: memref<1x16xi8>):
          affine.store %20, %arg3[%arg1, %arg2] : memref<1x16xi8>
          secret.yield
        }
      }
    }
    // CHECK: %[[mem1:.*]] = secret.generic
    %7 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      secret.yield %alloc : memref<1x16xi8>
    } -> !secret.secret<memref<1x16xi8>>
    // CHECK: affine.for %[[j:.*]] = 0 to 16
    affine.for %arg1 = 0 to 16 {
      // CHECK-NEXT:  %[[val1:.*]] = secret.generic
      %20 = secret.generic ins(%5 : !secret.secret<memref<1x16xi8>>) {
      ^bb0(%arg2: memref<1x16xi8>):
        %21 = affine.load %arg2[0, %arg1] : memref<1x16xi8>
        secret.yield %21 : i8
      } -> !secret.secret<i8>
        // CHECK: secret.generic ins(%[[mem1]], %[[val1]], %[[j]], %[[c0]] : !secret.secret<memref<1x16xi8>>, !secret.secret<i8>, index, index)
      secret.generic ins(%7, %20 : !secret.secret<memref<1x16xi8>>, !secret.secret<i8>) {
      ^bb0(%arg2: memref<1x16xi8>, %arg3: i8):
        affine.store %arg3, %arg2[0, %arg1] : memref<1x16xi8>
        secret.yield
      }
    }
    return %7 : !secret.secret<memref<1x16xi8>>
  }
}
