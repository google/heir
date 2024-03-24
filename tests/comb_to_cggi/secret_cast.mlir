// This test ensures that secret casting before and after generics lowers to CGGI properly.

// RUN: heir-opt --secret-distribute-generic --comb-to-cggi -cse %s | FileCheck %s

// CHECK: module
module attributes {tf_saved_model.semantics} {
  // CHECK: @main([[ARG:%.*]]: [[LWET:memref<1x1x8x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:memref<1x1x4x!lwe.lwe_ciphertext<.*>>]]
  func.func @main(%arg0: !secret.secret<memref<1x1xi8>>) -> !secret.secret<memref<1x1xi4>> {
    // CHECK: [[V0:%.*]] = memref.alloc() {alignment = 64 : i64} : [[OUT]]
    %0 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi4>
      secret.yield %alloc : memref<1x1xi4>
    } -> !secret.secret<memref<1x1xi4>>
    // CHECK: affine.for
    affine.for %arg1 = 0 to 1 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 1 {
        // CHECK: [[SUBVIEW:%.*]] = memref.subview [[ARG]]
        // CHECK:   to memref<8x!lwe.lwe_ciphertext
        // CHECK: [[ALLOC:%.*]] = memref.alloc()
        // CHECK: memref.copy [[SUBVIEW]], [[ALLOC]]
        %1 = secret.generic ins(%arg0, %arg1, %arg2 : !secret.secret<memref<1x1xi8>>, index, index) {
        ^bb0(%arg3: memref<1x1xi8>, %arg4: index, %arg5: index):
          %5 = memref.load %arg3[%arg4, %arg5] : memref<1x1xi8>
          secret.yield %5 : i8
        } -> !secret.secret<i8>
        %2 = secret.cast %1 : !secret.secret<i8> to !secret.secret<memref<8xi1>>
        // CHECK-COUNT-4: memref.load [[ALLOC]]
        // CHECK-COUNT-4: memref.store
        %3 = secret.generic ins(%2 : !secret.secret<memref<8xi1>>) {
        ^bb0(%arg3: memref<8xi1>):
          %c0 = arith.constant 0 : index
          %5 = memref.load %arg3[%c0] : memref<8xi1>
          %c1 = arith.constant 1 : index
          %6 = memref.load %arg3[%c1] : memref<8xi1>
          %c2 = arith.constant 2 : index
          %7 = memref.load %arg3[%c2] : memref<8xi1>
          %c3 = arith.constant 3 : index
          %8 = memref.load %arg3[%c3] : memref<8xi1>
          %alloc = memref.alloc() : memref<4xi1>
          memref.store %8, %alloc[%c0] : memref<4xi1>
          memref.store %7, %alloc[%c1] : memref<4xi1>
          memref.store %6, %alloc[%c2] : memref<4xi1>
          memref.store %5, %alloc[%c3] : memref<4xi1>
          secret.yield %alloc : memref<4xi1>
        } -> !secret.secret<memref<4xi1>>
        %4 = secret.cast %3 : !secret.secret<memref<4xi1>> to !secret.secret<i4>
        // memref.subview
        // memref.copy
        secret.generic ins(%0, %4, %arg1, %arg2 : !secret.secret<memref<1x1xi4>>, !secret.secret<i4>, index, index) {
        ^bb0(%arg3: memref<1x1xi4>, %arg4: i4, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x1xi4>
          secret.yield
        }
      }
    }
    // CHECK: return [[V0]] : [[OUT]]
    return %0 : !secret.secret<memref<1x1xi4>>
  }
}
