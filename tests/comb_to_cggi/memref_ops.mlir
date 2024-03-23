// RUN: heir-opt --secret-distribute-generic --comb-to-cggi %s | FileCheck %s

// CHECK: module
module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
  // CHECK: @memref_ops([[ARG:%.*]]: [[LWET:memref<1x1x8x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:memref<1x16x32x!lwe.lwe_ciphertext<.*>>]]
  func.func @memref_ops(%arg0: !secret.secret<memref<1x1xi8>>) -> !secret.secret<memref<1x16xi32>> {
    %c0 = arith.constant 0 : index
    %c-128_i16 = arith.constant -128 : i16
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    // CHECK: [[V2:%.*]] = memref.alloc() {alignment = 64 : i64} : [[V2T:memref<1x16x8x!lwe.lwe_ciphertext<.*>>]]
    %2 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      secret.yield %alloc : memref<1x16xi8>
    } -> !secret.secret<memref<1x16xi8>>
    // CHECK: affine.for
    affine.for %arg1 = 0 to 1 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 16 {
        // CHECK: [[V3:%.*]] = affine.load
        %5 = affine.load %0[%arg2, %arg1] : memref<16x1xi8>
        // CHECK: affine.for
        // CHECK: [[V9:%.*]] = lwe.encode
        // CHECK: [[V10:%.*]] = lwe.trivial_encrypt
        // CHECK: memref.store [[V10]], [[V2]]
        secret.generic ins(%2, %5, %arg1, %arg2 : !secret.secret<memref<1x16xi8>>, i8, index, index) {
        ^bb0(%arg3: memref<1x16xi8>, %arg4: i8, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x16xi8>
          secret.yield
        }
      }
    }
    // CHECK: [[V3:%.*]] = memref.alloc() {alignment = 64 : i64} : [[V3T:memref<1x16x32x!lwe.lwe_ciphertext<.*>>]]
    %3 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %5 = affine.load %1[%arg2] : memref<16xi32>
        secret.generic ins(%3, %5, %arg1, %arg2 : !secret.secret<memref<1x16xi32>>, i32, index, index) {
        ^bb0(%arg3: memref<1x16xi32>, %arg4: i32, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x16xi32>
          secret.yield
        }
      }
    }
    // CHECK: [[V4:%.*]] = memref.alloc() {alignment = 64 : i64} : [[OUT]]
    %4 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 16 {
      // [[SUB:%.*]] = memref.subview [[V3]]
      // [[ALLOC2:%.*]] = memref.alloc()
      // memref.copy [[SUB]], [[ALLOC2]]
      %5 = secret.generic ins(%3, %arg1, %c0 : !secret.secret<memref<1x16xi32>>, index, index) {
      ^bb0(%arg2: memref<1x16xi32>, %arg3: index, %arg4: index):
        %6 = memref.load %arg2[%arg4, %arg3] : memref<1x16xi32>
        secret.yield %6 : i32
      } -> !secret.secret<i32>
      // memref.subview [[V4]]
      // memref.copy [[ALLOC2]], [[V4]]
      secret.generic ins(%4, %5, %arg1, %c0 : !secret.secret<memref<1x16xi32>>, !secret.secret<i32>, index, index) {
      ^bb0(%arg2: memref<1x16xi32>, %arg3: i32, %arg4: index, %arg5: index):
        memref.store %arg3, %arg2[%arg5, %arg4] : memref<1x16xi32>
        secret.yield
      }
    }
    // CHECK: return [[V4]] : [[OUT]]
    return %4 : !secret.secret<memref<1x16xi32>>
  }

  // CHECK: @affine_ops([[ARG:%.*]]: [[LWET:memref<1x1x32x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:memref<1x1x32x!lwe.lwe_ciphertext<.*>>]]
  func.func @affine_ops(%arg0: !secret.secret<memref<1x1xi32>>) -> !secret.secret<memref<1x1xi32>> {
    // CHECK: [[V6:%.*]] = memref.alloc() {alignment = 64 : i64} : [[OUT]]
    %6 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      secret.yield %alloc : memref<1x1xi32>
    } -> !secret.secret<memref<1x1xi32>>
    // CHECK: [[SUB:%.*]] = memref.subview [[ARG]][0, 0, 0] [1, 1, 32] [1, 1, 1] : [[OUT]]
    // CHECK: [[ALLOC:%.*]] = memref.alloc() : [[I32T:memref<32x!lwe.lwe_ciphertext<.*>>]]
    // CHECK: memref.copy [[SUB]], [[ALLOC]]
    %7 = secret.generic ins(%arg0 : !secret.secret<memref<1x1xi32>>) {
    ^bb0(%arg1: memref<1x1xi32>):
      %20 = affine.load %arg1[0, 0] : memref<1x1xi32>
      secret.yield %20 : i32
    } -> !secret.secret<i32>
    // CHECK: [[SUB1:%.*]] = memref.subview [[V6]][0, 0, 0] [1, 1, 32] [1, 1, 1] : [[OUT]] to [[I32T1:memref<32x.*>]]
    // CHECK: memref.copy [[ALLOC]], [[SUB1]] : [[I32T]] to [[I32T1]]
    secret.generic ins(%6, %7 : !secret.secret<memref<1x1xi32>>, !secret.secret<i32>) {
    ^bb0(%arg1: memref<1x1xi32>, %arg2: i32):
      affine.store %arg2, %arg1[0, 0] : memref<1x1xi32>
      secret.yield
    }
    // CHECK: return [[V6]] : [[OUT]]
    return %6 : !secret.secret<memref<1x1xi32>>
  }

  // CHECK: @single_bit_memref([[ARG:%.*]]: [[LWET:memref<1x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:memref<1x!lwe.lwe_ciphertext<.*>>]]
  func.func @single_bit_memref(%arg0: !secret.secret<memref<1xi1>>) -> !secret.secret<memref<1xi1>> {
    // CHECK: [[V6:%.*]] = memref.alloc() {alignment = 64 : i64} : [[OUT]]
    %6 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xi1>
      secret.yield %alloc : memref<1xi1>
    } -> !secret.secret<memref<1xi1>>
    // CHECK: [[V0:%.*]] = affine.load [[ARG]][0] : [[LWET]]
    %7 = secret.generic ins(%arg0 : !secret.secret<memref<1xi1>>) {
    ^bb0(%arg1: memref<1xi1>):
      %20 = affine.load %arg1[0] : memref<1xi1>
      secret.yield %20 : i1
    } -> !secret.secret<i1>
    // CHECK: affine.store [[V0]], [[V6]][0] : [[OUT]]
    secret.generic ins(%6, %7 : !secret.secret<memref<1xi1>>, !secret.secret<i1>) {
    ^bb0(%arg1: memref<1xi1>, %arg2: i1):
      affine.store %arg2, %arg1[0] : memref<1xi1>
      secret.yield
    }
    // CHECK: return [[V6]] : [[OUT]]
    return %6 : !secret.secret<memref<1xi1>>
  }
}
