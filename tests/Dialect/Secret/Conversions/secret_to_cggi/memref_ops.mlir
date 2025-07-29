// RUN: heir-opt --secret-distribute-generic --secret-to-cggi %s | FileCheck %s

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext

// CHECK: module
module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
  // CHECK: @memref_ops
  // CHECK-SAME: ([[ARG:%.*]]: memref<1x1x8x![[ct_ty]]>) -> memref<1x16x32x![[ct_ty]]>
  func.func @memref_ops(%arg0: !secret.secret<memref<1x1xi8>>) -> !secret.secret<memref<1x16xi32>> {
    %c0 = arith.constant 0 : index
    %c-128_i16 = arith.constant -128 : i16
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    // CHECK: [[V2:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x16x8x![[ct_ty]]>
    %2 = secret.generic() {
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
        secret.generic(%2: !secret.secret<memref<1x16xi8>>, %5: i8, %arg1: index, %arg2: index) {
        ^bb0(%arg3: memref<1x16xi8>, %arg4: i8, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x16xi8>
          secret.yield
        }
      }
    }
    // CHECK: [[V3:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x16x32x![[ct_ty]]>
    %3 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %5 = affine.load %1[%arg2] : memref<16xi32>
        secret.generic(%3: !secret.secret<memref<1x16xi32>>, %5: i32, %arg1: index, %arg2: index) {
        ^bb0(%arg3: memref<1x16xi32>, %arg4: i32, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x16xi32>
          secret.yield
        }
      }
    }
    // CHECK: [[V4:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x16x32x![[ct_ty]]>
    %4 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 16 {
      // [[SUB:%.*]] = memref.subview [[V3]]
      // [[ALLOC2:%.*]] = memref.alloc()
      // memref.copy [[SUB]], [[ALLOC2]]
      %5 = secret.generic(%3: !secret.secret<memref<1x16xi32>>, %arg1: index, %c0: index) {
      ^bb0(%arg2: memref<1x16xi32>, %arg3: index, %arg4: index):
        %6 = memref.load %arg2[%arg4, %arg3] : memref<1x16xi32>
        secret.yield %6 : i32
      } -> !secret.secret<i32>
      // memref.subview [[V4]]
      // memref.copy [[ALLOC2]], [[V4]]
      secret.generic(%4: !secret.secret<memref<1x16xi32>>, %5: !secret.secret<i32>, %arg1: index, %c0: index) {
      ^bb0(%arg2: memref<1x16xi32>, %arg3: i32, %arg4: index, %arg5: index):
        memref.store %arg3, %arg2[%arg5, %arg4] : memref<1x16xi32>
        secret.yield
      }
    }
    // CHECK: return [[V4]] : memref<1x16x32x![[ct_ty]]>
    return %4 : !secret.secret<memref<1x16xi32>>
  }

  // CHECK: @affine_ops
  // CHECK-SAME: ([[ARG:%.*]]: memref<1x1x32x![[ct_ty]]>) -> memref<1x1x32x![[ct_ty]]>
  func.func @affine_ops(%arg0: !secret.secret<memref<1x1xi32>>) -> !secret.secret<memref<1x1xi32>> {
    // CHECK: [[V6:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x1x32x![[ct_ty]]>
    %6 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      secret.yield %alloc : memref<1x1xi32>
    } -> !secret.secret<memref<1x1xi32>>
    // CHECK: [[SUB:%.*]] = memref.subview [[ARG]][0, 0, 0] [1, 1, 32] [1, 1, 1] : memref<1x1x32x![[ct_ty]]>
    // CHECK: [[ALLOC:%.*]] = memref.alloc() : memref<32x![[ct_ty]]
    // CHECK: memref.copy [[SUB]], [[ALLOC]]
    %7 = secret.generic(%arg0 : !secret.secret<memref<1x1xi32>>) {
    ^bb0(%arg1: memref<1x1xi32>):
      %20 = affine.load %arg1[0, 0] : memref<1x1xi32>
      secret.yield %20 : i32
    } -> !secret.secret<i32>
    // CHECK: [[SUB1:%.*]] = memref.subview [[V6]][0, 0, 0] [1, 1, 32] [1, 1, 1] : memref<1x1x32x![[ct_ty]]> to memref<32x![[ct_ty]], strided<[1]>>
    // CHECK: memref.copy [[ALLOC]], [[SUB1]] : memref<32x![[ct_ty]]> to memref<32x![[ct_ty]], strided<[1]>>
    secret.generic(%6: !secret.secret<memref<1x1xi32>>, %7: !secret.secret<i32>) {
    ^bb0(%arg1: memref<1x1xi32>, %arg2: i32):
      affine.store %arg2, %arg1[0, 0] : memref<1x1xi32>
      secret.yield
    }
    // CHECK: return [[V6]] : memref<1x1x32x![[ct_ty]]>
    return %6 : !secret.secret<memref<1x1xi32>>
  }

  // CHECK: @single_bit_memref
  // CHECK-SAME: ([[ARG:%.*]]: memref<1x![[ct_ty]]>) -> memref<1x![[ct_ty]]>
  func.func @single_bit_memref(%arg0: !secret.secret<memref<1xi1>>) -> !secret.secret<memref<1xi1>> {
    // CHECK: [[c0:%.*]] = arith.constant 0
    // CHECK: [[V6:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x![[ct_ty]]>
    %6 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xi1>
      secret.yield %alloc : memref<1xi1>
    } -> !secret.secret<memref<1xi1>>
    // CHECK: [[V0:%.*]] = affine.load [[ARG]][0] : memref<1x![[ct_ty]]>
    %7 = secret.generic(%arg0 : !secret.secret<memref<1xi1>>) {
    ^bb0(%arg1: memref<1xi1>):
      %20 = affine.load %arg1[0] : memref<1xi1>
      secret.yield %20 : i1
    } -> !secret.secret<i1>
    // CHECK: memref.store [[V0]], [[V6]][[[c0]]] : memref<1x![[ct_ty]]>
    secret.generic(%6: !secret.secret<memref<1xi1>>, %7: !secret.secret<i1>) {
    ^bb0(%arg1: memref<1xi1>, %arg2: i1):
      affine.store %arg2, %arg1[0] : memref<1xi1>
      secret.yield
    }
    // CHECK: return [[V6]] : memref<1x![[ct_ty]]>
    return %6 : !secret.secret<memref<1xi1>>
  }

  // CHECK: @single_bit_plaintext_memref
  // CHECK-SAME: () -> memref<1x![[ct_ty]]>
  func.func @single_bit_plaintext_memref() -> !secret.secret<memref<1xi1>> {
    // CHECK: [[c0:%.*]] = arith.constant 0 : index
    // CHECK: [[TRUE:%.*]] = arith.constant true
    %true = arith.constant true
    // CHECK: [[V6:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x![[ct_ty]]>
    %6 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xi1>
      secret.yield %alloc : memref<1xi1>
    } -> !secret.secret<memref<1xi1>>
    // CHECK: [[ENC:%.*]] = lwe.encode [[TRUE]]
    // CHECK: [[LWE:%.*]] = lwe.trivial_encrypt [[ENC]]
    // CHECK: memref.store [[LWE]], [[V6]][[[c0]]] : memref<1x![[ct_ty]]>
    secret.generic(%6 : !secret.secret<memref<1xi1>>) {
    ^bb0(%arg1: memref<1xi1>):
      affine.store %true, %arg1[0] : memref<1xi1>
      secret.yield
    }
    // CHECK: return [[V6]] : memref<1x![[ct_ty]]>
    return %6 : !secret.secret<memref<1xi1>>
  }

  // CHECK: @collapse_shape
  // CHECK-SAME: (%[[arg0:.*]]: memref<1x1x![[ct_ty]]>)
  func.func @collapse_shape(%arg0: !secret.secret<memref<1x1xi1>>) -> !secret.secret<memref<1xi1>> {
    // CHECK: %[[v0:.*]] = memref.collapse_shape %[[arg0]]
    %6 = secret.generic(%arg0 : !secret.secret<memref<1x1xi1>>) {
    ^bb0(%arg1: memref<1x1xi1>):
      %alloc = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1xi1> into memref<1xi1>
      secret.yield %alloc : memref<1xi1>
    } -> !secret.secret<memref<1xi1>>
    return %6 : !secret.secret<memref<1xi1>>
  }
}
