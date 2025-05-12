// RUN: heir-opt --secret-distribute-generic --canonicalize --secret-to-cggi --split-input-file %s | FileCheck %s

// CHECK: @tensor_extract
// CHECK-SAME: ([[ARG:%.*]]: [[LWET:tensor<1x1x32x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:tensor<1x1x32x!lwe.lwe_ciphertext<.*>>]]
func.func @tensor_extract(%arg0: !secret.secret<tensor<1x1xi32>>) -> !secret.secret<tensor<1x1xi32>> {
  %c0 = arith.constant 0 : index
  // CHECK: [[V7:%.*]] = tensor.extract_slice [[ARG]]
  %7 = secret.generic(%arg0 : !secret.secret<tensor<1x1xi32>>) {
  ^bb0(%arg1: tensor<1x1xi32>):
    %20 = tensor.extract %arg1[%c0, %c0] : tensor<1x1xi32>
    secret.yield %20 : i32
  } -> !secret.secret<i32>

  // CHECK: [[V6:%.*]] = tensor.reshape [[V7]]
  %6 = secret.generic(%7: !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %alloc = tensor.from_elements %arg1 : tensor<1x1xi32>
    secret.yield %alloc : tensor<1x1xi32>
  } -> !secret.secret<tensor<1x1xi32>>

  // CHECK: return [[V6]] : [[OUT]]
  return %6 : !secret.secret<tensor<1x1xi32>>
}

// -----

// CHECK: @single_bit_plaintext_tensor
// CHECK-SAME: () -> [[OUT:tensor<1x!lwe.lwe_ciphertext<.*>>]]
func.func @single_bit_plaintext_tensor() -> !secret.secret<tensor<1xi1>> {
  // CHECK: [[TRUE:%.*]] = arith.constant true
  // CHECK: [[ENC:%.*]] = lwe.encode [[TRUE]]
  // CHECK: [[LWE:%.*]] = lwe.trivial_encrypt [[ENC]]
  %true = arith.constant true
  // CHECK: [[V6:%.*]] = tensor.from_elements [[LWE]]
  %6 = secret.generic() {
    %from_elements = tensor.from_elements %true : tensor<1xi1>
    secret.yield %from_elements : tensor<1xi1>
  } -> !secret.secret<tensor<1xi1>>
  // CHECK: return [[V6]] : [[OUT]]
  return %6 : !secret.secret<tensor<1xi1>>
}
