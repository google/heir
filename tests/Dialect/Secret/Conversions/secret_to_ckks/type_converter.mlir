// Tests that the type converter handles both converting tensors to SIMD slots
// if aligned, or to tensors of ciphertext.

// RUN: heir-opt --mlir-print-local-scope --secret-insert-mgmt-ckks=include-first-mul=false --generate-param-ckks --secret-distribute-generic --canonicalize --secret-to-ckks --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test_arg_packed
// CHECK-SAME: %[[arg0:.*]]: !lwe.new_lwe_ciphertext<{{.*}}message_type = tensor<1024xf32>{{.*}}>
func.func @test_arg_packed(%arg0 : !secret.secret<tensor<1024xf32>>) -> (!secret.secret<tensor<1024xf32>>) {
  // CHECK: return
  // CHECK-SAME: message_type = tensor<1024xf32>
  // CHECK-SAME: polynomialModulus = <1 + x**1024>
  return %arg0 : !secret.secret<tensor<1024xf32>>
}

// -----

// CHECK-LABEL: func @test_extract_not_packed
// CHECK-SAME: tensor<1023x!lwe.new_lwe_ciphertext<{{.*}}message_type = f32{{.*}}>
func.func @test_extract_not_packed(%arg0 : !secret.secret<tensor<1023xf32>>) -> (!secret.secret<f32>) {
  %c0 = arith.constant 0 : index
  %0 = secret.generic ins(%arg0 :  !secret.secret<tensor<1023xf32>>) {
  // CHECK: tensor.extract
    ^bb0(%ARG0 : tensor<1023xf32>):
      %1 = tensor.extract %ARG0[%c0] : tensor<1023xf32>
      secret.yield %1 : f32
  } -> !secret.secret<f32>
  // CHECK: return
  // CHECK-SAME: message_type = f32
  // CHECK-SAME: polynomialModulus = <1 + x**1024>
  return %0 : !secret.secret<f32>
}

// -----

// CHECK-LABEL: func @test_add_scalar_not_packed
// CHECK-SAME: !lwe.new_lwe_ciphertext<{{.*}}message_type = f32{{.*}}>
func.func @test_add_scalar_not_packed(%arg0 : !secret.secret<f32>) -> (!secret.secret<f32>) {
  %0 = secret.generic ins(%arg0 :  !secret.secret<f32>) {
  // CHECK: ckks.add
    ^bb0(%ARG0 : f32):
      %1 = arith.addf %ARG0, %ARG0 : f32
      secret.yield %1 : f32
  } -> !secret.secret<f32>
  // CHECK: return
  // CHECK-SAME: !lwe.new_lwe_ciphertext<{{.*}}message_type = f32>
  return %0 : !secret.secret<f32>
}

// -----

// CHECK-LABEL: func @test_add_plain_scalar_not_packed
// CHECK-SAME: !lwe.new_lwe_ciphertext<{{.*}}message_type = f32{{.*}}>
func.func @test_add_plain_scalar_not_packed(%arg0 : !secret.secret<f32>) -> (!secret.secret<f32>) {
  %c1_f32 = arith.constant 1.0 : f32
  %0 = secret.generic ins(%arg0 :  !secret.secret<f32>) {
  // CHECK: ckks.add_plain
    ^bb0(%ARG0 : f32):
      %1 = arith.addf %ARG0, %c1_f32 : f32
      secret.yield %1 : f32
  } -> !secret.secret<f32>
  // CHECK: return
  // CHECK-SAME: !lwe.new_lwe_ciphertext<{{.*}}message_type = f32{{.*}}>
  return %0 : !secret.secret<f32>
}

// -----

// CHECK-LABEL: func @test_insert
func.func @test_insert(%arg0 : !secret.secret<tensor<1023xf32>>, %arg1 : !secret.secret<f32>) -> (!secret.secret<tensor<1023xf32>>) {
  %c0 = arith.constant 0 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<1023xf32>>, !secret.secret<f32>) {
  // CHECK: tensor.insert
    ^bb0(%ARG0 : tensor<1023xf32>, %ARG1: f32):
      %1 = tensor.insert %ARG1 into %ARG0[%c0] : tensor<1023xf32>
      secret.yield %1 : tensor<1023xf32>
  } -> !secret.secret<tensor<1023xf32>>
  // CHECK: return
  // CHECK-SAME: message_type = f32
  // CHECK-SAME: polynomialModulus = <1 + x**1024>
  return %0 : !secret.secret<tensor<1023xf32>>
}

// -----

// CHECK-LABEL: func @test_2d_arg_packed
// CHECK-SAME: %[[arg0:.*]]: !lwe.new_lwe_ciphertext<{{.*}}message_type = tensor<1x1024xf32>{{.*}}>
func.func @test_2d_arg_packed(%arg0 : !secret.secret<tensor<1x1024xf32>>) -> (!secret.secret<tensor<1x1024xf32>>) {
  // CHECK: return
  // CHECK-SAME: message_type = tensor<1x1024xf32>>
  // CHECK-SAME: polynomialModulus = <1 + x**1024>
  return %arg0 : !secret.secret<tensor<1x1024xf32>>
}
