// RUN: heir-opt --convert-elementwise-to-affine=convert-dialects=kernel %s | FileCheck %s

// kernel.linear_transform is mapped over its `input` operand only; the
// `diagonals` operand is replicated into each scalarized op.

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>

!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

// CHECK: @test_linear_transform_ciphertext
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x!
func.func @test_linear_transform_ciphertext(%arg0: tensor<1x!ct>) -> tensor<1x!ct> {
  // CHECK: %[[DIAGONALS:[a-zA-Z0-9_]+]] = arith.constant {{.*}} : tensor<2x512xf64>
  %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
  // CHECK-NOT: tensor.extract{{(_slice)?}} %[[DIAGONALS]]
  // CHECK: affine.for %[[I:[a-zA-Z0-9_]+]] = 0 to 1
  // CHECK-NEXT: %[[CT:[a-zA-Z0-9_]+]] = tensor.extract %[[ARG0]][%[[I]]]
  // CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = kernel.linear_transform %[[CT]], %[[DIAGONALS]] {{.*}}diagonal_indices = array<i64: 0, 1>{{.*}} : !{{[a-zA-Z0-9_.]+}}, tensor<2x512xf64> -> !
  // CHECK-NEXT: tensor.insert %[[RES]]
  // CHECK-NEXT: affine.yield
  %0 = kernel.linear_transform %arg0, %diagonals {diagonal_indices = array<i64: 0, 1>} : tensor<1x!ct>, tensor<2x512xf64> -> tensor<1x!ct>
  return %0 : tensor<1x!ct>
}

// Each ciphertext in the input tensor is transformed independently, with the
// same diagonals.

// CHECK: @test_linear_transform_multi_ciphertext
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x!
func.func @test_linear_transform_multi_ciphertext(%arg0: tensor<2x!ct>) -> tensor<2x!ct> {
  // CHECK: %[[DIAGONALS:[a-zA-Z0-9_]+]] = arith.constant {{.*}} : tensor<2x512xf64>
  %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
  // CHECK-NOT: tensor.extract{{(_slice)?}} %[[DIAGONALS]]
  // CHECK: affine.for %[[I:[a-zA-Z0-9_]+]] = 0 to 2
  // CHECK-NEXT: %[[CT:[a-zA-Z0-9_]+]] = tensor.extract %[[ARG0]][%[[I]]]
  // CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = kernel.linear_transform %[[CT]], %[[DIAGONALS]] {{.*}}diagonal_indices = array<i64: 0, 1>{{.*}} : !{{[a-zA-Z0-9_.]+}}, tensor<2x512xf64> -> !
  // CHECK-NEXT: tensor.insert %[[RES]]
  // CHECK-NEXT: affine.yield
  %0 = kernel.linear_transform %arg0, %diagonals {diagonal_indices = array<i64: 0, 1>} : tensor<2x!ct>, tensor<2x512xf64> -> tensor<2x!ct>
  return %0 : tensor<2x!ct>
}

// CHECK: @test_linear_transform_cleartext
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<4xf32>, %[[DIAGONALS:[a-zA-Z0-9_]+]]: tensor<2x4xf32>)
func.func @test_linear_transform_cleartext(%arg0: tensor<4xf32>, %diagonals: tensor<2x4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: affine.for
  // CHECK: %[[RES:[a-zA-Z0-9_]+]] = kernel.linear_transform %[[ARG0]], %[[DIAGONALS]] {{.*}} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
  // CHECK-NOT: affine.for
  // CHECK: return %[[RES]]
  %0 = kernel.linear_transform %arg0, %diagonals {diagonal_indices = array<i64: 0, 1>} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
