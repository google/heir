// RUN: heir-opt --convert-elementwise-to-affine=convert-dialects=kernel %s | FileCheck %s

// kernel.apply_linear_transform is mapped over its `input` operand only; the
// `prepared` operand is replicated into each scalarized op.

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
!prepared = !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>

// CHECK: @test_apply_linear_transform_ciphertext
func.func @test_apply_linear_transform_ciphertext(%arg0: tensor<1x!ct>, %prepared: !prepared) -> tensor<1x!ct> {
  // CHECK-NOT: tensor.extract{{.*}} %arg1
  // CHECK: affine.for %[[I:.*]] = 0 to 1
  // CHECK: %[[CT:.*]] = tensor.extract %arg0[%[[I]]]
  // CHECK: %[[RES:.*]] = kernel.apply_linear_transform %[[CT]], %arg1 : !{{.*}}, <level = 1, slots = 512, log_bsgs_ratio = 0> -> !
  // CHECK: tensor.insert %[[RES]]
  // CHECK: affine.yield
  %0 = kernel.apply_linear_transform %arg0, %prepared : tensor<1x!ct>, !prepared -> tensor<1x!ct>
  return %0 : tensor<1x!ct>
}

// CHECK: @test_apply_linear_transform_multi_ciphertext
func.func @test_apply_linear_transform_multi_ciphertext(%arg0: tensor<2x!ct>, %prepared: !prepared) -> tensor<2x!ct> {
  // CHECK-NOT: tensor.extract{{.*}} %arg1
  // CHECK: affine.for %[[I:.*]] = 0 to 2
  // CHECK: %[[CT:.*]] = tensor.extract %arg0[%[[I]]]
  // CHECK: %[[RES:.*]] = kernel.apply_linear_transform %[[CT]], %arg1 : !{{.*}}, <level = 1, slots = 512, log_bsgs_ratio = 0> -> !
  // CHECK: tensor.insert %[[RES]]
  // CHECK: affine.yield
  %0 = kernel.apply_linear_transform %arg0, %prepared : tensor<2x!ct>, !prepared -> tensor<2x!ct>
  return %0 : tensor<2x!ct>
}
