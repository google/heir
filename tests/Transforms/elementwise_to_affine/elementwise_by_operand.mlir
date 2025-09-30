// RUN: heir-opt --convert-elementwise-to-affine %s | FileCheck %s

// Primarily we are testing for the ckks.relinearize op, which is mappable only
// over its non-key switching key argument.

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct1 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct2 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: @test_multiply_elementwise
func.func @test_multiply_elementwise(%arg0 : tensor<5x!ct>, %arg1: tensor<5x!ct>, %ksk: tensor<10x!ct>) -> tensor<5x!ct> {
  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.add
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %add = ckks.add %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>

  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.sub
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %sub = ckks.sub %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>

  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.negate
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %neg = ckks.negate %arg0 : tensor<5x!ct>

  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.mul
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %0 = ckks.mul %arg0, %arg1  : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct1>

  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.relinearize
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %1 = ckks.relinearize %0, %ksk {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (tensor<5x!ct1>, tensor<10x!ct>) -> tensor<5x!ct>

  // CHECK: affine.for
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: ckks.rescale
  // CHECK-NEXT: tensor.insert
  // CHECK-NEXT: affine.yield
  %2 = ckks.rescale %1  {to_ring = #ring_rns_L0_1_x1024_} : tensor<5x!ct> -> tensor<5x!ct2>
  return %arg0 : tensor<5x!ct>
}
