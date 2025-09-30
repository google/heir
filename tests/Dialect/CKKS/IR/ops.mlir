// RUN: heir-opt --mlir-print-local-scope --color %s | FileCheck %s

// This simply tests for syntax.

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

!pt = !lwe.lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct1 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct2 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

!ct_tensor = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_scalar = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

// CHECK: module
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: @test_multiply
  func.func @test_multiply(%arg0 : !ct, %arg1: !ct, %ksk: tensor<10x!ct>) -> !ct {
    %add = ckks.add %arg0, %arg1 : (!ct, !ct) -> !ct
    %sub = ckks.sub %arg0, %arg1 : (!ct, !ct) -> !ct
    %neg = ckks.negate %arg0 : !ct

    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    // CHECK: size = 3
    %0 = ckks.mul %arg0, %arg1  : (!ct, !ct) -> !ct1
    %1 = ckks.relinearize %0, %ksk {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (!ct1, tensor<10x!ct>) -> !ct
    %2 = ckks.rescale %1  {to_ring = #ring_rns_L0_1_x1024_} : !ct -> !ct2
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>>, polynomialModulus = <1 + x**1024>>
    return %arg0 : !ct
  }

  // CHECK: @test_relin_no_key
  func.func @test_relin_no_key(%arg0 : !ct1) -> !ct {
    %1 = ckks.relinearize %arg0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (!ct1) -> !ct
    return %1 : !ct
  }

  // CHECK: @test_ciphertext_plaintext
  func.func @test_ciphertext_plaintext(%arg0: !pt, %arg1: !pt, %arg2: !pt, %arg3: !ct) -> !ct {
    %add = ckks.add_plain %arg3, %arg0 : (!ct, !pt) -> !ct
    %sub = ckks.sub_plain %add, %arg1 : (!ct, !pt) -> !ct
    %mul = ckks.mul_plain %sub, %arg2 : (!ct, !pt) -> !ct
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    return %mul : !ct
  }

  // CHECK: @test_rotate_extract
  func.func @test_rotate_extract(%arg3: !ct_tensor) -> !ct_scalar {
    %c0 = arith.constant 0 : index
    %add = ckks.rotate %arg3 { offset = 1 } : !ct_tensor
    %ext = ckks.extract %add, %c0 : (!ct_tensor, index) -> !ct_scalar
    // CHECK: message_type = i16
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    return %ext : !ct_scalar
  }

  // CHECK: @test_multiply_elementwise
  func.func @test_multiply_elementwise(%arg0 : tensor<5x!ct>, %arg1: tensor<5x!ct>, %ksk: tensor<10x!ct>) -> tensor<5x!ct> {
    %add = ckks.add %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>
    %sub = ckks.sub %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>
    %neg = ckks.negate %arg0 : tensor<5x!ct>

    %0 = ckks.mul %arg0, %arg1  : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct1>
    %1 = ckks.relinearize %0, %ksk {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (tensor<5x!ct1>, tensor<10x!ct>) -> tensor<5x!ct>
    %2 = ckks.rescale %1  {to_ring = #ring_rns_L0_1_x1024_} : tensor<5x!ct> -> tensor<5x!ct2>
    return %arg0 : tensor<5x!ct>
  }

  // CHECK: @test_ciphertext_plaintext_elementwise
  func.func @test_ciphertext_plaintext_elementwise(%arg0: tensor<5x!pt>, %arg1: tensor<5x!pt>, %arg2: tensor<5x!pt>, %arg3: tensor<5x!ct>) -> tensor<5x!ct> {
    %add = ckks.add_plain %arg3, %arg0 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    %sub = ckks.sub_plain %add, %arg1 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    %mul = ckks.mul_plain %sub, %arg2 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    return %mul : tensor<5x!ct>
  }
}
