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

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct1 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct2 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

!ct_tensor = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_scalar = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

// CHECK: module
module {
  // CHECK: @test_rotate_cols_dynamic
  // CHECK-SAME: (%[[arg0:.*]]: !lwe
  func.func @test_rotate_cols_dynamic(%arg0: !ct) -> !ct {
    // CHECK: %[[c4:.*]] = arith.constant 4 : i32
    %c4 = arith.constant 4 : i32
    // CHECK: bgv.rotate_cols %[[arg0]], %[[c4]] : i32 : !lwe
    %rot = bgv.rotate_cols %arg0, %c4 : i32 : !ct
    return %rot : !ct
  }

  // CHECK: @test_multiply
  func.func @test_multiply(%arg0 : !ct, %arg1: !ct) -> !ct {
    %add = bgv.add %arg0, %arg1 : (!ct, !ct) -> !ct
    %sub = bgv.sub %arg0, %arg1 : (!ct, !ct) -> !ct
    %neg = bgv.negate %arg0 : !ct

    %0 = bgv.mul %arg0, %arg1  : (!ct, !ct) -> !ct1
    %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct
    %2 = bgv.modulus_switch %1  {to_ring = #ring_rns_L0_1_x1024_} : !ct -> !ct2
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>>, polynomialModulus = <1 + x**1024>>
    return %arg0 : !ct
  }

  // CHECK: @test_ciphertext_plaintext
  func.func @test_ciphertext_plaintext(%arg0: !pt, %arg1: !pt, %arg2: !pt, %arg3: !ct) -> !ct {
    %add = bgv.add_plain %arg3, %arg0 : (!ct, !pt) -> !ct
    %sub = bgv.sub_plain %add, %arg1 : (!ct, !pt) -> !ct
    %mul = bgv.mul_plain %sub, %arg2 : (!ct, !pt) -> !ct
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    return %mul : !ct
  }

  // CHECK: @test_multiply_elementwise
  func.func @test_multiply_elementwise(%arg0 : tensor<5x!ct>, %arg1: tensor<5x!ct>) -> tensor<5x!ct> {
    %add = bgv.add %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>
    %sub = bgv.sub %arg0, %arg1 : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct>
    %neg = bgv.negate %arg0 : tensor<5x!ct>

    %0 = bgv.mul %arg0, %arg1  : (tensor<5x!ct>, tensor<5x!ct>) -> tensor<5x!ct1>
    %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : tensor<5x!ct1> -> tensor<5x!ct>
    %2 = bgv.modulus_switch %1  {to_ring = #ring_rns_L0_1_x1024_} : tensor<5x!ct> -> tensor<5x!ct2>
    return %arg0 : tensor<5x!ct>
  }

  // CHECK: @test_ciphertext_plaintext_elementwise
  func.func @test_ciphertext_plaintext_elementwise(%arg0: tensor<5x!pt>, %arg1: tensor<5x!pt>, %arg2: tensor<5x!pt>, %arg3: tensor<5x!ct>) -> tensor<5x!ct> {
    %add = bgv.add_plain %arg3, %arg0 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    %sub = bgv.sub_plain %add, %arg1 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    %mul = bgv.mul_plain %sub, %arg2 : (tensor<5x!ct>, tensor<5x!pt>) -> tensor<5x!ct>
    // CHECK: ring = <coefficientType = !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>, polynomialModulus = <1 + x**1024>>
    return %mul : tensor<5x!ct>
  }
}
