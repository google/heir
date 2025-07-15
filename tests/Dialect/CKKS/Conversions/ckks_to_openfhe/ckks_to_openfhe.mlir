// RUN: heir-opt --mlir-print-local-scope --ckks-to-lwe --lwe-to-openfhe %s | FileCheck %s

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
#ciphertext_space_L1_D4_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 4>

!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D4 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D4_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L0 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: module
module {
  // CHECK: @test_ops
  // CHECK-SAME: ([[C:%.+]]: [[S:.*crypto_context]], [[X:%.+]]: [[T:.*lwe_ciphertext.*]], [[Y:%.+]]: [[T]], [[Z:%.+]]: [[P:.*lwe_plaintext[^)]*]])
  func.func @test_ops(%x : !ct, %y : !ct, %z : !pt) -> (!ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct) {
    // CHECK: %[[v1:.*]] = openfhe.negate [[C]], %[[x1:.*]] : ([[S]], [[T]]) -> [[T]]
    %negate = ckks.negate %x  : !ct
    // CHECK: %[[v2:.*]] = openfhe.add [[C]], %[[x2:.*]], %[[y2:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %add = ckks.add %x, %y  : (!ct, !ct) -> !ct
    // CHECK: %[[v3:.*]] = openfhe.sub [[C]], %[[x3:.*]], %[[y3:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %sub = ckks.sub %x, %y  : (!ct, !ct) -> !ct
    // CHECK: %[[v4:.*]] = openfhe.mul_no_relin [[C]], %[[x4:.*]], %[[y4:.*]]: ([[S]], [[T]], [[T]]) -> [[T2:.*]]
    %mul = ckks.mul %x, %y  : (!ct, !ct) -> !ct_D3
    // CHECK: %[[v5:.*]] = openfhe.rot [[C]], %[[x5:.*]] {index = 4 : i64}
    // CHECK-SAME: ([[S]], [[T]]) -> [[T]]
    %rot = ckks.rotate %x { offset = 4 } : !ct
    // CHECK: %[[v6:.*]] = openfhe.add_plain [[C]], %[[x6:.*]], %[[z6:.*]]: ([[S]], [[T]], [[P]]) -> [[T]]
    %add_plain = ckks.add_plain %x, %z : (!ct, !pt) -> !ct
    // CHECK: %[[v7:.*]] = openfhe.sub_plain [[C]], %[[x7:.*]], %[[z7:.*]]: ([[S]], [[T]], [[P]]) -> [[T]]
    %sub_plain = ckks.sub_plain %x, %z : (!ct, !pt) -> !ct
    // CHECK: %[[v7:.*]] = openfhe.mul_plain [[C]], %[[x8:.*]], %[[z8:.*]]: ([[S]], [[T]], [[P]]) -> [[T]]
    %mul_plain = ckks.mul_plain %x, %z : (!ct, !pt) -> !ct
    return %negate, %add, %sub, %mul, %rot, %add_plain, %sub_plain, %mul_plain : !ct, !ct, !ct, !ct_D3, !ct, !ct, !ct, !ct
  }

  // CHECK: @test_relin
  // CHECK-SAME: ([[C:.*]]: [[S:.*crypto_context]], [[X:%.+]]: [[T:.*lwe_ciphertext.*]])
  func.func @test_relin(%x : !ct_D4) -> !ct {
    // CHECK: %[[v6:.*]] = openfhe.relin [[C]], %[[x6:.*]]: ([[S]], [[T]]) -> [[T2:.*]]
    %relin = ckks.relinearize %x  {
      from_basis = array<i32: 0, 1, 2, 3>, to_basis = array<i32: 0, 1>
    }: (!ct_D4) -> !ct
    return %relin : !ct
  }
}
