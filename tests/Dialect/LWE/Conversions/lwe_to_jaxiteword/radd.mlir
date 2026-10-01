// RUN: heir-opt --lwe-to-jaxiteword %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
!ct_L1_ = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!pt_ = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>>

module {
  // CHECK: func.func @test_radd
  // CHECK-SAME: (%{{[^:]*}}: !jaxiteword.crypto_context<>, %{{[^:]*}}: !jaxiteword.eval_key<>
  func.func @test_radd(%ct: !ct_L1_, %ct_0: !ct_L1_) -> !ct_L1_ {
    // CHECK: jaxiteword.add
    %sum = lwe.radd %ct, %ct_0 : (!ct_L1_, !ct_L1_) -> !ct_L1_
    return %sum : !ct_L1_
  }

  // CHECK: func.func @test_radd_plain
  func.func @test_radd_plain(%ct: !ct_L1_, %pt: !pt_) -> !ct_L1_ {
    // CHECK: jaxiteword.add_plain {{.*}}, %{{.*}}, %{{.*}} : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    %sum = lwe.radd_plain %ct, %pt : (!ct_L1_, !pt_) -> !ct_L1_
    return %sum : !ct_L1_
  }

  // CHECK: func.func @test_radd_plain_reversed
  func.func @test_radd_plain_reversed(%pt: !pt_, %ct: !ct_L1_) -> !ct_L1_ {
    // CHECK: jaxiteword.add_plain {{.*}}, %{{.*}}, %{{.*}} : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    %sum = lwe.radd_plain %pt, %ct : (!pt_, !ct_L1_) -> !ct_L1_
    return %sum : !ct_L1_
  }

  // CHECK: func.func @test_ckks_add_plain
  func.func @test_ckks_add_plain(%ct: !ct_L1_, %pt: !pt_) -> !ct_L1_ {
    // CHECK: jaxiteword.add_plain
    %sum = ckks.add_plain %ct, %pt : (!ct_L1_, !pt_) -> !ct_L1_
    return %sum : !ct_L1_
  }

  // CHECK: func.func @test_rmul_plain_reversed
  func.func @test_rmul_plain_reversed(%pt: !pt_, %ct: !ct_L1_) -> !ct_L1_ {
    // CHECK: jaxiteword.mul_plain {{.*}}, %{{.*}}, %{{.*}} : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    %product = lwe.rmul_plain %pt, %ct : (!pt_, !ct_L1_) -> !ct_L1_
    return %product : !ct_L1_
  }
}
