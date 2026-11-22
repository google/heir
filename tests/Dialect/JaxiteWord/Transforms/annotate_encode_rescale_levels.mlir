// RUN: heir-opt --jaxiteword-annotate-encode-rescale-levels %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 1>
#modulus_chain_L0 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 0>
!rns_L1 = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L0 = !rns.rns<!Z1095233372161_i64_>
#ring_pt = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_ct_L1 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_ct_L0 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ct_space_L1 = #lwe.ciphertext_space<ring = #ring_ct_L1, encryption_type = lsb>
#ct_space_L0 = #lwe.ciphertext_space<ring = #ring_ct_L0, encryption_type = lsb>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_pt, encoding = #encoding>, ciphertext_space = #ct_space_L1, key = #key, modulus_chain = #modulus_chain_L1>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_pt, encoding = #encoding>, ciphertext_space = #ct_space_L0, key = #key, modulus_chain = #modulus_chain_L0>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_pt, encoding = #encoding>>
!storage = !preprocessing.storage<!pt>

module {
  func.func @annotates_jaxiteword_encode(%ctx: !jaxiteword.crypto_context<>, %input: tensor<1024xi16>, %ct: !ct_L1) -> !ct_L0 {
    // CHECK: jaxiteword.encode
    // CHECK-SAME: rescaleLevel = 1 : i64
    %pt = jaxiteword.encode %ctx, %input : (!jaxiteword.crypto_context<>, tensor<1024xi16>) -> !pt
    %mul = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    %out = jaxiteword.mod_reduce %ctx, %mul : (!jaxiteword.crypto_context<>, !ct_L1) -> !ct_L0
    return %out : !ct_L0
  }

  func.func @leaves_non_rescaled_encode(%ctx: !jaxiteword.crypto_context<>, %input: tensor<1024xi16>, %ct: !ct_L1) -> !ct_L1 {
    // CHECK: func.func @leaves_non_rescaled_encode
    // CHECK: jaxiteword.encode
    // CHECK-NOT: rescaleLevel
    %pt = jaxiteword.encode %ctx, %input : (!jaxiteword.crypto_context<>, tensor<1024xi16>) -> !pt
    %mul = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    return %mul : !ct_L1
  }

  func.func @stores_encoded_plaintext(%ctx: !jaxiteword.crypto_context<>, %input: tensor<1024xi16>, %storage: !storage) {
    // CHECK: func.func @stores_encoded_plaintext
    // CHECK: jaxiteword.encode
    // CHECK-SAME: rescaleLevel = 1 : i64
    %pt = jaxiteword.encode %ctx, %input : (!jaxiteword.crypto_context<>, tensor<1024xi16>) -> !pt
    preprocessing.store %pt, %storage[] site 7 <!pt> : !pt, !storage
    return
  }

  func.func @loads_encoded_plaintext(%ctx: !jaxiteword.crypto_context<>, %storage: !storage, %ct: !ct_L1) -> !ct_L0 {
    %pt = preprocessing.load %storage[] site 7 <!pt> : !storage, !pt
    %mul = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
    %out = jaxiteword.mod_reduce %ctx, %mul : (!jaxiteword.crypto_context<>, !ct_L1) -> !ct_L0
    return %out : !ct_L0
  }
}
