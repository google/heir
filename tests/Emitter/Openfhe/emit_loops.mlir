// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module attributes {scheme.ckks} {
  // CHECK: test_affine_for
  // CHECK-SAME: CryptoContextT [[cc:.*]], CiphertextT [[ct:.*]]) {
  // CHECK: MutableCiphertextT [[ct1:.*]] = [[ct]]->Clone();
  // CHECK: for (auto [[v0:.*]] = 1; [[v0]] < 2; ++[[v0]]) {
  // CHECK:   [[ct1]] = [[cc]]->EvalRotate([[ct1]], 1);
  // CHECK: }
  // CHECK: return [[ct1]];
  func.func @test_affine_for(%cc: !openfhe.crypto_context, %ct: !ct_L0_) -> !ct_L0_ {
    %1 = affine.for %arg0 = 1 to 2 iter_args(%arg1 = %ct) -> (!ct_L0_) {
      %ct_12 = openfhe.rot %cc, %arg1 {index = 1 : index} : (!openfhe.crypto_context, !ct_L0_) -> !ct_L0_
      affine.yield %ct_12 : !ct_L0_
    }
    return %1 : !ct_L0_
  }
}
