// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=bootstrap %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>


func.func @bootstrap(%arg0: !openfhe.crypto_context, %arg1: !ct_L0_) -> !ct_L0_ {
  %0 = openfhe.bootstrap %arg0, %arg1 : (!openfhe.crypto_context, !ct_L0_) -> !ct_L0_
  return %0 : !ct_L0_
}

// CHECK: @bootstrap
// CHECK: @bootstrap__generate_crypto_context
// CHECK: mulDepth = 20
// CHECK: openfhe.gen_context %{{.*}} {supportFHE = true}

// CHECK: @bootstrap__configure_crypto_context
// CHECK: openfhe.gen_mulkey
// CHECK: openfhe.setup_bootstrap %{{.*}} {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index}
// CHECK: openfhe.gen_bootstrapkey
