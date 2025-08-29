// RUN: heir-opt --openfhe-fast-rotation-precompute %s | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!cc = !openfhe.crypto_context
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L0 = !rns.rns<!Z1095233372161_i64>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**32>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32, encryption_type = lsb>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32, encryption_type = lsb>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L5_C1>

module {
  func.func @simple_sum(%cc: !cc, %ct: !ct_L1) -> !ct_L1 {
    // CHECK: openfhe.fast_rotation_precompute
    // CHECK-COUNT-4: openfhe.fast_rotation
    // CHECK-NOT: openfhe.rot
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi64>
    %ct_0 = openfhe.rot %cc, %ct {index = 16 : index} : (!cc, !ct_L1) -> !ct_L1
    %ct_1 = openfhe.add %cc, %ct, %ct_0 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    %ct_2 = openfhe.rot %cc, %ct {index = 8 : index} : (!cc, !ct_L1) -> !ct_L1
    %ct_3 = openfhe.add %cc, %ct_1, %ct_2 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    %ct_4 = openfhe.rot %cc, %ct {index = 5 : index} : (!cc, !ct_L1) -> !ct_L1
    %ct_5 = openfhe.add %cc, %ct_3, %ct_4 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    %ct_6 = openfhe.rot %cc, %ct {index = 12 : index} : (!cc, !ct_L1) -> !ct_L1
    %ct_7 = openfhe.add %cc, %ct_5, %ct_6 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    return %ct_7 : !ct_L1
  }
}
