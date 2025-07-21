// Regression test for https://github.com/google/heir/issues/1621
// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!Z1073750017_i64 = !mod_arith.int<1073750017 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!params = !openfhe.cc_params
!pk = !openfhe.public_key
!sk = !openfhe.private_key
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <67239937 : i64, 1073750017 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <67239937 : i64, 1073750017 : i64>, current = 1>
!rns_L0 = !rns.rns<!Z67239937_i64>
!rns_L1 = !rns.rns<!Z67239937_i64, !Z1073750017_i64>
#ring_Z65537_i64_1_x1024 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pt = !lwe.lwe_plaintext<application_data = <message_type = i1>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>>
!pt1 = !lwe.lwe_plaintext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = lsb>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb>
!ct_L0 = !lwe.lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = i1>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<application_data = <message_type = i64>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
module attributes {scheme.bgv} {
  // CHECK: CiphertextT cond
  // CHECK-SAME: CryptoContextT [[cc:.*]], int64_t [[v0:.*]], int64_t [[v1:.*]], CiphertextT [[ct:.*]]
  func.func @cond(%cc: !cc, %arg0: i64, %arg1: i64, %ct: !ct_L1) -> !ct_L0 {
    // CHECK: std::vector<int64_t> [[v2:.*]](1024, 1);
    // CHECK-NEXT: auto [[pt:.*]]_filled_n = [[cc]]->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    // CHECK-NEXT: auto [[pt]]_filled = [[v2]]
    // CHECK: const auto& [[pt]] = [[cc]]->MakePackedPlaintext
    %cst = arith.constant dense<1> : tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<1024xi64>) -> !pt
    %ct_0 = openfhe.negate %cc, %ct : (!cc, !ct_L1) -> !ct_L1
    %ct_1 = openfhe.add_plain %cc, %ct_0, %pt : (!cc, !ct_L1, !pt) -> !ct_L1
    %ct_2 = lwe.reinterpret_application_data %ct : !ct_L1 to !ct_L1_1
    %splat = tensor.splat %arg0 : tensor<1024xi64>
    // CHECK: [[cc]]->MakePackedPlaintext
    %pt_3 = openfhe.make_packed_plaintext %cc, %splat : (!cc, tensor<1024xi64>) -> !pt1
    %ct_4 = openfhe.mul_plain %cc, %ct_2, %pt_3 : (!cc, !ct_L1_1, !pt1) -> !ct_L1_1
    %ct_5 = lwe.reinterpret_application_data %ct_1 : !ct_L1 to !ct_L1_1
    %splat_6 = tensor.splat %arg1 : tensor<1024xi64>
    // CHECK: [[cc]]->MakePackedPlaintext
    %pt_7 = openfhe.make_packed_plaintext %cc, %splat_6 : (!cc, tensor<1024xi64>) -> !pt1
    %ct_8 = openfhe.mul_plain %cc, %ct_5, %pt_7 : (!cc, !ct_L1_1, !pt1) -> !ct_L1_1
    %ct_9 = openfhe.add %cc, %ct_4, %ct_8 : (!cc, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    // CHECK-NOT: [[pt]]_filled_n
    %pt_10 = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<1024xi64>) -> !pt1
    %ct_11 = openfhe.add_plain %cc, %ct_9, %pt_10 : (!cc, !ct_L1_1, !pt1) -> !ct_L1_1
    %ct_12 = openfhe.mod_reduce %cc, %ct_11 : (!cc, !ct_L1_1) -> !ct_L0
    // CHECK: return
    return %ct_12 : !ct_L0
  }
  func.func @cond__encrypt__arg2(%cc: !cc, %arg0: i1, %pk: !pk) -> !ct_L1 {
    %splat = tensor.splat %arg0 : tensor<1024xi1>
    %0 = arith.extui %splat : tensor<1024xi1> to tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %0 : (!cc, tensor<1024xi64>) -> !pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct_L1
    return %ct : !ct_L1
  }
  func.func @cond__decrypt__result0(%cc: !cc, %ct: !ct_L0, %sk: !sk) -> i64 {
    %pt = openfhe.decrypt %cc, %ct, %sk : (!cc, !ct_L0, !sk) -> !pt1
    %0 = lwe.rlwe_decode %pt {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x1024} : !pt1 -> i64
    return %0 : i64
  }
  func.func @cond__generate_crypto_context() -> !cc {
    %params = openfhe.gen_params  {encryptionTechniqueExtended = false, evalAddCount = 2 : i64, insecure = false, keySwitchCount = 0 : i64, mulDepth = 1 : i64, plainMod = 65537 : i64} : () -> !params
    %cc = openfhe.gen_context %params {supportFHE = false} : (!params) -> !cc
    return %cc : !cc
  }
  func.func @cond__configure_crypto_context(%cc: !cc, %sk: !sk) -> !cc {
    return %cc : !cc
  }
}
