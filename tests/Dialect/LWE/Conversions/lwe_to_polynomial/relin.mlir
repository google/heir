// RUN: heir-opt --lwe-to-polynomial %s | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
#ring_Z65537_i64_1_x1024 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb, size = 3>
!ct_L1 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L5_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L5_C1>

module {
  // CHECK: @test_relin(
  // CHECK-SAME: [[X:%.+]]: [[ct_ty:.*tensor<3x.*]],
  // CHECK-SAME: [[ksk:%.+]]: [[ksk_ty:tensor<10x.*]])
  func.func @test_relin(%ct: !ct_L1_D3, %arg0: tensor<10x!ct_L1>) -> !ct_L1 {
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    // CHECK-DAG: [[extracted0:%.+]] = tensor.extract [[X]]{{\[}}[[C0]]]
    // CHECK-DAG: [[extracted1:%.+]] = tensor.extract [[X]]{{\[}}[[C1]]]
    // CHECK-DAG: [[extracted2:%.+]] = tensor.extract [[X]]{{\[}}[[C2]]]

    // CHECK-NEXT: [[constOutput:%.+]], [[linearOutput:%.+]] = polynomial.key_switch_inner [[extracted2]], [[ksk]]
    // CHECK-DAG: [[tensor0:%.+]] = polynomial.add [[extracted0]], [[constOutput]]
    // CHECK-DAG: [[tensor1:%.+]] = polynomial.add [[extracted1]], [[linearOutput]]
    // CHECK-NEXT: [[result:%.+]] = tensor.from_elements [[tensor0]], [[tensor1]]
    // CHECK-NEXT: return [[result]]
    %ct_0 = ckks.relinearize %ct, %arg0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3, tensor<10x!ct_L1>) -> !ct_L1
    return %ct_0 : !ct_L1
  }
}
