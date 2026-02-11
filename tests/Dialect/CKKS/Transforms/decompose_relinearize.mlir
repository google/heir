// RUN: heir-opt --ckks-decompose-relinearize %s | FileCheck %s

#q0 = 1095233372161 : i64
#q1 = 1032955396097 : i64
#p0 = 261405424692085787 : i64

!Zq0 = !mod_arith.int<#q0>
!Zq1 = !mod_arith.int<#q1>
!Zp0 = !mod_arith.int<#p0>

// Input's type
#ring_L1x1024 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_L1x1024, encryption_type = lsb, size = 3>
#modulus_chain_L5_C1 = #lwe.modulus_chain<elements = <#q0, #q1>, current = 1>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
!ct_L1_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1x1024,
                                encoding = #inverse_canonical_encoding>,
                                ciphertext_space = #ciphertext_space_L1_D3,
                                key = #key,
                                modulus_chain = #modulus_chain_L5_C1>

// Output's type
#ciphertext_space_L1_D2 = #lwe.ciphertext_space<ring = #ring_L1x1024, encryption_type = lsb, size = 2>
!ct_L1_D2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1x1024,
                                encoding = #inverse_canonical_encoding>,
                                ciphertext_space = #ciphertext_space_L1_D2,
                                key = #key,
                                modulus_chain = #modulus_chain_L5_C1>

// KSK type
!rns_L2 = !rns.rns<!Zq0, !Zq1, !Zp0>
#ring_L2x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_L2x1024, encryption_type = lsb, size = 2>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L2x1024, encoding = #inverse_canonical_encoding>,
                             ciphertext_space = #ciphertext_space_L2,
                             key = #key>

module attributes {
    ckks.schemeParam = #ckks.scheme_param<
      logN = 10,
      Q = [1095233372161, 1032955396097],
      P = [261405424692085787],
      logDefaultScale = 45
    >
}
{
  // CHECK: func.func @test_relin(
  // CHECK-SAME: [[X:%.+]]: !ct_L1_D3,
  // CHECK-SAME: [[ksk:%.+]]: tensor<2x!ct_L2>) -> !ct_L1 {
  func.func @test_relin(%ct: !ct_L1_D3, %arg0: tensor<2x!ct_L2>) -> !ct_L1_D2 {
    // CHECK-DAG: [[C0:%.+]] = lwe.extract_coeff [[X]] {index = 0 : index}
    // CHECK-DAG: [[C1:%.+]] = lwe.extract_coeff [[X]] {index = 1 : index}
    // CHECK-DAG: [[C2:%.+]] = lwe.extract_coeff [[X]] {index = 2 : index}
    // CHECK-DAG: [[ksConstTerm:%.+]], [[ksLinearTerm:%.+]] = ckks.key_switch_inner [[C2]], [[ksk]]
    // CHECK-DAG: [[ksct:%.+]] = lwe.from_coeffs [[ksConstTerm]], [[ksLinearTerm]]
    // CHECK-DAG: [[subct:%.+]] = lwe.from_coeffs [[C0]], [[C1]]
    // CHECK-DAG: [[result:%.+]] = ckks.add [[ksct]], [[subct]]
    // CHECK-NEXT: return [[result]]
    %ct_0 = ckks.relinearize %ct, %arg0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3, tensor<2x!ct_L2>) -> !ct_L1_D2
    return %ct_0 : !ct_L1_D2
  }
}
