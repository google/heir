// RUN: heir-opt --lwe-decompose %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zp0 = !mod_arith.int<261405424692085787 : i64>

#key = #lwe.key<>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>

// Output type
#ring_L1x1024 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!ringelt_L1 = !lwe.lwe_ring_elt<ring = #ring_L1x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_L1x1024, encryption_type = lsb, size = 2>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1x1024, encoding = #inverse_canonical_encoding>,
                             ciphertext_space = #ciphertext_space_L1,
                             key = #key>

// KSK type
!rns_L2 = !rns.rns<!Zq0, !Zp0>
#ring_L2x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_L2x1024, encryption_type = lsb, size = 2>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1x1024, encoding = #inverse_canonical_encoding>,
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
  // CHECK: func.func @test_moddown(
  // CHECK-SAME: [[x:%.+]]: [[kskTy:!.+]]) -> [[outTy:!.+]] {
  func.func @test_moddown(%x: !ct_L2) -> !ct_L1 {
    // CHECK-DAG: [[rnsConst:%.+]] = rns.constant <[#mod_arith.value<60017030419 : !Z1095233372161_i64> : !Z1095233372161_i64]> : !rns_L0
    // CHECK-DAG: [[constTerm:%.+]] = lwe.extract_coeff %ct {index = 0 : index} : [[kskTy]]
    // CHECK-DAG: [[constTermQ:%.+]] = lwe.extract_slice [[constTerm]] {size = 1 : index, start = 0 : index} : [[kskrng:!.+]] -> [[qrng:!.+]]
    // CHECK-DAG: [[constTermP:%.+]] = lwe.extract_slice [[constTerm]] {size = 1 : index, start = 1 : index} : [[kskrng]] -> [[prng:!.+]]
    // CHECK-DAG: [[const_ext:%.+]] = lwe.convert_basis [[constTermP]] {targetBasis = !rns_L0} : [[prng]] -> [[qrng]]
    // CHECK-DAG: [[linearTerm:%.+]] = lwe.extract_coeff %ct {index = 1 : index} : [[kskTy]]
    // CHECK-DAG: [[linearTermQ:%.+]] = lwe.extract_slice [[linearTerm]] {size = 1 : index, start = 0 : index} : [[kskrng]] -> [[qrng]]
    // CHECK-DAG: [[linearTermP:%.+]] = lwe.extract_slice [[linearTerm]] {size = 1 : index, start = 1 : index} : [[kskrng]] -> [[prng]]
    // CHECK-DAG: [[linear_ext:%.+]] = lwe.convert_basis [[linearTermP]] {targetBasis = !rns_L0} : [[prng]] -> [[qrng]]
    // CHECK-DAG: [[ctq:%.+]] = lwe.from_coeffs [[constTermQ]], [[linearTermQ]] : ([[qrng]], [[qrng]]) -> [[outTy]]
    // CHECK-DAG: [[ctp:%.+]] = lwe.from_coeffs [[const_ext]], [[linear_ext]] : ([[qrng]], [[qrng]]) -> [[outTy]]
    // CHECK-DAG: [[diff:%.+]] = lwe.rsub [[ctq]], [[ctp]] : ([[outTy]], [[outTy]]) -> [[outTy]]
    // CHECK-DAG: [[result:%.+]] = lwe.mul_scalar [[diff]], [[rnsConst]] : ([[outTy]], !rns_L0) -> [[outTy]]
    // CHECK-DAG: return [[result]] : [[outTy]]
    %result = lwe.mod_down %x {targetBasis = !rns.rns<!Zq0>} : !ct_L2 -> !ct_L1
    return %result: !ct_L1
  }
}
