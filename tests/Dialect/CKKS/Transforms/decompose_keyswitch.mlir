// RUN: heir-opt --ckks-decompose-keyswitch %s --mlir-print-ir-after-failure | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<261405424692085787 : i64>

// Input's type
#ring_L1x1024 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!ringelt_L1 = !lwe.lwe_ring_elt<ring = #ring_L1x1024>

// KSK type
!rns_L2 = !rns.rns<!Zq0, !Zp0>
#ring_L2x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
// encryption_type probably doesn't make sense for KSKs
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_L2x1024, encryption_type = lsb, size = 2>
// encoding probably doesn't make sense for KSKs
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
// ModulusChain probably isn't appropriate for keyswitch keys. The problem is that in order to be a valid ciphertext, the modulus chain needs to include
// the key-switch primes, but these don't correspond to available "levels".
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
  // CHECK: func.func @test_keyswitch(
  // CHECK-SAME: [[x:%.+]]: !ringelt,
  // CHECK-SAME: [[ksk:%.+]]: tensor<1x!ct_L1>) -> (!ringelt, !ringelt) {
  func.func @test_keyswitch(%x: !ringelt_L1, %arg0: tensor<1x!ct_L2>) -> (!ringelt_L1, !ringelt_L1) {
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[part0:%.+]] = lwe.extract_slice [[x]] {size = 1 : index, start = 0 : index} : (!ringelt) -> !ringelt
    // !ringelt1 has the same LWERingElt type as the keyswitch key
    // CHECK-DAG: [[extPart0:%.+]] = lwe.convert_basis [[part0]] {targetBasis = !rns_L1} : (!ringelt) -> !ringelt1
    // CHECK-DAG: [[ksk0:%.+]] = tensor.extract [[ksk]][[[C0]]] : tensor<1x!ct_L1>
    // CHECK-DAG: [[dp:%.+]] = lwe.mul_ring_elt [[extPart0]], [[ksk0]] : (!ringelt1, !ct_L1) -> !ct_L1
    // CHECK-DAG: [[constTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 0 : index} : (!ct_L1) -> !ringelt1
    // CHECK-DAG: [[linearTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 1 : index} : (!ct_L1) -> !ringelt1
    // CHECK-DAG: [[const_ext:%.+]] = lwe.convert_basis [[constTerm]] {targetBasis = !rns_L0} : (!ringelt1) -> !ringelt
    // CHECK-DAG: [[linear_ext:%.+]] = lwe.convert_basis [[linearTerm]] {targetBasis = !rns_L0} : (!ringelt1) -> !ringelt
    // CHECK-DAG: return [[const_ext]], [[linear_ext]] : !ringelt, !ringelt
    %constTerm, %linearTerm = ckks.key_switch_inner %x, %arg0 : (!ringelt_L1, tensor<1x!ct_L2>) -> (!ringelt_L1, !ringelt_L1)
    return %constTerm, %linearTerm: !ringelt_L1, !ringelt_L1
  }
}
