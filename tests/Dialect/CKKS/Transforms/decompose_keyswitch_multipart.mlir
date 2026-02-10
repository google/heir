// RUN: heir-opt --ckks-decompose-keyswitch %s --mlir-print-ir-after-failure | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zq2 = !mod_arith.int<1005037682689 : i64>
!Zq3 = !mod_arith.int<998595133441 : i64>
!Zq4 = !mod_arith.int<972824936449 : i64>
!Zp0 = !mod_arith.int<261405424692085787 : i64>
!Zp1 = !mod_arith.int<959939837953 : i64>

// TEST: two full partitions, no partial partition
// Input's type
#ring_L1x1024 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
!ringelt_L1 = !lwe.lwe_ring_elt<ring = #ring_L1x1024>

// KSK type
!rns_L2 = !rns.rns<!Zq0, !Zq1, !Zp0>
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
      // TODO: make a ticket to convert these to integerAttrs
      Q = [1095233372161, 1032955396097],
      P = [261405424692085787],
      logDefaultScale = 45
    >
}
{
  // CHECK: func.func @test_keyswitch_2part(
  // CHECK-SAME: [[x:%.+]]: !ringelt,
  // CHECK-SAME: [[ksk:%.+]]: tensor<2x!ct_L2>) -> (!ringelt, !ringelt) {
  func.func @test_keyswitch_2part(%x: !ringelt_L1, %arg0: tensor<2x!ct_L2>) -> (!ringelt_L1, !ringelt_L1) {
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
    // CHECK-DAG: [[part0:%.+]] = lwe.extract_slice [[x]] {size = 1 : index, start = 0 : index} : (!ringelt) -> !ringelt1
    // CHECK-DAG: [[part1:%.+]] = lwe.extract_slice [[x]] {size = 1 : index, start = 1 : index} : (!ringelt) -> !ringelt2
    // CHECK-DAG: [[extPart0:%.+]] = lwe.convert_basis [[part0]] {targetBasis = !rns_L2} : (!ringelt1) -> !ringelt3
    // CHECK-DAG: [[extPart1:%.+]] = lwe.convert_basis [[part1]] {targetBasis = !rns_L2} : (!ringelt2) -> !ringelt3
    // CHECK-DAG: [[ksk0:%.+]] = tensor.extract [[ksk]][[[C0]]] : tensor<2x!ct_L2>
    // CHECK-DAG: [[ksk1:%.+]] = tensor.extract [[ksk]][[[C1]]] : tensor<2x!ct_L2>
    // CHECK-DAG: [[dp0:%.+]] = lwe.mul_ring_elt [[extPart0]], [[ksk0]] : (!ringelt3, !ct_L2) -> !ct_L2
    // CHECK-DAG: [[dp1:%.+]] = lwe.mul_ring_elt [[extPart1]], [[ksk1]] : (!ringelt3, !ct_L2) -> !ct_L2
    // CHECK-DAG: [[dp:%.+]] = lwe.radd [[dp0]], [[dp1]] : (!ct_L2, !ct_L2) -> !ct_L2
    // CHECK-DAG: [[constTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 0 : index} : (!ct_L2) -> !ringelt3
    // CHECK-DAG: [[linearTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 1 : index} : (!ct_L2) -> !ringelt3
    // CHECK-DAG: [[const_ext:%.+]] = lwe.convert_basis [[constTerm]] {targetBasis = !rns_L1} : (!ringelt3) -> !ringelt
    // CHECK-DAG: [[linear_ext:%.+]] = lwe.convert_basis [[linearTerm]] {targetBasis = !rns_L1} : (!ringelt3) -> !ringelt
    // CHECK-DAG: return [[const_ext]], [[linear_ext]] : !ringelt, !ringelt
    %constTerm, %linearTerm = ckks.key_switch_inner %x, %arg0 : (!ringelt_L1, tensor<2x!ct_L2>) -> (!ringelt_L1, !ringelt_L1)
    return %constTerm, %linearTerm: !ringelt_L1, !ringelt_L1
  }
}

// TEST: two full partitions, and a partial partition
#ring_L5x1024 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1, !Zq2, !Zq3, !Zq4>, polynomialModulus = <1 + x**1024>>
!ringelt_L5 = !lwe.lwe_ring_elt<ring = #ring_L5x1024>

// KSK type
!rns_L7 = !rns.rns<!Zq0, !Zq1, !Zq2, !Zq3, !Zq4, !Zp0, !Zp1>
#ring_L7x1024 = #polynomial.ring<coefficientType = !rns_L7, polynomialModulus = <1 + x**1024>>
// encryption_type probably doesn't make sense for KSKs
#ciphertext_space_L7 = #lwe.ciphertext_space<ring = #ring_L7x1024, encryption_type = lsb, size = 2>
// ModulusChain probably isn't appropriate for keyswitch keys. The problem is that in order to be a valid ciphertext, the modulus chain needs to include
// the key-switch primes, but these don't correspond to available "levels".
!ct_L7 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L7x1024, encoding = #inverse_canonical_encoding>,
                             ciphertext_space = #ciphertext_space_L7,
                             key = #key>

module attributes {
    ckks.schemeParam = #ckks.scheme_param<
      logN = 10,
      // TODO: make a ticket to convert these to integerAttrs
      Q = [1095233372161, 1032955396097, 1005037682689, 998595133441, 972824936449],
      P = [261405424692085787, 959939837953],
      logDefaultScale = 45
    >
}
{
  // CHECK: func.func @test_keyswitch_partialpart(
  // CHECK-SAME: [[x:%.+]]: !ringelt4,
  // CHECK-SAME: [[ksk:%.+]]: tensor<3x!ct_L6>) -> (!ringelt4, !ringelt4) {
  func.func @test_keyswitch_partialpart(%x: !ringelt_L5, %arg0: tensor<3x!ct_L7>) -> (!ringelt_L5, !ringelt_L5) {
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[part0:%.+]] = lwe.extract_slice [[x]] {size = 2 : index, start = 0 : index} : (!ringelt4) -> !ringelt
    // CHECK-DAG: [[part1:%.+]] = lwe.extract_slice [[x]] {size = 2 : index, start = 2 : index} : (!ringelt4) -> !ringelt5
    // CHECK-DAG: [[part2:%.+]] = lwe.extract_slice [[x]] {size = 1 : index, start = 4 : index} : (!ringelt4) -> !ringelt6
    // CHECK-DAG: [[extPart0:%.+]] = lwe.convert_basis [[part0]] {targetBasis = !rns_L6} : (!ringelt) -> !ringelt7
    // CHECK-DAG: [[extPart1:%.+]] = lwe.convert_basis [[part1]] {targetBasis = !rns_L6} : (!ringelt5) -> !ringelt7
    // CHECK-DAG: [[extPart2:%.+]] = lwe.convert_basis [[part2]] {targetBasis = !rns_L6} : (!ringelt6) -> !ringelt7
    // CHECK-DAG: [[ksk0:%.+]] = tensor.extract [[ksk]][[[C0]]] : tensor<3x!ct_L6>
    // CHECK-DAG: [[ksk1:%.+]] = tensor.extract [[ksk]][[[C1]]] : tensor<3x!ct_L6>
    // CHECK-DAG: [[ksk2:%.+]] = tensor.extract [[ksk]][[[C2]]] : tensor<3x!ct_L6>
    // CHECK-DAG: [[dp0:%.+]] = lwe.mul_ring_elt [[extPart0]], [[ksk0]] : (!ringelt7, !ct_L6) -> !ct_L6
    // CHECK-DAG: [[dp1:%.+]] = lwe.mul_ring_elt [[extPart1]], [[ksk1]] : (!ringelt7, !ct_L6) -> !ct_L6
    // CHECK-DAG: [[dp2:%.+]] = lwe.mul_ring_elt [[extPart2]], [[ksk2]] : (!ringelt7, !ct_L6) -> !ct_L6
    // CHECK-DAG: [[dpsum0:%.+]] = lwe.radd [[dp0]], [[dp1]] : (!ct_L6, !ct_L6) -> !ct_L6
    // CHECK-DAG: [[dp:%.+]] = lwe.radd [[dpsum0]], [[dp2]] : (!ct_L6, !ct_L6) -> !ct_L6
    // CHECK-DAG: [[constTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 0 : index} : (!ct_L6) -> !ringelt7
    // CHECK-DAG: [[linearTerm:%.+]] = lwe.extract_coeff [[dp]] {index = 1 : index} : (!ct_L6) -> !ringelt7
    // CHECK-DAG: [[const_ext:%.+]] = lwe.convert_basis [[constTerm]] {targetBasis = !rns_L4} : (!ringelt7) -> !ringelt4
    // CHECK-DAG: [[linear_ext:%.+]] = lwe.convert_basis [[linearTerm]] {targetBasis = !rns_L4} : (!ringelt7) -> !ringelt4
    // CHECK-DAG: return [[const_ext]], [[linear_ext]] : !ringelt4, !ringelt4
    %constTerm, %linearTerm = ckks.key_switch_inner %x, %arg0 : (!ringelt_L5, tensor<3x!ct_L7>) -> (!ringelt_L5, !ringelt_L5)
    return %constTerm, %linearTerm: !ringelt_L5, !ringelt_L5
  }
}
