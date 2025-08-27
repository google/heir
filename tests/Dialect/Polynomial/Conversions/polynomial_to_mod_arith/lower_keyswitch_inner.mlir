// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

// Test lowering of polynomial.key_switch_inner operation

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
!rns_new = !rns.rns<!Z1095233372161_i64>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_new = #polynomial.ring<coefficientType = !rns_new, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring_rns_L1_1_x1024>
!poly_new = !polynomial.polynomial<ring = #ring_new>

// CHECK: @test_keyswitch_inner_lowering
func.func @test_keyswitch_inner_lowering(%p: !poly, %ksk: tensor<10x2x!poly>) -> (!poly, !poly) {
  // CHECK-NOT: polynomial.key_switch_inner
  //FIXME: Write a real test
  %c, %l = polynomial.key_switch_inner %p, %ksk : (!poly, tensor<10x2x!poly>) -> (!poly, !poly)
  return %c, %l : !poly, !poly
}

// Expected Code
func.func @test_keyswitch_inner_lowered(%x: !poly, %ksk: tensor<10x2x!poly>) -> (!poly, !poly) {
  %xAsTensor = polynomial.to_tensor %xParts : !poly -> tensor<Nx!rns_L1>
  // x has L RNS terms
  // L is number of normal primes
  %xParts = mod_arith.partition %xAsTensor {partition_size L/beta} : tensor<Nx!rns_L1> -> tensor<Nx??x!rns...>
  // might need to do %xParts, %xPartsLeftover?

  // There are beta parts
  // each part has <= k RNS components
  // k is the number of key-switch ("special") primes

  // FIXME: IMPLEMENT FBC, should emit the full polys not just the new extended parts
  %extPartsAsTensor = mod_arith.fast_basis_conversion %xParts TARGET_BASIS ??? : tensor<...> -> ??? // ElementWiseMappable Op
  %extParts = polynomial.from_tensor %extPartsAsTensor
  %extPartsNTT = polynomial.ntt %extParts : !poly -> !tensor...

  // KSKs have L+k RNS components, might need to drop some elements to match ctxts
  // However, we assume that our relin op has been provided with the correctly sized ksk inputs

  // There are beta KSKs,
  // extParts has beta polynomials, each with L+k RNS components
  // for i in range(beta):
  // sum += (extParts[i] * ksk[i][0], extParts[i] * ksk[i][1])
  %dotProdNTT = ???  : tensor

  // iNTT(sum)
  %dotProd = polynomial.intt %dotProdNTT : !tensor -> !poly //fake ElementWiseMappable

  // mod down to remove all keyswitch primes
  %rescaledDotProd = polynomial.mod_switch %dotProd : !poly -> !poly_new

  return %c, %l : !poly, !poly
}
