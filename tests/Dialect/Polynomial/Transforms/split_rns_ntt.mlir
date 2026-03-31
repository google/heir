// RUN: heir-opt --split-rns-ntt %s | FileCheck %s

!zp17 = !mod_arith.int<17 : i32>
!zp257 = !mod_arith.int<257 : i32>
!rns = !rns.rns<!zp17, !zp257>

#poly_mod = #polynomial.int_polynomial<-1 + x**8>
#rns_ring = #polynomial.ring<coefficientType=!rns, polynomialModulus=#poly_mod>
!rns_coeff_poly = !polynomial.polynomial<ring=#rns_ring, form=coeff>
!rns_eval_poly = !polynomial.polynomial<ring=#rns_ring, form=eval>

#root17 = #mod_arith.value<9 : !zp17>
#root257 = #mod_arith.value<4 : !zp257>
#rns_root = #rns.value<[#root17, #root257]>
#primitive_root = #polynomial.primitive_root<value=#rns_root, degree=16 : i32>

module {
  // CHECK: func.func @split_rns_intt(%[[ARG:.+]]: [[RNS_COEFF:![^ ]+]]) -> [[RNS_COEFF]] {
  // CHECK: %[[SLICE0_COEFF:.+]] = polynomial.extract_single_slice %[[ARG]] {index = 0 : index} : [[RNS_COEFF]] -> [[LIMB0_COEFF:![^ ]+]]
  // CHECK: %[[EVAL0:.+]] = polynomial.ntt %[[SLICE0_COEFF]] : [[LIMB0_COEFF]]
  // CHECK: %[[SLICE1_COEFF:.+]] = polynomial.extract_single_slice %[[ARG]] {index = 1 : index} : [[RNS_COEFF]] -> [[LIMB1_COEFF:![^ ]+]]
  // CHECK: %[[EVAL1:.+]] = polynomial.ntt %[[SLICE1_COEFF]] : [[LIMB1_COEFF]]
  // CHECK: %[[PACKED_EVAL:.+]] = polynomial.pack %[[EVAL0]], %[[EVAL1]] : [[LIMB0_EVAL:![^ ]+]], [[LIMB1_EVAL:![^ ]+]]
  // CHECK: %[[SLICE0_EVAL:.+]] = polynomial.extract_single_slice %[[PACKED_EVAL]] {index = 0 : index} : [[RNS_EVAL:![^ ]+]] -> [[LIMB0_EVAL]]
  // CHECK: %[[COEFF0:.+]] = polynomial.intt %[[SLICE0_EVAL]] {root = #polynomial.primitive_root<value = #mod_arith.value<9 : {{![^ ]+}}> : {{![^ ]+}}, degree = 8 : i32>} : [[LIMB0_EVAL]]
  // CHECK: %[[SLICE1_EVAL:.+]] = polynomial.extract_single_slice %[[PACKED_EVAL]] {index = 1 : index} : [[RNS_EVAL]] -> [[LIMB1_EVAL]]
  // CHECK: %[[COEFF1:.+]] = polynomial.intt %[[SLICE1_EVAL]] {root = #polynomial.primitive_root<value = #mod_arith.value<4 : {{![^ ]+}}> : {{![^ ]+}}, degree = 8 : i32>} : [[LIMB1_EVAL]]
  // CHECK: %[[PACKED_COEFF:.+]] = polynomial.pack %[[COEFF0]], %[[COEFF1]] : [[LIMB0_COEFF]], [[LIMB1_COEFF]]
  // CHECK: return %[[PACKED_COEFF]] : [[RNS_COEFF]]
  func.func @split_rns_intt(%arg0 : !rns_coeff_poly) -> !rns_coeff_poly {
    %0 = polynomial.ntt %arg0 : !rns_coeff_poly
    %1 = polynomial.intt %0 {root=#primitive_root} : !rns_eval_poly
    return %1 : !rns_coeff_poly
  }
}
