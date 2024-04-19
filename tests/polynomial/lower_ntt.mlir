// RUN: heir-opt --polynomial-to-standard --cse %s | FileCheck %s

// This follows from example 3.8 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #_polynomial.polynomial<1 + x**4>
#ring = #_polynomial.ring<cmod=7681, ideal=#cycl, root=1925>
!poly_ty = !_polynomial.polynomial<#ring>

// CHECK:     func.func @lower_ntt() -> [[OUTPUT_TYPE:.*]] {
// CHECK-DAG:   %[[COEFFS:.*]] = arith.constant dense<[1, 2, 3, 4]> : [[INPUT_TYPE:.*]]
// CHECK-DAG:   %[[COEFFS_EXT:.*]] = arith.extui %cst : [[INPUT_TYPE]] to [[INTER_TYPE:.*]]
// CHECK-DAG:   %[[CMOD_VEC1:.*]] = arith.constant dense<7681> : [[ITER_TYPE1:tensor<2xi26>]]
// CHECK-DAG:   %[[ROOTS1:.*]] = arith.constant dense<[1925, 6468]> : [[ITER_TYPE1:.*]]
// CHECK-DAG:   %[[CMOD_VEC2:.*]] = arith.constant dense<7681> : [[ITER_TYPE2:tensor<1xi26>]]
// CHECK-DAG:   %[[ROOTS2:.*]] = arith.constant dense<3383> : [[ITER_TYPE2:.*]]

// CHECK:       %[[IDX0:.*]] = arith.constant 0 : index
// CHECK:       %[[EX0:.*]] = tensor.extract %[[COEFFS_EXT]][%[[IDX0]]] : [[INTER_TYPE]]
// CHECK:       %[[ELEM0:.*]] = tensor.from_elements %[[EX0]] : [[ITER_TYPE2]]

// CHECK:       %[[IDX1:.*]] = arith.constant 2 : index
// CHECK:       %[[EX1:.*]] = tensor.extract %[[COEFFS_EXT]][%[[IDX1]]] : [[INTER_TYPE]]
// CHECK:       %[[ELEM1:.*]] = tensor.from_elements %[[EX1]] : [[ITER_TYPE2]]

// CHECK:       %[[ROOTSB2:.*]] = arith.muli %[[ELEM1]], %[[ROOTS2]] : [[ITER_TYPE2]]
// CHECK:       %[[ROOTSB_MOD2:.*]] = arith.remui %[[ROOTSB2]], %[[CMOD_VEC2]] : [[ITER_TYPE2]]

// CHECK:       %[[CTPLUS2:.*]] = arith.addi %[[ELEM0]], %[[ROOTSB_MOD2]] : [[ITER_TYPE2]]
// CHECK:       %[[CTPLUS_MOD2:.*]] = arith.remui %[[CTPLUS2]], %[[CMOD_VEC2]] : [[ITER_TYPE2]]

// CHECK:       %[[CTMINUS2:.*]] = arith.subi %[[ELEM0]], %[[ROOTSB_MOD2]] : [[ITER_TYPE2]]
// CHECK:       %[[CTMINUS_SHIFT2:.*]] = arith.addi %[[CTMINUS2]], %[[CMOD_VEC2]] : [[ITER_TYPE2]]
// CHECK:       %[[CTMINUS_MOD2:.*]] = arith.remui %[[CTMINUS_SHIFT2]], %[[CMOD_VEC2]] : [[ITER_TYPE2]]

// CHECK:       %[[CONCAT_EMPTY2:.*]] = tensor.empty() : [[ITER_TYPE1]]
// CHECK:       %[[INSERT_TOP2:.*]] = tensor.insert_slice %[[CTPLUS_MOD2]] into %[[CONCAT_EMPTY2]][0] [1] [1] : [[ITER_TYPE2]] into [[ITER_TYPE1]]
// CHECK:       %[[EVENS_RES_UNCAST:.*]] = tensor.insert_slice %[[CTMINUS_MOD2]] into %[[INSERT_TOP2]][1] [1] [1] : [[ITER_TYPE2]] into [[ITER_TYPE1]]
// CHECK:       %[[EVENS_RES:.*]] = tensor.cast %[[EVENS_RES_UNCAST]]

// Iteration on the odds is removed for brevity but it follows the same as above
// CHECK:       %[[ODDS_RES_UNCAST:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[1] [1] [1] : [[ITER_TYPE2]] into [[ITER_TYPE1]]
// CHECK:       %[[ODDS_RES:.*]] = tensor.cast %[[ODDS_RES_UNCAST]]

// Merge the iterations:
// CHECK:       %[[ROOTSB1:.*]] = arith.muli %[[ODDS_RES]], %[[ROOTS1]] : [[ITER_TYPE1]]
// CHECK:       %[[ROOTSB_MOD1:.*]] = arith.remui %[[ROOTSB1]], %[[CMOD_VEC1]] : [[ITER_TYPE1]]

// CHECK:       %[[CTPLUS1:.*]] = arith.addi %[[EVENS_RES]], %[[ROOTSB_MOD1]] : [[ITER_TYPE1]]
// CHECK:       %[[CTPLUS_MOD1:.*]] = arith.remui %[[CTPLUS1]], %[[CMOD_VEC1]] : [[ITER_TYPE1]]

// CHECK:       %[[CTMINUS1:.*]] = arith.subi %[[EVENS_RES]], %[[ROOTSB_MOD1]] : [[ITER_TYPE1]]
// CHECK:       %[[CTMINUS_SHIFT1:.*]] = arith.addi %[[CTMINUS1]], %[[CMOD_VEC1]] : [[ITER_TYPE1]]
// CHECK:       %[[CTMINUS_MOD1:.*]] = arith.remui %[[CTMINUS_SHIFT1]], %[[CMOD_VEC1]] : [[ITER_TYPE1]]

// CHECK:       %[[CONCAT_EMPTY1:.*]] = tensor.empty() : [[INTER_TYPE]]
// CHECK:       %[[INSERT_TOP1:.*]] = tensor.insert_slice %[[CTPLUS_MOD1]] into %[[CONCAT_EMPTY1]][0] [2] [1] : [[ITER_TYPE1]] into [[INTER_TYPE]]
// CHECK:       %[[RES_UNCAST:.*]] = tensor.insert_slice %[[CTMINUS_MOD1]] into %[[INSERT_TOP1]][2] [2] [1] : [[ITER_TYPE1]] into [[INTER_TYPE]]
// CHECK:       %[[RES:.*]] = tensor.cast %[[RES_UNCAST]]
// CHECK:       %[[RES_TRUNC:.*]] = arith.trunci %[[RES]] : [[INTER_TYPE]] to [[INPUT_TYPE]]
// CHECK:       %[[RES_CAST:.*]] = tensor.cast %[[RES_TRUNC]] : [[INPUT_TYPE]] to [[OUTPUT_TYPE]]
// CHECK:       return %[[RES_CAST]] : [[OUTPUT_TYPE]]

func.func @lower_ntt() -> tensor<4xi13, #ring> {
  %coeffs = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi13>
  %poly = _polynomial.from_tensor %coeffs : tensor<4xi13> -> !poly_ty
  %ret = _polynomial.ntt %poly : !poly_ty -> tensor<4xi13, #ring>
  return %ret : tensor<4xi13, #ring>
}
