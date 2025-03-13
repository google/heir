// RUN: heir-opt --polynomial-to-mod-arith --cse %s | FileCheck %s

// This follows from example 3.8 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=1925:i32, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-DAG: #[[ID_MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[C_DIV_MAP:.*]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG: #[[ADD_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[MUL_MAP:.*]] = affine_map<(d0, d1) -> (d0 * d1)>
// CHECK-DAG: #[[DIV_MAP:.*]] = affine_map<(d0, d1) -> (d0 floordiv d1)>
// CHECK-DAG: #[[ADD_DIV_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 floordiv 2)>
// CHECK-DAG: #[[ROOT_MAP:.*]] = affine_map<(d0, d1) -> ((d0 * 2 + 1) * d1)>

// CHECK:     func.func @lower_ntt() -> [[OUTPUT_TYPE:.*]] {
// CHECK:      %[[COEFFS:.*]] = arith.constant dense<[1, 2, 3, 4]> : [[INT_TYPE:.*]]
// CHECK:      %[[COEFFS_ENC:.*]] = mod_arith.encapsulate %[[COEFFS]] : [[INT_TYPE]] -> [[MOD_TYPE:.*]]
// CHECK:      %[[REVERSE_BIT_ORDER_COEFFS:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xindex>
// CHECK:      %[[OUTPUT_VEC:.*]] = arith.constant dense<0> : [[INT_TYPE]]
// CHECK:      %[[OUTPUT_ENC:.*]] = mod_arith.encapsulate %[[OUTPUT_VEC]] : [[INT_TYPE]] -> [[MOD_TYPE]]
// CHECK:      %[[ORDERED_INPUT:.*]] = linalg.generic {indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]], iterator_types = ["parallel"]}
// CHECK-SAME:   ins(%[[REVERSE_BIT_ORDER_COEFFS]] : tensor<4xindex>) outs(%[[OUTPUT_ENC]] : [[MOD_TYPE]]) {
// CHECK:       ^bb0(%[[REV_INDEX:.*]]: index, %[[OUT:.*]]: [[COEFF_TYPE:!Z7681_i32]]):
// CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[COEFFS_ENC]][%[[REV_INDEX]]] : [[MOD_TYPE]]
// CHECK:         linalg.yield %[[EXTRACTED]] : [[COEFF_TYPE]]
// CHECK:       } -> [[MOD_TYPE]]

// CHECK-DAG:  %[[INITIAL_VALUE:.*]] = mod_arith.reduce %[[ORDERED_INPUT]] : [[MOD_TYPE]]
// CHECK-DAG:  %[[ROOTS:.*]] = arith.constant dense<[1, 1925, 3383, 6468]> : [[INT_TYPE]]
// CHECK-DAG:  %[[ROOTS_ENC:.*]] = mod_arith.encapsulate %[[ROOTS]] : [[INT_TYPE]] -> [[MOD_TYPE]]

// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[N:.*]] = arith.constant 4 : index

// CHECK:        %[[RES:.]]:3 = affine.for %[[_:.*]] = 0 to 2
// CHECK-SAME:     iter_args(%[[TARGET:.*]] = %[[INITIAL_VALUE]], %[[BATCH_SIZE:.*]] = %[[TWO]], %[[ROOT_EXP:.*]] = %[[TWO]]) -> ([[MOD_TYPE]], index, index) {
// CHECK:          %[[INNER_RES:.]] = affine.for %[[INDEX:.*]] = #[[ID_MAP]](%[[ZERO]]) to #[[DIV_MAP]](%[[N]], %[[BATCH_SIZE]])
// CHECK-SAME:       iter_args(%[[INNER_TARGET:.*]] = %[[TARGET]]) -> ([[MOD_TYPE]]) {
// CHECK:            %[[INDEX_K:.*]] = affine.apply #[[MUL_MAP]](%[[BATCH_SIZE]], %[[INDEX]])
// CHECK:            %[[ARITH_RES:.*]] = affine.for %[[INDEX_J:.*]] = #[[ID_MAP]](%[[ZERO]]) to #[[C_DIV_MAP]](%[[BATCH_SIZE]])
// CHECK-SAME:         iter_args(%[[ARITH_TARGET:.*]] = %[[INNER_TARGET]]) -> ([[MOD_TYPE]]) {
// CHECK:              %[[INDEX_A:.*]] = affine.apply #[[ADD_MAP]](%[[INDEX_J]], %[[INDEX_K]])
// CHECK:              %[[A:.*]] = tensor.extract %[[ARITH_TARGET]][%[[INDEX_A]]] : [[MOD_TYPE]]

// CHECK:              %[[INDEX_B:.*]] = affine.apply #[[ADD_DIV_MAP]](%[[INDEX_A]], %[[BATCH_SIZE]])
// CHECK:              %[[B:.*]] = tensor.extract %[[ARITH_TARGET]][%[[INDEX_B]]] : [[MOD_TYPE]]

// CHECK:              %[[ROOT_INDEX:.*]] = affine.apply #[[ROOT_MAP]](%[[INDEX_J]], %[[ROOT_EXP]])
// CHECK:              %[[ROOT:.*]] = tensor.extract %[[ROOTS_ENC]][%[[ROOT_INDEX]]] : [[MOD_TYPE]]

// CHECK:              %[[ROOTSB:.*]] = mod_arith.mul %[[B]], %[[ROOT]] : [[COEFF_TYPE]]
// CHECK:              %[[CTPLUS:.*]] = mod_arith.add %[[A]], %[[ROOTSB]] : [[COEFF_TYPE]]
// CHECK:              %[[CTMINUS:.*]] = mod_arith.sub %[[A]], %[[ROOTSB]] : [[COEFF_TYPE]]

// CHECK:              %[[INSERT_PLUS:.*]] = tensor.insert %[[CTPLUS]] into %[[ARITH_TARGET]][%[[INDEX_A]]] : [[MOD_TYPE]]
// CHECK:              %[[INSERT_MINUS:.*]] = tensor.insert %[[CTMINUS]] into %[[INSERT_PLUS]][%[[INDEX_B]]] : [[MOD_TYPE]]
// CHECK:              affine.yield %[[INSERT_MINUS]] : [[MOD_TYPE]]

// CHECK:            affine.yield %[[ARITH_RES]] : [[MOD_TYPE]]

// CHECK:          %[[NEXT_BATCH_SIZE:.*]] = arith.muli %[[BATCH_SIZE]], %[[TWO]] : index
// CHECK:          %[[NEXT_ROOT_EXP:.*]] = arith.divui %[[ROOT_EXP]], %[[TWO]] : index
// CHECK:          affine.yield %[[INNER_RES]], %[[NEXT_BATCH_SIZE]], %[[NEXT_ROOT_EXP]] : [[MOD_TYPE]], index, index

// CHECK:       %[[RES_CAST:.*]] = tensor.cast %[[RES]]#0 : [[MOD_TYPE]] to [[OUTPUT_TYPE]]
// CHECK:       return %[[RES_CAST]] : [[OUTPUT_TYPE]]

func.func @lower_ntt() -> tensor<4x!coeff_ty, #ring> {
  %coeffsRaw = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %poly = polynomial.from_tensor %coeffs : tensor<4x!coeff_ty> -> !poly_ty
  %ret = polynomial.ntt %poly {root=#root} : !poly_ty -> tensor<4x!coeff_ty, #ring>
  return %ret : tensor<4x!coeff_ty, #ring>
}
