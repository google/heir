// RUN: heir-opt --polynomial-to-standard --cse %s | FileCheck %s

// This follows from example 3.8 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

#cycl = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7681 : i32, polynomialModulus=#cycl>
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
// CHECK:      %[[COEFFS:.*]] = arith.constant dense<[1, 2, 3, 4]> : [[INPUT_TYPE:.*]]
// CHECK:      %[[REVERSE_BIT_ORDER_COEFFS:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xindex>
// CHECK:      %[[OUTPUT_VEC:.*]] = arith.constant dense<0> : [[INPUT_TYPE]]
// CHECK:      %[[ORDERED_INPUT:.*]] = linalg.generic {indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]], iterator_types = ["parallel"]}
// CHECK-SAME:   ins(%[[REVERSE_BIT_ORDER_COEFFS]] : tensor<4xindex>) outs(%[[OUTPUT_VEC]] : [[INPUT_TYPE]]) {
// CHECK:       ^bb0(%[[REV_INDEX:.*]]: index, %[[OUT:.*]]: i32):
// CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[COEFFS]][%[[REV_INDEX]]] : [[INPUT_TYPE]]
// CHECK:         linalg.yield %[[EXTRACTED]] : i32
// CHECK:       } -> [[INPUT_TYPE]]

// CHECK-DAG:  %[[INITIAL_VALUE:.*]] = arith.extui %[[ORDERED_INPUT]] : [[INPUT_TYPE]] to [[INTER_TYPE:.*]]
// CHECK-DAG:  %[[CMOD:.*]] = arith.constant 7681 : [[ELEM_TYPE:i64]]
// CHECK-DAG:  %[[ROOTS:.*]] = arith.constant dense<[1, 1925, 3383, 6468]> : [[INTER_TYPE]]

// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[N:.*]] = arith.constant 4 : index

// CHECK:        %[[RES:.]]:3 = affine.for %[[_:.*]] = 0 to 2
// CHECK-SAME:     iter_args(%[[TARGET:.*]] = %[[INITIAL_VALUE]], %[[BATCH_SIZE:.*]] = %[[TWO]], %[[ROOT_EXP:.*]] = %[[TWO]]) -> ([[INTER_TYPE]], index, index) {
// CHECK:          %[[INNER_RES:.]] = affine.for %[[INDEX:.*]] = #[[ID_MAP]](%[[ZERO]]) to #[[DIV_MAP]](%[[N]], %[[BATCH_SIZE]])
// CHECK-SAME:       iter_args(%[[INNER_TARGET:.*]] = %[[TARGET]]) -> ([[INTER_TYPE]]) {
// CHECK:            %[[INDEX_K:.*]] = affine.apply #[[MUL_MAP]](%[[BATCH_SIZE]], %[[INDEX]])
// CHECK:            %[[ARITH_RES:.*]] = affine.for %[[INDEX_J:.*]] = #[[ID_MAP]](%[[ZERO]]) to #[[C_DIV_MAP]](%[[BATCH_SIZE]])
// CHECK-SAME:         iter_args(%[[ARITH_TARGET:.*]] = %[[INNER_TARGET]]) -> ([[INTER_TYPE]]) {
// CHECK:              %[[INDEX_A:.*]] = affine.apply #[[ADD_MAP]](%[[INDEX_J]], %[[INDEX_K]])
// CHECK:              %[[A:.*]] = tensor.extract %[[ARITH_TARGET]][%[[INDEX_A]]] : [[INTER_TYPE]]
// CHECK:              %[[INDEX_B:.*]] = affine.apply #[[ADD_DIV_MAP]](%[[INDEX_A]], %[[BATCH_SIZE]])
// CHECK:              %[[B:.*]] = tensor.extract %[[ARITH_TARGET]][%[[INDEX_B]]] : [[INTER_TYPE]]
// CHECK:              %[[ROOT_INDEX:.*]] = affine.apply #[[ROOT_MAP]](%[[INDEX_J]], %[[ROOT_EXP]])
// CHECK:              %[[ROOT:.*]] = tensor.extract %[[ROOTS]][%[[ROOT_INDEX]]] : [[INTER_TYPE]]

// CHECK:              %[[ROOTSB:.*]] = arith.muli %[[B]], %[[ROOT]] : [[ELEM_TYPE]]
// CHECK:              %[[ROOTSB_MOD:.*]] = arith.remui %[[ROOTSB]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[CTPLUS:.*]] = arith.addi %[[A]], %[[ROOTSB_MOD]] : [[ELEM_TYPE]]
// CHECK:              %[[CTPLUS_MOD:.*]] = arith.remui %[[CTPLUS]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[CTMINUS:.*]] = arith.subi %[[A]], %[[ROOTSB_MOD]] : [[ELEM_TYPE]]
// CHECK:              %[[CTMINUS_SHIFT:.*]] = arith.addi %[[CTMINUS]], %[[CMOD]] : [[ELEM_TYPE]]
// CHECK:              %[[CTMINUS_MOD:.*]] = arith.remui %[[CTMINUS_SHIFT]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[INSERT_PLUS:.*]] = tensor.insert %[[CTPLUS_MOD]] into %[[ARITH_TARGET]][%[[INDEX_A]]] : [[INTER_TYPE]]
// CHECK:              %[[INSERT_MINUS:.*]] = tensor.insert %[[CTMINUS_MOD]] into %[[INSERT_PLUS]][%[[INDEX_B]]] : [[INTER_TYPE]]
// CHECK:              affine.yield %[[INSERT_MINUS]] : [[INTER_TYPE]]

// CHECK:            affine.yield %[[ARITH_RES]] : [[INTER_TYPE]]

// CHECK:          %[[NEXT_BATCH_SIZE:.*]] = arith.muli %[[BATCH_SIZE]], %[[TWO]] : index
// CHECK:          %[[NEXT_ROOT_EXP:.*]] = arith.divui %[[ROOT_EXP]], %[[TWO]] : index
// CHECK:          affine.yield %[[INNER_RES]], %[[NEXT_BATCH_SIZE]], %[[NEXT_ROOT_EXP]] : [[INTER_TYPE]], index, index

// CHECK:       %[[RES_TRUNC:.*]] = arith.trunci %[[RES]]#0 : [[INTER_TYPE]] to [[INPUT_TYPE]]
// CHECK:       %[[RES_CAST:.*]] = tensor.cast %[[RES_TRUNC]] : [[INPUT_TYPE]] to [[OUTPUT_TYPE]]
// CHECK:       return %[[RES_CAST]] : [[OUTPUT_TYPE]]

func.func @lower_ntt() -> tensor<4xi32, #ring> {
  %coeffs = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %poly = polynomial.from_tensor %coeffs : tensor<4xi32> -> !poly_ty
  %ret = polynomial.ntt %poly {root=#root} : !poly_ty -> tensor<4xi32, #ring>
  return %ret : tensor<4xi32, #ring>
}
