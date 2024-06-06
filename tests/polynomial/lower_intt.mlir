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

// CHECK:     func.func @lower_intt() -> [[OUTPUT_TYPE:.*]] {
// CHECK:      %[[COEFFS:.*]] = arith.constant dense<[1, 2, 3, 4]> : [[INPUT_TYPE:.*]]
// CHECK:      %[[CAST:.*]] = tensor.cast %[[COEFFS]] : [[INPUT_TYPE]] to [[OUTPUT_TYPE]]
// CHECK-DAG:  %[[INITIAL_VALUE:.*]] = arith.extui %[[CAST]] : [[OUTPUT_TYPE]] to [[INTER_TYPE:.*]]
// CHECK-DAG:  %[[CMOD:.*]] = arith.constant 7681 : [[ELEM_TYPE:i64]]
// CHECK-DAG:  %[[ROOTS:.*]] = arith.constant dense<[1, 1213, 4298, 5756]> : [[INTER_TYPE]]

// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[N:.*]] = arith.constant 4 : index

// CHECK:        %[[RES:.]]:3 = affine.for %[[_:.*]] = 0 to 2
// CHECK-SAME:     iter_args(%[[TARGET:.*]] = %[[INITIAL_VALUE]], %[[BATCH_SIZE:.*]] = %[[N]], %[[ROOT_EXP:.*]] = %[[ONE]]) -> ([[INTER_TYPE]], index, index) {
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

// CHECK:              %[[APLUSB:.*]] = arith.addi %[[A]], %[[B]] : [[ELEM_TYPE]]
// CHECK:              %[[GSPLUS:.*]] = arith.remui %[[APLUSB]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[AMINUSB:.*]] = arith.subi %[[A]], %[[B]] : [[ELEM_TYPE]]
// CHECK:              %[[AMINUSB_SHIFT:.*]] = arith.addi %[[AMINUSB]], %[[CMOD]] : [[ELEM_TYPE]]
// CHECK:              %[[AMINUSB_MOD:.*]] = arith.remui %[[AMINUSB_SHIFT]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[ROOTS_MUL:.*]] = arith.muli %[[AMINUSB_MOD]], %[[ROOT]] : [[ELEM_TYPE]]
// CHECK:              %[[GSMINUS:.*]] = arith.remui %[[ROOTS_MUL]], %[[CMOD]] : [[ELEM_TYPE]]

// CHECK:              %[[INSERT_PLUS:.*]] = tensor.insert %[[GSPLUS]] into %[[ARITH_TARGET]][%[[INDEX_A]]] : [[INTER_TYPE]]
// CHECK:              %[[INSERT_MINUS:.*]] = tensor.insert %[[GSMINUS]] into %[[INSERT_PLUS]][%[[INDEX_B]]] : [[INTER_TYPE]]
// CHECK:              affine.yield %[[INSERT_MINUS]] : [[INTER_TYPE]]

// CHECK:            affine.yield %[[ARITH_RES]] : [[INTER_TYPE]]

// CHECK:          %[[NEXT_BATCH_SIZE:.*]] = arith.divui %[[BATCH_SIZE]], %[[TWO]] : index
// CHECK:          %[[NEXT_ROOT_EXP:.*]] = arith.muli %[[ROOT_EXP]], %[[TWO]] : index
// CHECK:          affine.yield %[[INNER_RES]], %[[NEXT_BATCH_SIZE]], %[[NEXT_ROOT_EXP]] : [[INTER_TYPE]], index, index

// CHECK:       %[[N_INV_VEC:.*]] = arith.constant dense<5761> : [[INTER_TYPE]]
// CHECK:       %[[CMOD_VEC:.*]] = arith.constant dense<7681> : [[INTER_TYPE]]
// CHECK:       %[[RES_MUL:.*]] = arith.muli %[[RES]]#0, %[[N_INV_VEC]] : [[INTER_TYPE]]
// CHECK:       %[[RES_MOD:.*]] = arith.remui %[[RES_MUL]], %[[CMOD_VEC]] : [[INTER_TYPE]]
// CHECK:       %[[RES_TRUNC:.*]] = arith.trunci %[[RES_MOD]] : [[INTER_TYPE]] to [[OUTPUT_TYPE]]

// CHECK:      %[[REVERSE_BIT_ORDER_COEFFS:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xindex>
// CHECK:      %[[OUTPUT_VEC:.*]] = arith.constant dense<0> : [[OUTPUT_TYPE]]
// CHECK:      %[[ORDERED_OUTPUT:.*]] = linalg.generic {indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]], iterator_types = ["parallel"]}
// CHECK-SAME:   ins(%[[REVERSE_BIT_ORDER_COEFFS]] : tensor<4xindex>) outs(%[[OUTPUT_VEC]] : [[OUTPUT_TYPE]]) {
// CHECK:       ^bb0(%[[REV_INDEX:.*]]: index, %[[OUT:.*]]: i32):
// CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[RES_TRUNC]][%[[REV_INDEX]]] : [[OUTPUT_TYPE]]
// CHECK:         linalg.yield %[[EXTRACTED]] : i32
// CHECK:       } -> [[OUTPUT_TYPE]]

// CHECK:       return %[[ORDERED_OUTPUT]] : [[OUTPUT_TYPE]]

func.func @lower_intt() -> !poly_ty {
  %ntt_coeffs = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32, #ring>
  %ret = polynomial.intt %ntt_coeffs {root=#root} : tensor<4xi32, #ring> -> !poly_ty
  return %ret : !poly_ty
}
