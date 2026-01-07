// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK: !Z65536_i32 = !mod_arith.int<65536 : i32>
// CHECK: #[[LHS_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[RHS_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @lower_poly_mul(%[[poly0:.*]]: [[INPUT_TENSOR_TY:tensor<1024x!Z65536_i32>]], %[[poly1:.*]]: [[INPUT_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] {
// CHECK:      %[[NAIVE_POLYMUL_OUTPUT_STORAGE:.*]] = arith.constant dense<0> : [[NAIVE_POLYMUL_TENSOR_TY_STORAGE:tensor<2047xi32>]]
// CHECK:      %[[NAIVE_POLYMUL_OUTPUT:.*]] = mod_arith.encapsulate %[[NAIVE_POLYMUL_OUTPUT_STORAGE]] : [[NAIVE_POLYMUL_TENSOR_TY_STORAGE]] -> [[NAIVE_POLYMUL_TENSOR_TY:tensor<2047x!Z65536_i32>]]
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[INPUT_TENSOR_TY]], [[INPUT_TENSOR_TY]])
// CHECK-SAME:     outs(%[[NAIVE_POLYMUL_OUTPUT]] : [[NAIVE_POLYMUL_TENSOR_TY]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: [[COEFF_TY:!Z65536_i32]], %[[RHS_IN:.*]]: [[COEFF_TY]], %[[OUT:.*]]: [[COEFF_TY]]):
// CHECK:       %[[MULTED:.*]] = mod_arith.mul %[[LHS_IN]], %[[RHS_IN]]
// CHECK:       %[[SUMMED:.*]] = mod_arith.add %[[MULTED]], %[[OUT]]
// CHECK:       linalg.yield %[[SUMMED]]
// CHECK:     } -> [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:     %[[MODDED_RESULT:.*]] = call @__heir_poly_mod_2047x_65536_i32_1_x1024(%[[GENERIC_RESULT]]) : ([[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]]
// CHECK:     return %[[MODDED_RESULT]]
// CHECK: }

// CHECK: func.func private @__heir_poly_mod_2047x_65536_i32_1_x1024(%[[MOD_ARG0:.*]]: [[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:    %[[c1_modarith:.*]] = mod_arith.constant 1 : [[COEFF_TY]]
// CHECK:    %[[c1024:.*]] = arith.constant 1024 : index
// CHECK:    %[[rem_result:.*]] = scf.while (%[[WHILE_ARG1:.*]] = %[[MOD_ARG0]]) : ([[NAIVE_POLYMUL_TENSOR_TY]]) -> [[NAIVE_POLYMUL_TENSOR_TY]] {
// CHECK:      %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK:      %[[c1:.*]] = arith.constant 1 : index
// CHECK:      %[[c2046:.*]] = arith.constant 2046 : index

                // Don't need to re-test the lowering of leading_term in detail
// CHECK:      %[[DEG1:.*]] = scf.while
// CHECK:        tensor.extract
// CHECK:        arith.cmpi eq
// CHECK:        scf.condition
// CHECK:      } do {
// CHECK:        arith.subi
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      %[[LC1:.*]] = tensor.extract %[[WHILE_ARG1]][%[[DEG1]]]
// CHECK:      %[[CMP1:.*]] = arith.cmpi sge, %[[DEG1]], %[[c1024]] : index
// CHECK:      scf.condition(%[[CMP1]]) %[[WHILE_ARG1]]
// CHECK:    } do {
// CHECK:    ^[[bb0:.*]](%[[DIVIDEND:.*]]: [[NAIVE_POLYMUL_TENSOR_TY]]):
// CHECK:      %[[DIVISOR:.*]] = arith.constant dense<"0x[[DIVISOR_COEFFS:010*10*]]"> : tensor<2047xi32>
// CHECK:      %[[DIVISOR_MODARITH:.*]] = mod_arith.encapsulate %[[DIVISOR]]
// CHECK:      %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK:      %[[c1:.*]] = arith.constant 1 : index
// CHECK:      %[[c2046:.*]] = arith.constant 2046 : index
// CHECK:      %[[v1:.*]] = scf.while (%[[arg2:.*]] = %[[c2046]]) : (index) -> index {
// CHECK:        %[[extracted_4:.*]] = tensor.extract %[[DIVIDEND]][%[[arg2]]] : [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:        %[[extracted_4_reduced:.*]] = mod_arith.reduce %[[extracted_4]]
// CHECK:        %[[extracted_4_conv:.*]] = mod_arith.extract %[[extracted_4_reduced]]
// CHECK:        %[[cmp1:.*]] = arith.cmpi eq, %[[extracted_4_conv]], %[[c0_i32]] : i32
// CHECK:        scf.condition(%[[cmp1]]) %[[arg2]] : index
// CHECK:      } do {
// CHECK:      ^bb0(%[[arg2:.*]]: index):
// CHECK:        %[[sub1:.*]] = arith.subi %[[arg2]], %[[c1]] : index
// CHECK:        scf.yield %[[sub1]] : index
// CHECK:      }
// CHECK:      %[[extracted:.*]] = tensor.extract %[[DIVIDEND]][%[[v1]]] : [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:      %[[v2:.*]] = arith.subi %[[v1]], %[[c1024]] : index
// CHECK:      %[[extracted_mul:.*]] = mod_arith.mul %[[extracted]], %[[c1_modarith]] : [[COEFF_TY]]
// CHECK:      %[[extracted_mul_extracted:.*]] = mod_arith.extract %[[extracted_mul]]
// CHECK:      %[[splat:.*]] = tensor.splat %[[extracted_mul_extracted]] : tensor<2047xi32>
// CHECK:      %[[splat_enc:.*]] = mod_arith.encapsulate %[[splat]]
// CHECK:      %[[mul:.*]] = mod_arith.mul %[[DIVISOR_MODARITH]], %[[splat_enc]]
// CHECK:      %[[empty:.*]] = tensor.empty() : [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:      %[[c2047:.*]] = arith.constant 2047 : index
// CHECK:      %[[v5:.*]] = arith.subi %[[c2047]], %[[v2]] : index
// CHECK:      %[[extracted_slice_0:.*]] = tensor.extract_slice %[[mul]][0] [%[[v5]]] [1]
// CHECK:      %[[extracted_slice_1:.*]] = tensor.extract_slice %[[mul]][%[[v5]]] [%[[v2]]] [1]
// CHECK:      %[[inserted_slice:.*]] = tensor.insert_slice %[[extracted_slice_0]] into %[[empty]][%[[v2]]] [%[[v5]]] [1]
// CHECK:      %[[inserted_slice_2:.*]] = tensor.insert_slice %[[extracted_slice_1]] into %[[inserted_slice]][0] [%[[v2]]] [1]
// CHECK:      %[[add:.*]] = mod_arith.sub %[[DIVIDEND]], %[[inserted_slice_2]]
// CHECK:      scf.yield %[[add]]
// CHECK:    }
// CHECK:    %[[extracted_slice:.*]] = tensor.extract_slice %[[rem_result]][0] [1024] [1]
// CHECK:    return %[[extracted_slice]]
// CHECK:  }
// CHECK: }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly2 = polynomial.mul %poly0, %poly1 : !poly_ty
  return %poly2 : !poly_ty
}
