// RUN: heir-opt --polynomial-to-standard --split-input-file %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#cycl_2048>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK: #[[LHS_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[RHS_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @lower_poly_mul(%[[poly0:.*]]: [[INPUT_TENSOR_TY:tensor<1024xi32>]], %[[poly1:.*]]: [[INPUT_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] {
// CHECK:      %[[NAIVE_POLYMUL_OUTPUT:.*]] = arith.constant dense<0> : [[NAIVE_POLYMUL_TENSOR_TY:tensor<2047xi32>]]
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[INPUT_TENSOR_TY]], [[INPUT_TENSOR_TY]])
// CHECK-SAME:     outs(%[[NAIVE_POLYMUL_OUTPUT]] : [[NAIVE_POLYMUL_TENSOR_TY]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:       %[[LHS_EXT:.*]] = arith.extsi %[[LHS_IN]] : i32 to i64
// CHECK:       %[[RHS_EXT:.*]] = arith.extsi %[[RHS_IN]] : i32 to i64
// CHECK:       %[[OUT_EXT:.*]] = arith.extsi %[[OUT]] : i32 to i64
// CHECK:       %[[MULTED:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK:       %[[SUMMED:.*]] = arith.addi %[[MULTED]], %[[OUT_EXT]] : i64
// CHECK:       %[[RESULT:.*]] = arith.trunci %[[SUMMED]] : i64 to i32
// CHECK:       linalg.yield %[[RESULT]] : i32
// CHECK:     } -> [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:     %[[MODDED_RESULT:.*]] = call @__heir_poly_mod_4294967296_1_x1024(%[[GENERIC_RESULT]]) : ([[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]]
// CHECK:     return
// CHECK: }

// CHECK: func.func private @__heir_poly_mod_4294967296_1_x1024(%[[MOD_ARG0:.*]]: [[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:    %[[c_minus1:.*]] = arith.constant -1 : i32
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
// CHECK:      %[[DIVISOR:.*]] = arith.constant dense<"0x[[DIVISOR_COEFFS:010*10*]]"> : [[NAIVE_POLYMUL_TENSOR_TY]]

// CHECK:      %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK:      %[[c1:.*]] = arith.constant 1 : index
// CHECK:      %[[c2046:.*]] = arith.constant 2046 : index
// CHECK:      %[[v1:.*]] = scf.while (%[[arg2:.*]] = %[[c2046]]) : (index) -> index {
// CHECK:        %[[extracted_4:.*]] = tensor.extract %[[DIVIDEND]][%[[arg2]]] : tensor<2047xi32>
// CHECK:        %[[cmp1:.*]] = arith.cmpi eq, %[[extracted_4]], %[[c0_i32]] : i32
// CHECK:        scf.condition(%[[cmp1]]) %[[arg2]] : index
// CHECK:      } do {
// CHECK:      ^bb0(%[[arg2:.*]]: index):
// CHECK:        %[[sub1:.*]] = arith.subi %[[arg2]], %[[c1]] : index
// CHECK:        scf.yield %[[sub1]] : index
// CHECK:      }
// CHECK:      %[[extracted:.*]] = tensor.extract %[[DIVIDEND]][%[[v1]]] : tensor<2047xi32>
// CHECK:      %[[v2:.*]] = arith.subi %[[v1]], %[[c1024]] : index
// CHECK:      %[[splat:.*]] = tensor.splat %[[extracted]] : tensor<2047xi32>
// CHECK:      %[[mul:.*]] = arith.muli %cst, %[[splat]] : tensor<2047xi32>
// CHECK:      %[[empty:.*]] = tensor.empty() : tensor<2047xi32>
// CHECK:      %[[c2047:.*]] = arith.constant 2047 : index
// CHECK:      %[[v5:.*]] = arith.subi %[[c2047]], %[[v2]] : index
// CHECK:      %[[extracted_slice_0:.*]] = tensor.extract_slice %[[mul]][0] [%[[v5]]] [1] : tensor<2047xi32> to tensor<?xi32>
// CHECK:      %[[extracted_slice_1:.*]] = tensor.extract_slice %[[mul]][%[[v5]]] [%[[v2]]] [1] : tensor<2047xi32> to tensor<?xi32>
// CHECK:      %[[inserted_slice:.*]] = tensor.insert_slice %[[extracted_slice_0]] into %[[empty]][%[[v2]]] [%[[v5]]] [1] : tensor<?xi32> into tensor<2047xi32>
// CHECK:      %[[inserted_slice_2:.*]] = tensor.insert_slice %[[extracted_slice_1]] into %[[inserted_slice]][0] [%[[v2]]] [1] : tensor<?xi32> into tensor<2047xi32>
// CHECK:      %[[splat_3:.*]] = tensor.splat %[[c_minus1]] : tensor<2047xi32>
// CHECK:      %[[mul_2:.*]] = arith.muli %[[inserted_slice_2]], %[[splat_3]] : tensor<2047xi32>
// CHECK:      %[[add:.*]] = arith.addi %[[DIVIDEND]], %[[mul_2]] : tensor<2047xi32>
// CHECK:      scf.yield %[[add]] : tensor<2047xi32>
// CHECK:    }
// CHECK:    %[[extracted_slice:.*]] = tensor.extract_slice %[[rem_result]][0] [1024] [1] : tensor<2047xi32> to tensor<1024xi32>
// CHECK:    return %[[extracted_slice]] : tensor<1024xi32>
// CHECK:  }
// CHECK: }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly2 = polynomial.mul %poly0, %poly1 : !poly_ty
  return %poly2 : !poly_ty
}

// -----

// Test a non-machine-word sized modulus to ensure it remsi's correctly

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 18 : i32, polynomialModulus=#cycl_2048>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK: #[[LHS_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[RHS_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @lower_poly_mul(%[[poly0:.*]]: [[INPUT_TENSOR_TY:tensor<1024xi32>]], %[[poly1:.*]]: [[INPUT_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] {
// CHECK:      %[[POLYMUL_OUTPUT:.*]] = arith.constant dense<0> : [[POLYMUL_TENSOR_TY:tensor<2047xi32>]]
// CHECK:      %[[CMOD:.*]] = arith.constant 18 : i64
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[INPUT_TENSOR_TY]], [[INPUT_TENSOR_TY]])
// CHECK-SAME:     outs(%[[POLYMUL_OUTPUT]] : [[POLYMUL_TENSOR_TY]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:       %[[LHS_EXT:.*]] = arith.extsi %[[LHS_IN]] : i32 to i64
// CHECK:       %[[RHS_EXT:.*]] = arith.extsi %[[RHS_IN]] : i32 to i64
// CHECK:       %[[OUT_EXT:.*]] = arith.extsi %[[OUT]] : i32 to i64
// CHECK:       %[[MULTED:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK:       %[[SUMMED:.*]] = arith.addi %[[MULTED]], %[[OUT_EXT]] : i64
// CHECK:       %[[MODDED:.*]] = arith.remsi %[[SUMMED]], %[[CMOD]] : i64
// CHECK:       %[[ADDCMOD:.*]] = arith.addi %[[MODDED]], %[[CMOD]] : i64
// CHECK:       %[[CONGRUENT:.*]] = arith.remsi %[[ADDCMOD]], %[[CMOD]] : i64
// CHECK:       %[[RESULT:.*]] = arith.trunci %[[CONGRUENT]] : i64 to i32
// CHECK:       linalg.yield %[[RESULT]] : i32
// CHECK:     } -> [[POLYMUL_TENSOR_TY]]
// CHECK:     %[[MODDED_RESULT:.*]] = call @__heir_poly_mod_18_1_x1024(%[[GENERIC_RESULT]]) : ([[POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]]
// CHECK:     return
// CHECK: }

// CHECK: func.func private @__heir_poly_mod_18_1_x1024(%[[MOD_ARG0:.*]]: [[POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    // No need to re-test above
// CHECK: }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly = polynomial.mul %poly0, %poly1 : !poly_ty
  return %poly : !poly_ty
}
