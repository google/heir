// RUN: heir-opt --polynomial-to-standard --split-input-file %s | FileCheck %s

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=4294967296, ideal=#cycl_2048>
!poly_ty = !polynomial.polynomial<#ring>

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

// TODO(#551): clean up test so that variables names are captured and not hard-coded
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c2046 = arith.constant 2046 : index
// CHECK:      %1 = scf.while (%arg2 = %c2046) : (index) -> index {
// CHECK:        %extracted_4 = tensor.extract %arg1[%arg2] : tensor<2047xi32>
// CHECK:        %8 = arith.cmpi eq, %extracted_4, %c0_i32 : i32
// CHECK:        scf.condition(%8) %arg2 : index
// CHECK:      } do {
// CHECK:      ^bb0(%arg2: index):
// CHECK:        %8 = arith.subi %arg2, %c1 : index
// CHECK:        scf.yield %8 : index
// CHECK:      }
// CHECK:      %extracted = tensor.extract %arg1[%1] : tensor<2047xi32>
// CHECK:      %2 = arith.subi %1, %c1024 : index
// CHECK:      %splat = tensor.splat %extracted : tensor<2047xi32>
// CHECK:      %3 = arith.muli %cst, %splat : tensor<2047xi32>
// CHECK:      %4 = tensor.empty() : tensor<2047xi32>
// CHECK:      %c2047 = arith.constant 2047 : index
// CHECK:      %5 = arith.subi %c2047, %2 : index
// CHECK:      %extracted_slice_0 = tensor.extract_slice %3[0] [%5] [1] : tensor<2047xi32> to tensor<?xi32>
// CHECK:      %extracted_slice_1 = tensor.extract_slice %3[%5] [%2] [1] : tensor<2047xi32> to tensor<?xi32>
// CHECK:      %inserted_slice = tensor.insert_slice %extracted_slice_0 into %4[%2] [%5] [1] : tensor<?xi32> into tensor<2047xi32>
// CHECK:      %inserted_slice_2 = tensor.insert_slice %extracted_slice_1 into %inserted_slice[0] [%2] [1] : tensor<?xi32> into tensor<2047xi32>
// CHECK:      %splat_3 = tensor.splat %c-1_i32 : tensor<2047xi32>
// CHECK:      %6 = arith.muli %inserted_slice_2, %splat_3 : tensor<2047xi32>
// CHECK:      %7 = arith.addi %arg1, %6 : tensor<2047xi32>
// CHECK:      scf.yield %7 : tensor<2047xi32>
// CHECK:    }
// CHECK:    %extracted_slice = tensor.extract_slice %0[0] [1024] [1] : tensor<2047xi32> to tensor<1024xi32>
// CHECK:    return %extracted_slice : tensor<1024xi32>
// CHECK:  }
// CHECK: }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly2 = polynomial.mul(%poly0, %poly1) {ring = #ring} : !poly_ty
  return %poly2 : !poly_ty
}

// -----

// We then also want to test a non-machine-word sized modulus

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=18, ideal=#cycl_2048>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK: #[[LHS_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[RHS_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @lower_poly_mul(%[[poly0:.*]]: [[INPUT_TENSOR_TY:tensor<1024xi5>]], %[[poly1:.*]]: [[INPUT_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] {
// CHECK:      %[[POLYMUL_OUTPUT:.*]] = arith.constant dense<0> : [[POLYMUL_TENSOR_TY:tensor<2047xi5>]]
// CHECK:      %[[CMOD:.*]] = arith.constant 18 : i10
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[INPUT_TENSOR_TY]], [[INPUT_TENSOR_TY]])
// CHECK-SAME:     outs(%[[POLYMUL_OUTPUT]] : [[POLYMUL_TENSOR_TY]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: i5, %[[RHS_IN:.*]]: i5, %[[OUT:.*]]: i5):
// CHECK:       %[[LHS_EXT:.*]] = arith.extsi %[[LHS_IN]] : i5 to i10
// CHECK:       %[[RHS_EXT:.*]] = arith.extsi %[[RHS_IN]] : i5 to i10
// CHECK:       %[[OUT_EXT:.*]] = arith.extsi %[[OUT]] : i5 to i10
// CHECK:       %[[MULTED:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i10
// CHECK:       %[[SUMMED:.*]] = arith.addi %[[MULTED]], %[[OUT_EXT]] : i10
// CHECK:       %[[MODDED:.*]] = arith.remsi %[[SUMMED]], %[[CMOD]] : i10
// CHECK:       %[[ADDCMOD:.*]] = arith.addi %[[MODDED]], %[[CMOD]] : i10
// CHECK:       %[[CONGRUENT:.*]] = arith.remsi %[[ADDCMOD]], %[[CMOD]] : i10
// CHECK:       %[[RESULT:.*]] = arith.trunci %[[CONGRUENT]] : i10 to i5
// CHECK:       linalg.yield %[[RESULT]] : i5
// CHECK:     } -> [[POLYMUL_TENSOR_TY]]
// CHECK:     %[[MODDED_RESULT:.*]] = call @__heir_poly_mod_18_1_x1024(%[[GENERIC_RESULT]]) : ([[POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]]
// CHECK:     return
// CHECK: }

// CHECK: func.func private @__heir_poly_mod_18_1_x1024(%[[MOD_ARG0:.*]]: [[POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    // No need to re-test above
// CHECK: }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly = polynomial.mul(%poly0, %poly1) {ring = #ring} : !poly_ty
  return %poly : !poly_ty
}
