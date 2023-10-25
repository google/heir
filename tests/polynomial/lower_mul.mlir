// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=4294967296, ideal=#cycl_2048>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK: #[[LHS_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[RHS_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @lower_poly_mul(%[[poly0:.*]]: [[INPUT_TENSOR_TY:tensor<1024xi32>]], %[[poly1:.*]]: [[INPUT_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] {
// CHECK:      %[[NAIVE_POLYMUL_OUTPUT:.*]] = arith.constant dense<0> : [[NAIVE_POLYMUL_TENSOR_TY:tensor<2047xi64>]]
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[INPUT_TENSOR_TY]], [[INPUT_TENSOR_TY]])
// CHECK-SAME:     outs(%[[NAIVE_POLYMUL_OUTPUT]] : [[NAIVE_POLYMUL_TENSOR_TY]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[OUT:.*]]: i64):
// CHECK:       %[[LHS_EXT:.*]] = arith.extui %[[LHS_IN]] : i32 to i64
// CHECK:       %[[RHS_EXT:.*]] = arith.extui %[[RHS_IN]] : i32 to i64
// CHECK:       %[[MULTED:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK:       %[[SUMMED:.*]] = arith.addi %[[MULTED]], %[[OUT]] : i64
// CHECK:       linalg.yield %[[SUMMED]] : i64
// CHECK:     } -> [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:     %[[MODDED_RESULT:.*]] = call @__heir_poly_mod_4294967296_1_x1024(%[[GENERIC_RESULT]]) : ([[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]]
// CHECK:     return
// CHECK: }

// CHECK: func.func private @__heir_poly_mod_4294967296_1_x1024(%[[MOD_ARG0:.*]]: [[NAIVE_POLYMUL_TENSOR_TY]]) -> [[INPUT_TENSOR_TY]] attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:    %[[c_minus1:.*]] = arith.constant -1 : i32
// CHECK:    %[[modulus_tensor:.*]] = arith.constant dense<4294967296> : [[NAIVE_POLYMUL_TENSOR_TY]]
// CHECK:    %[[c1024:.*]] = arith.constant 1024 : index
// CHECK:    %[[rem_result:.*]] = scf.while (%[[WHILE_ARG1:.*]] = %[[MOD_ARG0]]) : ([[NAIVE_POLYMUL_TENSOR_TY]]) -> [[NAIVE_POLYMUL_TENSOR_TY]] {
// CHECK:      %[[c0_i64:.*]] = arith.constant 0 : i64
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

// CHECK:      %c0_i64 = arith.constant 0 : i64
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c2046 = arith.constant 2046 : index
// CHECK:      %3 = scf.while (%arg2 = %c2046) : (index) -> index {
// CHECK:        %extracted_5 = tensor.extract %arg1[%arg2] : tensor<2047xi64>
// CHECK:        %11 = arith.cmpi eq, %extracted_5, %c0_i64 : i64
// CHECK:        scf.condition(%11) %arg2 : index
// CHECK:      } do {
// CHECK:      ^bb0(%arg2: index):
// CHECK:        %11 = arith.subi %arg2, %c1 : index
// CHECK:        scf.yield %11 : index
// CHECK:      }
// CHECK:      %extracted = tensor.extract %arg1[%3] : tensor<2047xi64>
// CHECK:      %4 = arith.subi %3, %c1024 : index
// CHECK:      %splat = tensor.splat %extracted : tensor<2047xi64>
// CHECK:      %5 = arith.muli %cst_0, %splat : tensor<2047xi64>
// CHECK:      %6 = tensor.empty() : tensor<2047xi64>
// CHECK:      %c2047 = arith.constant 2047 : index
// CHECK:      %7 = arith.subi %c2047, %4 : index
// CHECK:      %extracted_slice_1 = tensor.extract_slice %5[0] [%7] [1] : tensor<2047xi64> to tensor<?xi64>
// CHECK:      %extracted_slice_2 = tensor.extract_slice %5[%7] [%4] [1] : tensor<2047xi64> to tensor<?xi64>
// CHECK:      %inserted_slice = tensor.insert_slice %extracted_slice_1 into %6[%4] [%7] [1] : tensor<?xi64> into tensor<2047xi64>
// CHECK:      %inserted_slice_3 = tensor.insert_slice %extracted_slice_2 into %inserted_slice[0] [%4] [1] : tensor<?xi64> into tensor<2047xi64>
// CHECK:      %8 = arith.extui %c-1_i32 : i32 to i64
// CHECK:      %splat_4 = tensor.splat %8 : tensor<2047xi64>
// CHECK:      %9 = arith.muli %inserted_slice_3, %splat_4 : tensor<2047xi64>
// CHECK:      %10 = arith.addi %arg1, %9 : tensor<2047xi64>
// CHECK:      scf.yield %10 : tensor<2047xi64>
// CHECK:    }
// CHECK:    %1 = arith.remui %0, %cst : tensor<2047xi64>
// CHECK:    %2 = arith.trunci %1 : tensor<2047xi64> to tensor<2047xi32>
// CHECK:    %extracted_slice = tensor.extract_slice %2[0] [1024] [1] : tensor<2047xi32> to tensor<1024xi32>
// CHECK:    return %extracted_slice : tensor<1024xi32>
// CHECK:  }
// CHECK:  }

func.func @lower_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly2 = polynomial.mul(%poly0, %poly1) {ring = #ring} : !poly_ty
  return %poly2 : !poly_ty
}

// TODO(https://github.com/google/heir/issues/199): add proper tests for a
// non-1 divisor leading coefficient
