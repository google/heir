// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl = #polynomial.polynomial<1 + x**4>
#ring = #polynomial.ring<cmod=7681, ideal=#cycl, root=1925>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK: #[[MATRIX_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[VEC_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK:     func.func @lower_ntt() -> [[OUTPUT_TYPE:.*]] {
// CHECK-DAG:   %[[COEFFS:.*]] = arith.constant dense<[1, 2, 3, 4]> : [[INPUT_TYPE:.*]]
// CHECK-DAG:   %[[PRIM_MAT:.*]] = arith.constant dense<{{.}}[1, 1925, 3383, -1724], [1, -1724, -3894, 1925], [1, -2436, 3383, 1213], [1, 1213, -3894, -2436]{{.}}> : [[MAT_TYPE:.*]]
// CHECK-DAG:   %[[NTT_OUTPUT:.*]] = arith.constant dense<0> : [[INPUT_TYPE]]
// CHECK-DAG:   %[[CMOD:.*]] = arith.constant 7681 : i26
// CHECK:      %[[GENERIC_RESULT:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MATRIX_MAP]], #[[VEC_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[generic_arg0:.*]], %[[generic_arg1:.*]] : [[MAT_TYPE]], [[INPUT_TYPE]])
// CHECK-SAME:     outs(%[[NTT_OUTPUT]] : [[INPUT_TYPE]])
// CHECK:     ^[[BB0:.*]](%[[LHS_IN:.*]]: i13, %[[RHS_IN:.*]]: i13, %[[OUT:.*]]: i13):
// CHECK:       %[[LHS_EXT:.*]] = arith.extui %[[LHS_IN]] : i13 to i26
// CHECK:       %[[RHS_EXT:.*]] = arith.extui %[[RHS_IN]] : i13 to i26
// CHECK:       %[[OUT_EXT:.*]] = arith.extui %[[OUT]] : i13 to i26
// CHECK:       %[[MULTED:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i26
// CHECK:       %[[SUMMED:.*]] = arith.addi %[[MULTED]], %[[OUT_EXT]] : i26
// CHECK:       %[[MODDED:.*]] = arith.remui %[[SUMMED]], %[[CMOD]] : i26
// CHECK:       %[[RESULT:.*]] = arith.trunci %[[MODDED]] : i26 to i13
// CHECK:       linalg.yield %[[RESULT]] : i13
// CHECK:     } -> [[INPUT_TYPE]]
// CHECK:     %[[RES:.*]] = tensor.cast %[[GENERIC_RESULT]] : [[INPUT_TYPE]] to [[OUTPUT_TYPE:.*]]
// CHECK:     return %[[RES]] : [[OUTPUT_TYPE]]

func.func @lower_ntt() -> tensor<4xi13, #ring> {
  %coeffs = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi13>
  %poly = polynomial.from_tensor %coeffs : tensor<4xi13> -> !poly_ty
  %ret = polynomial.ntt %poly : !poly_ty -> tensor<4xi13, #ring>
  return %ret : tensor<4xi13, #ring>
}
