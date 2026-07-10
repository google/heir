// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics="ciphertext-size=1024 unroll-kernels=false" | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 1023 }">
// Layout for matrix: only maps diagonals 0 and 2 (ct = 0, 2)
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 4 = 0 and (-i0 + slot) mod 4 = 0 and (i0 - i1) mod 2 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 3 and 0 <= slot <= 1023 }">

// CHECK: func.func @square_zero_diag
// CHECK-DAG: %[[CST_ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[IS_NOT_ZERO_DIAG_TENSOR:.*]] = arith.constant dense<[1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<4xf64>
// CHECK: secret.generic
// CHECK:   %[[LOOP:.*]] = scf.for %[[I:.*]] =
// CHECK:     %[[IDX:.*]] = arith.addi
// CHECK:     %[[BOUND_COND:.*]] = arith.cmpi slt, %[[IDX]], %c4
// CHECK:     %[[RES_IF:.*]] = scf.if %[[BOUND_COND]]
// CHECK:       %[[VAL:.*]] = tensor.extract %[[IS_NOT_ZERO_DIAG_TENSOR]][%[[IDX]]]
// CHECK:       %[[COND:.*]] = arith.cmpf one, %[[VAL]], %[[CST_ZERO]]
// CHECK:       %[[RES_IF2:.*]] = scf.if %[[COND]]
// CHECK:         %{{.*}} = tensor.extract_slice
// CHECK:         %{{.*}} = tensor_ext.rotate
// CHECK:         %{{.*}} = arith.mulf
// CHECK:         %[[ADD:.*]] = arith.addf
// CHECK:         scf.yield %[[ADD]]
// CHECK:       else
// CHECK:         scf.yield
// CHECK:       scf.yield %[[RES_IF2]]
// CHECK:     else
// CHECK:       scf.yield
// CHECK:     scf.yield %[[RES_IF]]
// CHECK: return
func.func @square_zero_diag(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
  %cst = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // Diagonals 1 and 3 are zero.
  %cst_0 = arith.constant dense<[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<4xf32>):
    %1 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<4x4xf32>
    %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<4xf32>
    %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%1, %input0 : tensor<4x4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %3 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<tensor<4xf32>>
}
