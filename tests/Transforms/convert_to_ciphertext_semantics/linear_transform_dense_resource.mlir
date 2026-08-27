// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=8 | FileCheck %s

// The same matvec as the second case of linear_transform.mlir, but with the
// weights in a dense_resource blob rather than inline.
//
// The diagonals must come out with the same values as the inline case:
// diagonal ct holds M[i][(i + ct) mod 4] at every slot congruent to i.

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 7 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 4 = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 3 and 0 <= slot <= 7 }">

module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  // CHECK: func.func @matvec_resource_weights
  // Diagonal ct = 0 is the matrix diagonal, all ones. Diagonal 1 wraps round to
  // M[3][0] = 7, so a misread blob shows up as a changed value, not just as a
  // missing op.
  // CHECK: %[[DIAGS:.*]] = arith.constant dense<
  // CHECK-SAME: {{\[}}1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  // CHECK-SAME: {{\[}}2.000000e+00, 2.000000e+00, 2.000000e+00, 7.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 7.000000e+00]
  // CHECK-SAME: {{\[}}3.000000e+00, 3.000000e+00, 6.000000e+00, 6.000000e+00, 3.000000e+00, 3.000000e+00, 6.000000e+00, 6.000000e+00]
  // CHECK-SAME: {{\[}}4.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 4.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00]
  // CHECK: secret.generic
  // CHECK: kernel.linear_transform %{{.*}}, %[[DIAGS]]
  // CHECK-SAME: diagonal_indices = array<i64: 0, 1, 2, 3>
  // CHECK: secret.yield
  func.func @matvec_resource_weights(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %cst_0 = arith.constant dense_resource<matvec_weights> : tensor<4x4xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<4xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<4x4xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<4xf32>
      %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%1, %input0 : tensor<4x4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
      secret.yield %3 : tensor<4xf32>
    } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<4xf32>>
  }
}

{-#
  dialect_resources: {
    builtin: {
      matvec_weights: "0x040000000000803f0000004000004040000080400000a0400000803f00000040000040400000c0400000a0400000803f000000400000e0400000c0400000a0400000803f"
    }
  }
#-}
