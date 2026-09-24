// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=min-slot-count=8 | FileCheck %s

// CHECK: module
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  // CHECK: @main
  func.func @main(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 3 }">}) -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">}) {
    %cst = arith.constant dense<0.0> : tensor<2xf32>
    %cst_mat = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %0 = tensor_ext.assign_layout %cst_mat {
      layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0: ct + slot - i1 - 4e0 = 0 and 0 <= ct <= 3) and slot - i0 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 3 }">,
      tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0: ct + slot - i1 - 4e0 = 0 and 0 <= ct <= 3) and slot - i0 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 3 }">
    } : tensor<2x4xf32>
    %1 = tensor_ext.assign_layout %cst {
      layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">,
      tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">
    } : tensor<2xf32>
    %2 = secret.generic(%arg0 : !secret.secret<tensor<4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 3 }">}) {
    ^body(%input: tensor<4xf32>):
      // The diagonals are an operand rather than an attribute, so the packed
      // matrix can be a resource or a preprocessed value instead of inline IR.
      // CHECK: %[[diags:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 6.000000e+00
      // CHECK: kernel.linear_transform %{{.*}}, %[[diags]]
      // CHECK-SAME: diagonal_indices = array<i64: 0, 1, 2, 3>
      %3 = linalg.matvec {
        secret.kernel = #secret.kernel<name = "MatvecDiagonal", force = false>,
        tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">
      } ins(%0, %input : tensor<2x4xf32>, tensor<4xf32>) outs(%1 : tensor<2xf32>) -> tensor<2xf32>
      secret.yield %3 : tensor<2xf32>
    } -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">})
    return %2 : !secret.secret<tensor<2xf32>>
  }
}

// -----

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 7 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 4 = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 3 and 0 <= slot <= 7 }">

module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  // CHECK: func.func @matvec_to_linear_transform
  // CHECK-SAME: (%[[ARG0:.*]]: !secret.secret<tensor<1x8xf32>> {{.*}}) -> (!secret.secret<tensor<1x8xf32>> {{.*}})
  // CHECK: secret.generic
  // CHECK: kernel.linear_transform {{%[a-zA-Z0-9_]+}}
  // CHECK-SAME: diagonal_indices = array<i64: 0, 1, 2, 3>
  // CHECK: secret.yield
  func.func @matvec_to_linear_transform(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %cst_0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.0, 3.0], [6.0, 5.0, 1.0, 2.0], [7.0, 6.0, 5.0, 1.0]]> : tensor<4x4xf32>
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
