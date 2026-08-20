// RUN: heir-opt %s --convert-to-ciphertext-semantics="min-slot-count=1024 unroll-kernels=false" | FileCheck %s
//
// A backend that evaluates a linear transform directly takes the compact
// rotate_and_reduce form for a matvec, rather than the expanded
// rotate/multiply/accumulate DAG. A squat matrix (rows < cols) still needs
// the rotate-and-add halving afterwards.
//
// CHECK: @matvec
// CHECK: tensor_ext.rotate_and_reduce
// CHECK-SAME: tensor_ext.lintrans
// CHECK-NOT: scf.for
// The squat post-reduction: one halving rotate-and-add for 512x1024, then the
// bias add.
// CHECK: tensor_ext.rotate
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: return

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
module attributes {
  backend.lattigo,
  backend.config_override = {has_kernel_linear_transform = true},
  scheme.ckks
} {
  func.func @matvec(%arg0: !secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512x784xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<784xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<512x784xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<512xf32>
      %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%1, %input0 : tensor<512x784xf32>, tensor<784xf32>) outs(%2 : tensor<512xf32>) -> tensor<512xf32>
      secret.yield %3 : tensor<512xf32>
    } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<512xf32>>
  }
}
