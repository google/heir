// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=8192

// A smoke test for matvec with nontrivial alignments

#alignment = #tensor_ext.alignment<in = [16], out = [8192]>
#alignment1 = #tensor_ext.alignment<in = [16, 16], out = [16, 16]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 8192), alignment = #alignment>
#layout1 = #tensor_ext.layout<map = (d0, d1) -> ((-d0 + d1) mod 16, (d0 + ((-d0 + d1) floordiv 16) * 16) mod 8192), alignment = #alignment1>
func.func @matvec(%arg0: !secret.secret<tensor<16xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<16xf32>> {tensor_ext.layout = #layout}) {
  %cst = arith.constant dense<0.0> : tensor<16xf32>
  %cst_0 = arith.constant dense<0.0> : tensor<16x16xf32>
  %1 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<16x16xf32>
  %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<16xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) attrs = {__argattrs = [{tensor_ext.layout = #layout}], __resattrs = [{tensor_ext.layout = #layout}]} {
  ^body(%input0: tensor<16xf32>):
    %3 = linalg.matvec {tensor_ext.layout = #layout} ins(%1, %input0 : tensor<16x16xf32>, tensor<16xf32>) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    secret.yield %3 : tensor<16xf32>
  } -> !secret.secret<tensor<16xf32>>
  return %0 : !secret.secret<tensor<16xf32>>
}
