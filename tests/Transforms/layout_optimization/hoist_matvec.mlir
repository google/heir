// RUN: heir-opt --layout-optimization %s | FileCheck %s

#vec_layout = #tensor_ext.new_layout<domainSize=1, relation="(d, ct, slot) : ((d - slot) mod 1024 == 4, d >= 0, 0 >= d, slot >= 0, 1023 >= slot, ct == 0)">
#vec_layout_2 = #tensor_ext.new_layout<domainSize=1, relation="(d, ct, slot) : ((d - slot) mod 1024 == 7, d >= 0, 0 >= d, slot >= 0, 1023 >= slot, ct == 0)">
#mat_layout = #tensor_ext.new_layout<domainSize=2, relation="(row, col, ct, slot) : ((slot - row) mod 512 == 0, (ct + slot - col) mod 512 == 0, row >= 0, col >= 0, ct >= 0, slot >= 0, 1023 >= slot, 511 >= ct, 511 >= row, 511 >= col)">

// CHECK: #tensor_ext.new_layout<domainSize=2, localSize=7,

func.func @main(%arg0: tensor<512x512xf32>, %arg1: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout}) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout_2}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
  %0 = tensor.empty() : tensor<512xf32>
  // CHECK: tensor_ext.assign_layout
  // CHECK-NEXT: tensor_ext.assign_layout
  // CHECK-NEXT: secret.generic
  %1 = tensor_ext.assign_layout %0 {layout = #vec_layout, tensor_ext.layout = #vec_layout} : tensor<512xf32>
  %2 = tensor_ext.assign_layout %arg0 {layout = #mat_layout, tensor_ext.layout = #mat_layout} : tensor<512x512xf32>
  %3 = secret.generic(%arg1: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout}) {
  ^body(%input0: tensor<512xf32>):
    // two convert_layout ops here, one for hoisted vec conversion and one for change to matrix layout
    // CHECK: tensor_ext.convert_layout
    // CHECK-NEXT: tensor_ext.convert_layout
    // CHECK-NEXT: linalg.matvec
    // CHECK-NOT: tensor_ext.convert_layout
    %4 = linalg.matvec {tensor_ext.layout = #vec_layout, secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>}
        ins(%2, %input0 : tensor<512x512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>) -> tensor<512xf32>
    %5 = tensor_ext.convert_layout %4 {from_layout = #vec_layout, tensor_ext.layout = #vec_layout_2, to_layout = #vec_layout_2} : tensor<512xf32>
    secret.yield %5 : tensor<512xf32>
  } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout_2})
  return %3 : !secret.secret<tensor<512xf32>>
}
