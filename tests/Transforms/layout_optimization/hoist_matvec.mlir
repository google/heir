// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

#vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 4 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">
#vec_layout_2 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 7 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">
#mat_layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 512 = 0 and (ct + slot - col) mod 512 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 1023 >= slot and 511 >= ct and 511 >= row and 511 >= col }">

// CHECK: #tensor_ext.layout<"{ [i0, i1] -> [ct, slot]

func.func @main(%arg0: tensor<512x512xf32>, %arg1: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout}) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout_2}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
  %0 = tensor.empty() : tensor<512xf32>
  // CHECK: tensor_ext.assign_layout
  // CHECK-NEXT: tensor_ext.assign_layout
  // CHECK-NEXT: secret.generic
  // CHECK-NOT: tensor_ext.assign_layout
  // CHECK-NOT: tensor_ext.convert_layout
  %1 = tensor_ext.assign_layout %0 {layout = #vec_layout, tensor_ext.layout = #vec_layout} : tensor<512xf32>
  %2 = tensor_ext.assign_layout %arg0 {layout = #mat_layout, tensor_ext.layout = #mat_layout} : tensor<512x512xf32>
  %3 = secret.generic(%arg1: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout}) {
  ^body(%input0: tensor<512xf32>):
    %4 = linalg.matvec {tensor_ext.layout = #vec_layout, secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>}
        ins(%2, %input0 : tensor<512x512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>) -> tensor<512xf32>
    %5 = tensor_ext.convert_layout %4 {from_layout = #vec_layout, tensor_ext.layout = #vec_layout_2, to_layout = #vec_layout_2} : tensor<512xf32>
    secret.yield %5 : tensor<512xf32>
  } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout_2})
  return %3 : !secret.secret<tensor<512xf32>>
}
