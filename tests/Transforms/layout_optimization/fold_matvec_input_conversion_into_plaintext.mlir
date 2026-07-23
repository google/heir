// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

#permuted = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - 3i) mod 8 = 0 and 0 <= i <= 7 and 0 <= slot <= 7 }">
#row_major = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 8 = 0 and 0 <= i <= 7 and 0 <= slot <= 7 }">
#result = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 4 = 0 and 0 <= i <= 3 and 0 <= slot <= 7 }">
#matrix = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (row - col + ct) mod 4 = 0 and (-col + ct + slot) mod 8 = 0 and 0 <= row <= 3 and 0 <= col <= 7 and 0 <= ct <= 3 and 0 <= slot <= 7 }">

// CHECK: #[[MATRIX:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 + i1 + ct) mod 4 = 0 and (-3i1 + ct + slot) mod 8 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 7 and 0 <= ct <= 3 and 0 <= slot <= 7 }">
// CHECK: func.func @fold_matvec_input_conversion_into_plaintext(%[[WEIGHTS:arg[0-9]+]]:
func.func @fold_matvec_input_conversion_into_plaintext(
    %weights: tensor<4x8xf32>,
    %input: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #permuted})
    -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #result}) {
  %empty = tensor.empty() : tensor<4xf32>
  %init = tensor_ext.assign_layout %empty {layout = #result, tensor_ext.layout = #result} : tensor<4xf32>
  %matrix = tensor_ext.assign_layout %weights {layout = #matrix, tensor_ext.layout = #matrix} : tensor<4x8xf32>
  %0 = secret.generic(
      %input: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #permuted}) {
  ^body(%arg0: tensor<8xf32>):
    // CHECK: %[[PACKED_WEIGHTS:.*]] = tensor_ext.assign_layout %[[WEIGHTS]]
    // CHECK-SAME: layout = #[[MATRIX]]
    // CHECK: ^body(%[[INPUT:.*]]: tensor<8xf32>)
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: linalg.matvec
    // CHECK-SAME: ins(%[[PACKED_WEIGHTS]], %[[INPUT]]
    %converted = tensor_ext.convert_layout %arg0 {from_layout = #permuted, tensor_ext.layout = #row_major, to_layout = #row_major} : tensor<8xf32>
    %result = linalg.matvec {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>, tensor_ext.layout = #result}
        ins(%matrix, %converted : tensor<4x8xf32>, tensor<8xf32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %result : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>> {tensor_ext.layout = #result}
  return %0 : !secret.secret<tensor<4xf32>>
}
