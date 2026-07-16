// RUN: heir-opt --convert-to-ciphertext-semantics=ciphertext-size=16 --split-input-file %s | FileCheck %s

#layout_src = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i and 0 <= i <= 4 and 0 <= slot <= 15 }">
#layout_dst = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i - 2 and 2 <= i <= 6 and 0 <= slot <= 15 }">

// CHECK: #[[layout_dst:.*]] = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i - 2 and 2 <= i <= 6 and 0 <= slot <= 15 }">
// CHECK: #[[layout_remap:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and slot = i1 and 0 <= i1 <= 4 }">
// CHECK: #[[original_type_dst:.*]] = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #[[layout_dst]]>

module {
  // CHECK: func.func @pad_1d
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1x16xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<5xf32>, layout = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i and 0 <= i <= 4 and 0 <= slot <= 15 }">>})
  // CHECK-SAME: -> (!secret.secret<tensor<1x16xf32>> {tensor_ext.original_type = #[[original_type_dst]]})
  func.func @pad_1d(%arg0: !secret.secret<tensor<5xf32>> {tensor_ext.layout = #layout_src}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout_dst}) {
    // CHECK: %[[res:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<1x16xf32>>)
    // CHECK: ^body(%[[input0:.*]]: tensor<1x16xf32>):
    // CHECK: %[[remap:.*]] = tensor_ext.remap %[[input0]] {permutation = #[[layout_remap]]} : tensor<1x16xf32>
    // CHECK: secret.yield %[[remap]] : tensor<1x16xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<5xf32>> {tensor_ext.layout = #layout_src}) {
    ^body(%input0: tensor<5xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %padded = tensor.pad %input0 low[2] high[1] {
      ^body(%arg1: index):
        tensor.yield %c0 : f32
      } {tensor_ext.layout = #layout_dst} : tensor<5xf32> to tensor<8xf32>
      secret.yield %padded : tensor<8xf32>
    } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout_dst})
    return %2 : !secret.secret<tensor<8xf32>>
  }
}

// -----

#layout_src_2d = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-5i0 - i1 + slot + 16*floor((5i0 + i1)/16)) mod 32 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 4 and -15 + 5i0 + i1 <= 16ct <= 5i0 + i1 and 0 <= slot <= 15 }">
#layout_dst_2d = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (7 - 5i0 - i1 + slot + 16*floor((-7 + 5i0 + i1)/16)) mod 32 = 0 and 0 < i0 <= 5 and 2 <= i1 <= 6 and -22 + 5i0 + i1 <= 16ct <= -7 + 5i0 + i1 and 0 <= slot <= 15 }">
#original_type_dst_2d = #tensor_ext.original_type<originalType = tensor<6x7xf32>, layout = #layout_dst_2d>

// CHECK: #[[layout_remap_2d:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and (-i1 + slot) mod 32 = 0 and i0 >= 0 and 0 <= i1 <= 24 - 16i0 and i1 <= 15 and 0 <= slot <= 15 }">
// CHECK: #[[original_type_dst_2d:.*]] = #tensor_ext.original_type<originalType = tensor<6x7xf32>, layout = #[[layout_dst_2d:.*]]>

module {
  // CHECK: func.func @pad_2d
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<2x16xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<5x5xf32>, layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-5i0 - i1 + slot + 16*floor((5i0 + i1)/16)) mod 32 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 4 and -15 + 5i0 + i1 <= 16ct <= 5i0 + i1 and 0 <= slot <= 15 }">>})
  // CHECK-SAME: -> (!secret.secret<tensor<2x16xf32>> {tensor_ext.original_type = #[[original_type_dst_2d]]})
  func.func @pad_2d(%arg0: !secret.secret<tensor<5x5xf32>> {tensor_ext.layout = #layout_src_2d}) -> (!secret.secret<tensor<6x7xf32>> {tensor_ext.layout = #layout_dst_2d}) {
    // CHECK: %[[res:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<2x16xf32>>)
    // CHECK: ^body(%[[input0:.*]]: tensor<2x16xf32>):
    // CHECK: %[[remap:.*]] = tensor_ext.remap %[[input0]] {permutation = #[[layout_remap_2d]]} : tensor<2x16xf32>
    // CHECK: secret.yield %[[remap]] : tensor<2x16xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<5x5xf32>> {tensor_ext.layout = #layout_src_2d}) {
    ^body(%input0: tensor<5x5xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %padded = tensor.pad %input0 low[1, 2] high[0, 0] {
      ^body(%arg1: index, %arg2: index):
        tensor.yield %c0 : f32
      } {tensor_ext.layout = #layout_dst_2d} : tensor<5x5xf32> to tensor<6x7xf32>
      secret.yield %padded : tensor<6x7xf32>
    } -> (!secret.secret<tensor<6x7xf32>> {tensor_ext.layout = #layout_dst_2d})
    return %2 : !secret.secret<tensor<6x7xf32>>
  }
}
