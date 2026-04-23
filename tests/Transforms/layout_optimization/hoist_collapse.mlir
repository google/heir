// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

// Hoist a convert_layout operation above a collapse operation. The collapse is a trivial rank-reduction, so the layout should not change.

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and (-i1 + slot) mod 1024 = 0 and 0 <= i0 <= 127 and 0 <= i1 <= 767 and 0 <= slot <= 16383 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and (-768i1 - i2 + slot + 16384*floor((768i1 + i2)/16384)) mod 131072 = 0 and 0 <= i1 <= 127 and 0 <= i2 <= 767 and 0 <= ct <= 5 and -16383 + 768i1 + i2 <= 16384ct <= 768i1 + i2 and 0 <= slot <= 16383 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-768i0 - i1 + slot + 16384*floor((768i0 + i1)/16384)) mod 131072 = 0 and 0 <= i0 <= 127 and 0 <= i1 <= 767 and -16383 + 768i0 + i1 <= 16384ct <= 768i0 + i1 and 0 <= slot <= 16383 }">
module {
  // CHECK: func.func @collapse_hoist
  // CHECK: secret.generic
  // CHECK-NEXT: ^body([[input0:.*]]: tensor<1x128x768xf32>)
  // CHECK-NEXT: %[[collapsed:.*]] = tensor.collapse_shape [[input0]]
  // CHECK-NEXT: secret.yield %[[collapsed]]
  func.func @collapse_hoist(%arg0: !secret.secret<tensor<1x128x768xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<128x768xf32>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x128x768xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x128x768xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1], [2]] {tensor_ext.layout = #layout2} : tensor<1x128x768xf32> into tensor<128x768xf32>
      %1 = tensor_ext.convert_layout %collapsed {from_layout = #layout2, tensor_ext.layout = #layout, to_layout = #layout} : tensor<128x768xf32>
      secret.yield %1 : tensor<128x768xf32>
    } -> (!secret.secret<tensor<128x768xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<128x768xf32>>
  }
}
