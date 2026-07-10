// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=4096 | FileCheck %s

// Check the diagonalized filters shape and the row stride of the second
// convolution is 3 since the gap factor is 1.

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-3i2 - i3 + slot) mod 8 = 0 and 0 <= i2 <= 1 and 0 <= i3 <= 2 and 0 <= slot <= 31 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-3i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 2 and 0 <= i3 <= 2 and 0 <= slot <= 31 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot, o2, o3, o4] : i0 = 0 and i1 = 0 and ct = 0 and slot = 0 and o4 = 4i2 + i3 + 4o2 + o3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= o2 <= 2 and o3 >= 0 and -i3 <= o3 <= 3 - i3 and o3 <= 2 }">
#layout4 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2] : i0 = 0 and i1 = 0 and ct = 3i2 + i3 and slot = 0 and o2 = i4 and 0 <= i2 <= 2 and 0 <= i3 <= 2 and 0 <= i4 <= 15 }">
#layout5 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i1 = 0 and ct = i0 and slot = i2 and 0 <= i0 <= 8 and 0 <= i2 <= 15 }">
#layout6 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 16 = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 8 and 0 <= i1 <= 15 and 0 <= ct <= 15 and 0 <= slot <= 31 }">
#layout7 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 15 and 0 <= slot <= 31 and 16*floor((-3 - 4i2 - i3 + ct)/16) <= -16 - 4i2 - i3 + ct and 16*floor((7 + slot)/16) >= 45 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= 46 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= -31 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= slot and 16*floor((7 + slot)/16) <= -16 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) }">
#layout8 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot, o2, o3, o4] : i0 = 0 and i1 = 0 and i3 = 0 and ct = 0 and slot = 0 and o4 = 3i2 + 3o2 + o3 and 0 <= i2 <= 1 and 0 <= o2 <= 1 and 0 <= o3 <= 2 }">
#layout9 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2] : i0 = 0 and i1 = 0 and ct = 3i2 + i3 and slot = 0 and o2 = i4 and 0 <= i2 <= 1 and 0 <= i3 <= 2 and 0 <= i4 <= 8 }">
#layout10 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i1 = 0 and ct = i0 and slot = i2 and 0 <= i0 <= 5 and 0 <= i2 <= 8 }">
#layout11 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 8 = 0 and (-i1 + ct + slot) mod 16 = 0 and 0 <= i0 <= 5 and 0 <= i1 <= 8 and 0 <= ct <= 7 and 0 <= slot <= 31 }">
#layout12 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and i3 = 0 and (-3i2 + ct) mod 8 = 0 and 0 <= i2 <= 1 and 0 <= ct <= 7 and 0 <= slot <= 31 and -5 - 3i2 + ct + slot <= 16*floor((7 + ct + slot)/16) <= -3i2 + ct + slot and 16*floor((7 + ct + slot)/16) <= ct + slot }">
module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @conv2d_chain
  // CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[c_minus_3:.*]] = arith.constant -3 : index
  // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[c_minus_4:.*]] = arith.constant -4 : index
  // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: secret.generic
  // CHECK: ^body(%[[input:.*]]: tensor<1x4096xf32>):
  // CHECK: %[[filter1:.*]] = func.call @_assign_layout_{{.*}} : (tensor<1x1x2x2xf32>) -> tensor<16x4096xf32>
  // CHECK: %[[filter2:.*]] = func.call @_assign_layout_{{.*}} : (tensor<1x1x2x1xf32>) -> tensor<8x4096xf32>
  // Verify filter row stride 3 since row interchanging doesn't happen with gap_factor = 1
  // CHECK: %[[slice2:.*]] = tensor.extract_slice %[[filter2]][%[[c3]], 0] [1, 4096] [1, 1] : tensor<8x4096xf32> to tensor<1x4096xf32>
  // CHECK: %[[diag2:.*]] = tensor_ext.rotate %[[slice2]], %[[c_minus_3]] : tensor<1x4096xf32>, index
  // CHECK: arith.mulf %[[diag2]], {{.*}} : tensor<1x4096xf32>
  // CHECK: tensor_ext.rotate {{.*}}, %[[c3]] : tensor<1x4096xf32>, index
  // CHECK: secret.yield
  func.func @conv2d_chain(%arg0: !secret.secret<tensor<1x1x4x4xf32>> {heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64>, result_shape = array<i64: 1, 1, 4, 4>}, tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x1x2x3xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x1x2x1xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
    %0 = tensor.empty() : tensor<1x1x3x3xf32>
    %1 = tensor.empty() : tensor<1x1x2x3xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<1x1x4x4xf32>> {heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64>, result_shape = array<i64: 1, 1, 4, 4>}, tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x1x4x4xf32>):
      %3 = tensor_ext.assign_layout %0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<1x1x3x3xf32>
      %4 = tensor_ext.assign_layout %cst_0 {layout = [#layout3, #layout4, #layout5, #layout6], tensor_ext.layout = #layout7} : tensor<1x1x2x2xf32>
      %5 = linalg.conv_2d_nchw_fchw {heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 4, 4>, result_shape = array<i64: 1, 1, 3, 3>}, secret.kernel = #kernel, strides = dense<1> : vector<2xi64>, tensor_ext.layout = #layout2} ins(%input0, %4 : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%3 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
      %6 = tensor_ext.assign_layout %1 {layout = #layout, tensor_ext.layout = #layout} : tensor<1x1x2x3xf32>
      %7 = tensor_ext.assign_layout %cst {layout = [#layout8, #layout9, #layout10, #layout11], tensor_ext.layout = #layout12} : tensor<1x1x2x1xf32>
      %8 = linalg.conv_2d_nchw_fchw {heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 3, 3>, result_shape = array<i64: 1, 1, 2, 3>}, secret.kernel = #kernel, strides = dense<1> : vector<2xi64>, tensor_ext.layout = #layout} ins(%5, %7 : tensor<1x1x3x3xf32>, tensor<1x1x2x1xf32>) outs(%6 : tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
      secret.yield %8 : tensor<1x1x2x3xf32>
    } -> (!secret.secret<tensor<1x1x2x3xf32>> {tensor_ext.layout = #layout})
    return %2 : !secret.secret<tensor<1x1x2x3xf32>>
  }
}
