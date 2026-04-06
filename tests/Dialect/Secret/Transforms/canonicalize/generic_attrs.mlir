// RUN: heir-opt --canonicalize %s | FileCheck %s

// This is a regression test specifically testing that the result attrs of the assign_layout that is hoisted out is preserved.

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-25i1 - 5i2 - i3 + slot) mod 128 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 4 and 0 <= i3 <= 1023 - 25i1 - 5i2 and i3 <= 4 and 0 <= slot <= 1023 and 1024*floor((-128 + 25i1 + 5i2 + i3)/1024) <= -1024 + 25i1 + 5i2 + i3 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-10i2 - i3 + slot) mod 128 = 0 and 0 <= i2 <= 9 and 0 <= i3 <= 9 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e0, e1, e2, e3, e4, e6: i1 = 0 and 128e6 = -10i2 - i3 + ct + slot - 20e0 - 2e1 and 0 <= i0 <= 3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 15 and 0 <= slot <= 1023 and 0 <= e0 <= 4 and 0 <= e1 <= 35 - 5i2 - 10e0 and e1 <= 4 and 0 <= e2 <= 1 and 16*floor((slot)/16) >= slot - 2e3 and slot - 4e2 - 2e3 + 2e4 <= 16*floor((slot)/16) <= 1 + slot - 4e2 - 2e3 + 2e4 and 16*floor((slot)/16) <= 3 + slot - 2e3 and -1 + 25i0 + 4slot + 5e0 + e1 - 2e2 - 8e3 <= 64*floor((slot)/16) <= 25i0 + 4slot + 5e0 + e1 - 2e2 - 8e3 and 50i0 + 7slot + 10e0 + 2e1 - 4e2 - 16e3 + 4e4 <= 112*floor((slot)/16) <= 1 + 50i0 + 7slot + 10e0 + 2e1 - 4e2 - 16e3 + 4e4) }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e1, e2, e3: i0 = 0 and ct = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 4 and 0 <= i3 <= 4 and 0 <= slot <= 99 and 128*floor((-1 - 25i1 - 5i2 - i3)/128) <= -29 - 25i1 - 5i2 - i3 and 0 <= e1 <= 4 and slot <= 2e2 <= 3 + slot and 124 + 25i1 + 5i2 + i3 + 25slot + 128*floor((-1 - 25i1 - 5i2 - i3)/128) - 5e1 <= 50e2 <= 128 + 25i1 + 5i2 + i3 + 25slot + 128*floor((-1 - 25i1 - 5i2 - i3)/128) - 5e1 and -1 - slot + 4e1 + 2e2 <= 2e3 <= -slot + 4e1 + 2e2 and -257 - 50i1 - 10i2 - 2i3 - 49slot - 256*floor((-1 - 25i1 - 5i2 - i3)/128) + 10e1 + 100e2 <= 10e3 <= -256 - 50i1 - 10i2 - 2i3 - 49slot - 256*floor((-1 - 25i1 - 5i2 - i3)/128) + 10e1 + 100e2) }">
module {
  // CHECK: @conv2d_nchw
  // CHECK: tensor_ext.assign_layout
  // CHECK-SAME: tensor_ext.layout
  // CHECK: secret.generic
  // CHECK: return
  func.func @conv2d_nchw(%arg0: !secret.secret<tensor<4x1x2x2xf32>> {tensor_ext.layout = #layout2}) -> (!secret.secret<tensor<4x1x2x2xf32>> {tensor_ext.layout = #layout2}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4x5x5xf32>
    %cst_0 = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4x1x2x2xf32>> {tensor_ext.layout = #layout2}) {
    ^body(%input0: tensor<4x1x2x2xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<4x1x2x2xf32>
      %2 = arith.addf %input0, %1 {tensor_ext.layout = #layout2} : tensor<4x1x2x2xf32>
      secret.yield %2 : tensor<4x1x2x2xf32>
    } -> (!secret.secret<tensor<4x1x2x2xf32>> {tensor_ext.layout = #layout2})
    return %0 : !secret.secret<tensor<4x1x2x2xf32>>
  }
}
