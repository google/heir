// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=4096 | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-196i1 - 14i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 13 and 0 <= slot <= 4095 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-784i1 - 28i2 - i3 + slot) mod 4096 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 27 and 0 <= i3 <= 4095 - 784i1 - 28i2 and i3 <= 27 and 0 <= slot <= 4095 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : (784i0 - 784i1 - 28i2 - i3 + ct - slot - 4*floor((240 + slot)/1024) - 28*floor((slot - 2*floor((240 + slot)/1024))/14)) mod 4096 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and -195 - 196i0 + slot <= 1024*floor((240 + slot)/1024) <= -196i0 + slot and 1024*floor((240 + slot)/1024) <= slot and 14*floor((slot - 2*floor((240 + slot)/1024))/14) >= 392i0 - 392i1 - 14i2 - slot + 2046*floor((240 + slot)/1024) and 14*floor((slot - 2*floor((240 + slot)/1024))/14) <= 1567 + 392i0 - 392i1 - 14i2 - slot + 2046*floor((240 + slot)/1024) }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-196i1 - 14i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 4095 - 196i1 - 14i2 and i3 <= 13 and 0 <= slot <= 4095 and 4096*floor((-1024 + 196i1 + 14i2 + i3)/4096) <= -4096 + 196i1 + 14i2 + i3 }">
module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @pooling
  func.func @pooling(%arg0: !secret.secret<tensor<1x4x28x28xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout}) {
    // CHECK: arith.constant dense<0{{.*}}> : tensor<1x4096xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4x14x14xf32>
    %cst_0 = arith.constant dense<[[[[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]]]]> : tensor<4x4x2x2xf32>
    // CHECK: secret.generic
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x4x28x28xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x4x28x28xf32>):
      // CHECK: func.call @_assign_layout_
      %1 = tensor_ext.assign_layout %cst_0 {domainSchedule = array<i64: 0, 1>, layout = #layout2, tensor_ext.layout = #layout2} : tensor<4x4x2x2xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout3, tensor_ext.layout = #layout3} : tensor<1x4x14x14xf32>
      // CHECK-COUNT-1024: tensor_ext.rotate
      %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, secret.kernel = #kernel, strides = dense<2> : vector<2xi64>, tensor_ext.layout = #layout} ins(%input0, %1 : tensor<1x4x28x28xf32>, tensor<4x4x2x2xf32>) outs(%2 : tensor<1x4x14x14xf32>) -> tensor<1x4x14x14xf32>
      secret.yield %3 : tensor<1x4x14x14xf32>
    } -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout})
    // CHECK: return
    return %0 : !secret.secret<tensor<1x4x14x14xf32>>
  }
}
