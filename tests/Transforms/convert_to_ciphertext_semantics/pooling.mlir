// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=4096 | FileCheck %s
// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics="ciphertext-size=4096 unroll-kernels=false" | FileCheck %s --check-prefix=ROLL

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e1, e2, e3: i0 = 0 and ct = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 13 and 0 <= slot <= 4095 and 1024*floor((-1 - 196i1 - 14i2 - i3)/1024) <= -241 - 196i1 - 14i2 - i3 and 0 <= e1 <= 13 and 1024*floor((240 + slot)/1024) >= slot - 2e2 and slot - 4e1 - 2e2 + 2e3 <= 1024*floor((240 + slot)/1024) <= 1 + slot - 4e1 - 2e2 + 2e3 and 1024*floor((240 + slot)/1024) <= 3 + slot - 2e2 and 1024*floor((240 + slot)/1024) <= slot and 1011 + 196i1 + 14i2 + i3 + 196slot + 1024*floor((-1 - 196i1 - 14i2 - i3)/1024) - 14e1 - 392e2 <= 200704*floor((240 + slot)/1024) <= 1024 + 196i1 + 14i2 + i3 + 196slot + 1024*floor((-1 - 196i1 - 14i2 - i3)/1024) - 14e1 - 392e2 and 2048 + 392i1 + 28i2 + 2i3 + 391slot + 2048*floor((-1 - 196i1 - 14i2 - i3)/1024) - 28e1 - 784e2 + 28e3 <= 400384*floor((240 + slot)/1024) <= 2049 + 392i1 + 28i2 + 2i3 + 391slot + 2048*floor((-1 - 196i1 - 14i2 - i3)/1024) - 28e1 - 784e2 + 28e3) }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-784i1 - 28i2 - i3 + slot) mod 4096 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 27 and 0 <= i3 <= 4095 - 784i1 - 28i2 and i3 <= 27 and 0 <= slot <= 4095 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-196i1 - 14i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 4095 - 196i1 - 14i2 and i3 <= 13 and 0 <= slot <= 4095 and 4096*floor((-1024 + 196i1 + 14i2 + i3)/4096) <= -4096 + 196i1 + 14i2 + i3 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot, o2, o3, o4] : ct = i0 and slot = i1 and o4 = 28i2 + i3 + 56o2 + 2o3 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= o2 <= 13 and 0 <= o3 <= 13 }">
#layout4 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2, o3, o4] : ct = 0 and slot = i1 and o4 = i4 and (i0 + o3) mod 2 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 13 and 0 <= i4 <= 3135 and -1 + i0 + 4i2 <= 2o2 <= i0 + 4i2 and 2i3 <= o3 <= 1 + 2i3 }">
#layout5 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2] : i0 = 0 and ct = 28i2 + i3 and slot = i1 and o2 = i4 and 0 <= i1 <= 3 and 0 <= i2 <= 27 and 0 <= i3 <= 27 and 0 <= i4 <= 3135 }">
#layout6 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = i0 and slot = 784i1 + i2 and 0 <= i0 <= 783 and 0 <= i1 <= 3 and 0 <= i2 <= 783 }">
#layout7 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 1024 = 0 and (-i1 + ct + slot) mod 4096 = 0 and 0 <= i0 <= 783 and 0 <= i1 <= 3135 and 0 <= ct <= 1023 and 0 <= slot <= 4095 }">
#layout8 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e0, e1, e2, e3, e4, e5: 1024e4 = -i0 - 784i1 - 28i2 - i3 + ct + 2e0 - 56e1 - 2e2 + 28e3 and 4096e5 = -784i1 - 28i2 - i3 + ct + slot - 56e1 - 2e2 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and i0 <= 2e0 <= 27 + i0 and 0 <= e1 <= 13 and 0 <= e2 <= 13 and -1 - i0 + 2e0 <= 2e2 <= -i0 + 2e0 and 0 <= e3 <= 27 and -1 + i0 + 4e1 <= 2e3 <= i0 + 4e1) }">
module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @pooling
  // ROLL: func.func @pooling
  // ROLL: func.call @_assign_layout
  // ROLL-COUNT-2: scf.for
  // ROLL: scf.if
  // ROLL: scf.for

  // CHECK-COUNT-65: tensor_ext.rotate
  // CHECK: return

  func.func @pooling(%arg0: !secret.secret<tensor<1x4x28x28xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4x14x14xf32>
    %cst_0 = arith.constant dense<[[[[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]]]]> : tensor<4x4x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x4x28x28xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x4x28x28xf32>):
      %1 = tensor_ext.assign_layout %cst {layout = #layout2, tensor_ext.layout = #layout2} : tensor<1x4x14x14xf32>
      %2 = tensor_ext.assign_layout %cst_0 {layout = [#layout3, #layout4, #layout5, #layout6, #layout7], tensor_ext.layout = #layout8} : tensor<4x4x2x2xf32>
      %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, secret.kernel = #kernel, strides = dense<2> : vector<2xi64>, tensor_ext.layout = #layout} ins(%input0, %2 : tensor<1x4x28x28xf32>, tensor<4x4x2x2xf32>) outs(%1 : tensor<1x4x14x14xf32>) -> tensor<1x4x14x14xf32>
      secret.yield %3 : tensor<1x4x14x14xf32>
    } -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<1x4x14x14xf32>>
  }
}
