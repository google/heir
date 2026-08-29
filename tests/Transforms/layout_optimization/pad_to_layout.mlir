// RUN: heir-opt --layout-propagation --layout-optimization --convert-to-ciphertext-semantics="min-slot-count=4096" %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 12 and 0 <= slot <= 4095 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-48i1 - i2 + slot) mod 512 = 0 and 0 <= i1 <= 9 and 0 <= i2 <= 4095 - 48i1 and i2 <= 47 and 0 <= slot <= 4095 and 4096*floor((-512 + 48i1 + i2)/4096) <= -4096 + 48i1 + i2 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-48i0 - i1 + slot) mod 512 = 0 and 0 <= i0 <= 9 and 0 <= i1 <= 47 and 0 <= slot <= 4095 and 4096*floor((-512 + 48i0 + i1)/4096) <= -4096 + 48i0 + i1 }">

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (1 - 48i0 - i1 + slot) mod 512 = 0 and 0 <= i0 <= 9 and 0 < i1 <= 48 and 0 <= slot <= 4095 and 4096*floor((-513 + 48i0 + i1)/4096) <= -4097 + 48i0 + i1 }">
// CHECK: #[[layout1:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0, e1, e2, e3, e4: i0 = 0 and ct = 0 and 512e3 = -i1 + slot + 512e1 and 0 <= i1 <= 4095 and 0 <= slot <= 4095 and -4607 + i1 - 4096e0 <= 512e1 <= -4096 + i1 - 4096e0 and -4607 + i1 - 512e1 <= 4096e2 <= -4096 + i1 - 512e1 and 0 <= e4 <= 9 and -47 + i1 - 512e1 <= 48e4 <= i1 - 512e1) }">

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @tcresnet8small
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1x4096xf32>>
  func.func @tcresnet8small(%arg0: !secret.secret<tensor<1x10x48xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 10, 48>}, tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<10x50xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: %[[res:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<1x4096xf32>> {{.*}})
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x10x48xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 10, 48>}, tensor_ext.layout = #layout1}) {
    // CHECK: ^body(%[[input0:.*]]: tensor<1x4096xf32>):
    ^body(%input0: tensor<1x10x48xf32>):
      // CHECK: debug.validate
      debug.validate %input0 {metadata = "input", name = "input", tensor_ext.layout = []} : tensor<1x10x48xf32>
      // CHECK: %[[remap:.*]] = tensor_ext.remap %[[input0]] {permutation = #[[layout1]]} : tensor<1x4096xf32>
      %collapsed = tensor.collapse_shape %input0 [[0, 1], [2]] {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 10, 48>}, tensor_ext.layout = #layout2} : tensor<1x10x48xf32> into tensor<10x48xf32>
      %padded = tensor.pad %collapsed low[0, 1] high[0, 1] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %cst : f32
      } {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 10, 48>}, tensor_ext.layout = #layout2} : tensor<10x48xf32> to tensor<10x50xf32>
      // CHECK: secret.yield %[[remap]]
      secret.yield %padded : tensor<10x50xf32>
    } -> (!secret.secret<tensor<10x50xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<10x50xf32>>
  }
}
