// RUN: heir-opt --layout-propagation=ciphertext-size=16 --split-input-file %s | FileCheck %s

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (2 - i0 + slot) mod 8 = 0 and 2 <= i0 <= 6 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @pad_1d
  // CHECK: tensor.pad
  // CHECK: tensor_ext.layout = #[[layout]]
  func.func @pad_1d(%arg0: !secret.secret<tensor<5xf32>>) -> !secret.secret<tensor<8xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<5xf32>>) {
    ^body(%input0: tensor<5xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %padded = tensor.pad %input0 low[2] high[1] {
      ^body(%arg1: index):
        tensor.yield %c0 : f32
      } : tensor<5xf32> to tensor<8xf32>
      secret.yield %padded : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %2 : !secret.secret<tensor<8xf32>>
  }
}

// -----

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-5i0 - i1 + slot + 16*floor((5i0 + i1)/16)) mod 32 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 4 and -15 + 5i0 + i1 <= 16ct <= 5i0 + i1 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @pad_2d_5x5_to_5x6
  // CHECK: tensor.pad
  // CHECK: tensor_ext.layout = #[[layout]]
  func.func @pad_2d_5x5_to_5x6(%arg0: !secret.secret<tensor<5x5xf32>>) -> !secret.secret<tensor<5x6xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<5x5xf32>>) {
    ^body(%input0: tensor<5x5xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %padded = tensor.pad %input0 low[0, 0] high[0, 1] {
      ^body(%arg1: index, %arg2: index):
        tensor.yield %c0 : f32
      } : tensor<5x5xf32> to tensor<5x6xf32>
      secret.yield %padded : tensor<5x6xf32>
    } -> !secret.secret<tensor<5x6xf32>>
    return %2 : !secret.secret<tensor<5x6xf32>>
  }
}

// -----

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (7 - 5i0 - i1 + slot + 16*floor((-7 + 5i0 + i1)/16)) mod 32 = 0 and 0 < i0 <= 5 and 2 <= i1 <= 6 and -22 + 5i0 + i1 <= 16ct <= -7 + 5i0 + i1 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @pad_2d_5x5_to_6x7
  // CHECK: tensor.pad
  // CHECK: tensor_ext.layout = #[[layout]]
  func.func @pad_2d_5x5_to_6x7(%arg0: !secret.secret<tensor<5x5xf32>>) -> !secret.secret<tensor<6x7xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<5x5xf32>>) {
    ^body(%input0: tensor<5x5xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %padded = tensor.pad %input0 low[1, 2] high[0, 0] {
      ^body(%arg1: index, %arg2: index):
        tensor.yield %c0 : f32
      } : tensor<5x5xf32> to tensor<6x7xf32>
      secret.yield %padded : tensor<6x7xf32>
    } -> !secret.secret<tensor<6x7xf32>>
    return %2 : !secret.secret<tensor<6x7xf32>>
  }
}
