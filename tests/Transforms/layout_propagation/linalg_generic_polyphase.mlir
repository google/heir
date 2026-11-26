// RUN: heir-opt --layout-propagation=min-slot-count=16 --split-input-file %s | FileCheck %s
// RUN: heir-opt --layout-propagation=min-slot-count=16 --layout-optimization --split-input-file %s | FileCheck %s --check-prefix=OPT

// A stride-2 window over a row-major input needs a rotation per output
// position. Placing the strided result at the input's strided positions would
// spread it over twice the ciphertexts, so the result keeps the dense default
// layout and the input is converted to a polyphase layout: even elements
// first, then odd ones, each with the result's pitch. The input is a function
// argument, so layout optimization folds the conversion into it.

#window = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#tap = affine_map<(d0, d1) -> (d1)>
#out = affine_map<(d0, d1) -> (d0)>

// CHECK: func.func @strided_window
// CHECK: %[[converted:.*]] = tensor_ext.convert_layout %{{.*}}
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[converted]],

// OPT-DAG: #[[polyphase:layout[0-9]*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (31i0 + 16ct + slot) mod 63 = 0 and 0 <= i0 <= 63 and 0 <= slot <= 15 and i0 - 32ct <= 2slot <= 63 + i0 - 32ct }">
// OPT: func.func @strided_window(%{{.*}}: !secret.secret<tensor<64xf32>> {{.*}}tensor_ext.layout = #[[polyphase]]}
// OPT-NOT: tensor_ext.convert_layout
// OPT: linalg.generic
func.func @strided_window(%arg0: !secret.secret<tensor<64xf32>>) -> !secret.secret<tensor<32xf32>> {
  %filter = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<32xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<64xf32>>) {
  ^body(%input0: tensor<64xf32>):
    %1 = linalg.generic {indexing_maps = [#window, #tap, #out], iterator_types = ["parallel", "reduction"]} ins(%input0, %filter : tensor<64xf32>, tensor<2xf32>) outs(%zero : tensor<32xf32>) {
    ^bb0(%in: f32, %w: f32, %acc: f32):
      %2 = arith.mulf %in, %w : f32
      %3 = arith.addf %acc, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<32xf32>
    secret.yield %1 : tensor<32xf32>
  } -> !secret.secret<tensor<32xf32>>
  return %0 : !secret.secret<tensor<32xf32>>
}

// -----

// With unit stride the input and the result have the same pitch, and the
// input keeps its layout.

#window = affine_map<(d0, d1) -> (d0 + d1)>
#tap = affine_map<(d0, d1) -> (d1)>
#out = affine_map<(d0, d1) -> (d0)>

// CHECK: func.func @unit_stride_window
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.generic
func.func @unit_stride_window(%arg0: !secret.secret<tensor<64xf32>>) -> !secret.secret<tensor<62xf32>> {
  %filter = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %zero = arith.constant dense<0.0> : tensor<62xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<64xf32>>) {
  ^body(%input0: tensor<64xf32>):
    %1 = linalg.generic {indexing_maps = [#window, #tap, #out], iterator_types = ["parallel", "reduction"]} ins(%input0, %filter : tensor<64xf32>, tensor<3xf32>) outs(%zero : tensor<62xf32>) {
    ^bb0(%in: f32, %w: f32, %acc: f32):
      %2 = arith.mulf %in, %w : f32
      %3 = arith.addf %acc, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<62xf32>
    secret.yield %1 : tensor<62xf32>
  } -> !secret.secret<tensor<62xf32>>
  return %0 : !secret.secret<tensor<62xf32>>
}
