// RUN: heir-opt --softmax-canonicalizations --canonicalize %s | FileCheck %s

// CHECK: func.func @composed_softmax(
// CHECK-SAME: %[[ARG0:.*]]: tensor<64xf32> {secret.secret}) -> tensor<64xf32> {
// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<64xf32>
// CHECK: %[[PRE_OP:.*]] = arith.addf %[[ARG0]], %[[CST]] : tensor<64xf32>
// CHECK-NOT: linalg.broadcast
// CHECK-NOT: linalg.reduce
// CHECK-NOT: math.exp
// CHECK-NOT: linalg.generic
// CHECK-NOT: arith.divf
// CHECK: %[[SOFTMAX:.*]] = math_ext.softmax %[[PRE_OP]] {dimension = 0 : i64} : tensor<64xf32>
// CHECK: %[[POST_OP:.*]] = arith.addf %[[SOFTMAX]], %[[CST]] : tensor<64xf32>
// CHECK: return %[[POST_OP]] : tensor<64xf32>
// CHECK: }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

module {
  func.func @composed_softmax(%arg0: tensor<64xf32> {secret.secret}) -> tensor<64xf32> {
    %dim = arith.constant dense<0xFF800000> : tensor<f32>
    %zero = arith.constant dense<0.000000e+00> : tensor<f32>
    %idx_zero = arith.constant dense<0> : tensor<i64>

    // Add 2.0 to the input before softmax.
    %two = arith.constant dense<2.0> : tensor<64xf32>
    %pre_op = arith.addf %arg0, %two : tensor<64xf32>

    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%pre_op : tensor<64xf32>) outs(%dim, %idx_zero : tensor<f32>, tensor<i64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_1 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    %empty1 = tensor.empty() : tensor<64xf32>
    %broadcast_max = linalg.broadcast ins(%0#0 : tensor<f32>) outs(%empty1 : tensor<64xf32>) dimensions = [0]

    %sub = arith.subf %pre_op, %broadcast_max : tensor<64xf32>

    %exp = math.exp %sub : tensor<64xf32>

    %reduce = linalg.reduce ins(%exp : tensor<64xf32>) outs(%zero : tensor<f32>) dimensions = [0]
      ( %in: f32, %init: f32 ) {
        %add = arith.addf %in, %init : f32
        linalg.yield %add : f32
      }

    %empty2 = tensor.empty() : tensor<64xf32>
    %broadcast_sum = linalg.broadcast ins(%reduce : tensor<f32>) outs(%empty2 : tensor<64xf32>) dimensions = [0]

    %div = arith.divf %exp, %broadcast_sum : tensor<64xf32>

    // Add 2.0 to the output after softmax.
    %post_op = arith.addf %div, %two : tensor<64xf32>

    return %post_op : tensor<64xf32>
  }
}
