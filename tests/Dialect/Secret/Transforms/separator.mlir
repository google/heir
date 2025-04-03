// Tests whether secret.separator can be bufferized.

// RUN: heir-opt --one-shot-bufferize %s | FileCheck %s

// CHECK: module
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
module attributes {tf_saved_model.semantics} {
  // CHECK: @bufferize
  func.func @bufferize(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x16xi8> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> : tensor<16x1xi8>
    %cst_0 = arith.constant dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> : tensor<16xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c127_i32 = arith.constant 127 : i32
    %c2039655736_i64 = arith.constant 2039655736 : i64
    %c38_i64 = arith.constant 38 : i64
    %c137438953472_i64 = arith.constant 137438953472 : i64
    %0 = bufferization.alloc_tensor() : tensor<1x16xi8>
    %transposed = linalg.transpose ins(%cst : tensor<16x1xi8>) outs(%0 : tensor<1x16xi8>) permutation = [1, 0]
    %1 = bufferization.alloc_tensor() : tensor<1x16xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<16xi32>) outs(%1 : tensor<1x16xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x16xi32>
    // CHECK: linalg.quantized_matmul
    // CHECK-SAME: outs(%[[v1:.*]])
    %3 = linalg.quantized_matmul ins(%arg0, %transposed, %c-128_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x16xi8>, i32, i32) outs(%2 : tensor<1x16xi32>) -> tensor<1x16xi32>
    // CHECK: secret.separator %[[v1]]
    secret.separator %3 : tensor<1x16xi32>
    %4 = bufferization.alloc_tensor() : tensor<1x16xi8>
    // CHECK: linalg.generic
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x16xi32>) outs(%4 : tensor<1x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %8 = arith.extsi %in : i32 to i64
      %9 = arith.muli %8, %c2039655736_i64 : i64
      %10 = arith.addi %9, %c137438953472_i64 : i64
      %11 = arith.cmpi sge, %in, %c0_i32 : i32
      %12 = arith.select %11, %c1073741824_i64, %c-1073741824_i64 : i64
      %13 = arith.addi %12, %10 : i64
      %14 = arith.shrsi %13, %c38_i64 : i64
      %15 = arith.trunci %14 : i64 to i32
      %16 = arith.addi %15, %c-128_i32 : i32
      %17 = arith.maxsi %16, %c-128_i32 : i32
      %18 = arith.minsi %17, %c127_i32 : i32
      %19 = arith.trunci %18 : i32 to i8
      linalg.yield %19 : i8
    } -> tensor<1x16xi8>
    %6 = bufferization.alloc_tensor() : tensor<1x16xi8>
    // CHECK: linalg.generic
    // CHECK-SAME: outs(%[[v2:.*]])
    %7 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xi8>) outs(%6 : tensor<1x16xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<1x16xi8>
    // CHECK: secret.separator %[[v2]]
    secret.separator %7 : tensor<1x16xi8>
    // CHECK-NEXT: return
    return %7 : tensor<1x16xi8>
  }
}
