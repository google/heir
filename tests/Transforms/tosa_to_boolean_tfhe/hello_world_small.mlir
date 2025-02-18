// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// A reduced dimension version of hello world to speed Yosys up.

// CHECK-LABEL: module
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0)>
module {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x1xi8> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<[[-9, -54, 57]]> : tensor<1x3xi8>
    %cst_0 = arith.constant dense<[0, 0, -5438]> : tensor<3xi32>
    %cst_1 = arith.constant dense<[-729, 1954, 610]> : tensor<3xi32>
    %cst_2 = arith.constant dense<429> : tensor<1xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c127_i32 = arith.constant 127 : i32
    %c5_i32 = arith.constant 5 : i32
    %c2039655736_i64 = arith.constant 2039655736 : i64
    %c38_i64 = arith.constant 38 : i64
    %c137438953472_i64 = arith.constant 137438953472 : i64
    %cst_3 = arith.constant dense<[[-12, 9, -12], [26, 25, 36], [-19, 33, -32]]> : tensor<3x3xi8>
    %c1561796795_i64 = arith.constant 1561796795 : i64
    %c37_i64 = arith.constant 37 : i64
    %c68719476736_i64 = arith.constant 68719476736 : i64
    %cst_4 = arith.constant dense<[[-39], [59], [39]]> : tensor<3x1xi8>
    %c1630361836_i64 = arith.constant 1630361836 : i64
    %c36_i64 = arith.constant 36 : i64
    %c34359738368_i64 = arith.constant 34359738368 : i64
    %0 = tensor.empty() : tensor<1x3xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<3xi32>) outs(%0 : tensor<1x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x3xi32>
    %2 = linalg.quantized_matmul ins(%arg0, %cst, %c-128_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x3xi8>, i32, i32) outs(%1 : tensor<1x3xi32>) -> tensor<1x3xi32>
    %3 = tensor.empty() : tensor<1x3xi8>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x3xi32>) outs(%3 : tensor<1x3xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c2039655736_i64 : i64
      %15 = arith.addi %14, %c137438953472_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c38_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c-128_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x3xi8>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<3xi32>) outs(%0 : tensor<1x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x3xi32>
    %6 = linalg.quantized_matmul ins(%4, %cst_3, %c-128_i32, %c0_i32 : tensor<1x3xi8>, tensor<3x3xi8>, i32, i32) outs(%5 : tensor<1x3xi32>) -> tensor<1x3xi32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x3xi32>) outs(%3 : tensor<1x3xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c1561796795_i64 : i64
      %15 = arith.addi %14, %c68719476736_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c37_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c-128_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x3xi8>
    %8 = tensor.empty() : tensor<1x1xi32>
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1xi32>) outs(%8 : tensor<1x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x1xi32>
    %10 = linalg.quantized_matmul ins(%7, %cst_4, %c-128_i32, %c0_i32 : tensor<1x3xi8>, tensor<3x1xi8>, i32, i32) outs(%9 : tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tensor.empty() : tensor<1x1xi8>
    %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x1xi32>) outs(%11 : tensor<1x1xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c1630361836_i64 : i64
      %15 = arith.addi %14, %c34359738368_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c36_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c5_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x1xi8>
    // CHECK: return
    return %12 : tensor<1x1xi8>
  }
}
