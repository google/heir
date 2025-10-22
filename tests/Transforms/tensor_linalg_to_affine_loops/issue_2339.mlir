// RUN: heir-opt --tensor-linalg-to-affine-loops %s | FileCheck %s

#map = affine_map<() -> ()>
module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<i32>)
  // CHECK: %[[v0:.*]] = tensor.empty() : tensor<i8>
  // CHECK: %[[extracted:.*]] = tensor.extract %[[arg0]][] : tensor<i32>
  // CHECK: %[[v1:.*]] = arith.extsi %[[extracted]] : i32 to i64
  // CHECK: return
  func.func @main(%arg0: tensor<i32>) -> tensor<1x1xi8> {
    %c34359738368_i64 = arith.constant 34359738368 : i64
    %c36_i64 = arith.constant 36 : i64
    %c1630361836_i64 = arith.constant 1630361836 : i64
    %c5_i32 = arith.constant 5 : i32
    %c127_i32 = arith.constant 127 : i32
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %0 = tensor.empty() : tensor<i8>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<i32>) outs(%0 : tensor<i8>) {
    ^bb0(%in: i32, %out: i8):
      %2 = arith.extsi %in : i32 to i64
      %3 = arith.muli %2, %c1630361836_i64 : i64
      %4 = arith.addi %3, %c34359738368_i64 : i64
      %5 = arith.cmpi sge, %in, %c0_i32 : i32
      %6 = arith.select %5, %c1073741824_i64, %c-1073741824_i64 : i64
      %7 = arith.addi %6, %4 : i64
      %8 = arith.shrsi %7, %c36_i64 : i64
      %9 = arith.trunci %8 : i64 to i32
      %10 = arith.addi %9, %c5_i32 : i32
      %11 = arith.maxsi %10, %c-128_i32 : i32
      %12 = arith.minsi %11, %c127_i32 : i32
      %13 = arith.trunci %12 : i32 to i8
      linalg.yield %13 : i8
    } -> tensor<i8>
    %expanded = tensor.expand_shape %1 [] output_shape [1, 1] : tensor<i8> into tensor<1x1xi8>
    return %expanded : tensor<1x1xi8>
  }
}
