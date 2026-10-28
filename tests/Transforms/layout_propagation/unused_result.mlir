// RUN: heir-opt --layout-propagation=min-slot-count=32 %s | FileCheck %s

// A row max that also tracks the argmax (as torch emits for log_softmax): the
// argmax and its init are dropped, leaving a single-result max reduction for
// the generic kernel.

// CHECK: func.func @row_max
// CHECK: linalg.generic
// CHECK-NOT: tensor<2xi64>
// CHECK: arith.maximumf
// CHECK: return
func.func @row_max(%arg0: !secret.secret<tensor<2x16xf32>>) -> !secret.secret<tensor<2xf32>> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<2xi64>) -> tensor<2xi64>
  %2 = tensor.empty() : tensor<2xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2xf32>) -> tensor<2xf32>
  %r = secret.generic(%arg0: !secret.secret<tensor<2x16xf32>>) {
  ^body(%x: tensor<2x16xf32>):
    %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<2x16xf32>) outs(%3, %1 : tensor<2xf32>, tensor<2xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maximumf %in, %out : f32
      %8 = arith.cmpf ogt, %in, %out : f32
      %9 = arith.select %8, %6, %out_1 : i64
      linalg.yield %7, %9 : f32, i64
    } -> (tensor<2xf32>, tensor<2xi64>)
    secret.yield %4#0 : tensor<2xf32>
  } -> !secret.secret<tensor<2xf32>>
  return %r : !secret.secret<tensor<2xf32>>
}

// The same after compare-to-sign-rewrite: the argmax accumulator reaches the
// yield through a chain of ops.

// CHECK: func.func @row_max_arithmetic_select
// CHECK: linalg.generic
// CHECK-NOT: tensor<2xi64>
// CHECK: arith.maximumf
// CHECK: return
func.func @row_max_arithmetic_select(%arg0: !secret.secret<tensor<2x16xf32>>) -> !secret.secret<tensor<2xf32>> {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<2xi64>) -> tensor<2xi64>
  %2 = tensor.empty() : tensor<2xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2xf32>) -> tensor<2xf32>
  %r = secret.generic(%arg0: !secret.secret<tensor<2x16xf32>>) {
  ^body(%x: tensor<2x16xf32>):
    %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<2x16xf32>) outs(%3, %1 : tensor<2xf32>, tensor<2xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maximumf %in, %out : f32
      %8 = arith.cmpf ogt, %in, %out : f32
      %9 = arith.extui %8 : i1 to i64
      %10 = arith.subi %c1_i64, %9 : i64
      %11 = arith.muli %9, %6 : i64
      %12 = arith.muli %10, %out_1 : i64
      %13 = arith.addi %11, %12 : i64
      linalg.yield %7, %13 : f32, i64
    } -> (tensor<2xf32>, tensor<2xi64>)
    secret.yield %4#0 : tensor<2xf32>
  } -> !secret.secret<tensor<2xf32>>
  return %r : !secret.secret<tensor<2xf32>>
}
