// RUN: rm -rf %t && mkdir %t
// RUN: heir-opt --externalize-constants="threshold-elements=4 output-dir=%t runtime-load-dir=runtime_dir" %s | FileCheck %s
// RUN: ls %t/*.bin | count 1

// CHECK-LABEL: func.func @splat
func.func @splat() -> (tensor<4xi32>, tensor<4xi1>, tensor<4xi32>) {
  // CHECK-NOT: preprocessing.load_resource
  // CHECK: arith.constant dense<0> : tensor<4xi32>
  %splat_i32 = arith.constant dense<0> : tensor<4xi32>

  // CHECK: arith.constant dense<true> : tensor<4xi1>
  %splat_i1 = arith.constant dense<true> : tensor<4xi1>

  // A non-splat of the same size is still externalized, so the skip above is
  // not disabling the pass wholesale.
  // CHECK: preprocessing.load_resource "runtime_dir/constant_{{.*}}.bin" : tensor<4xi32>
  %dense_i32 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  return %splat_i32, %splat_i1, %dense_i32 : tensor<4xi32>, tensor<4xi1>, tensor<4xi32>
}
