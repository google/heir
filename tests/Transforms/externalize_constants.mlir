// RUN: rm -rf %t && mkdir %t
// RUN: heir-opt --externalize-constants="threshold-elements=3 output-dir=%t runtime-load-dir=runtime_dir" %s | FileCheck %s
// RUN: ls %t/constant_*.bin
// RUN: not heir-opt --externalize-constants="threshold-elements=3 output-dir=/nonexistent/dir" %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// CHECK: func.func @test_externalize
func.func @test_externalize() -> (tensor<2xi32>, tensor<4xi32>, tensor<4xi1>, tensor<4xi32>, tensor<4xi1>) {
  // CHECK-NEXT: %[[CST_SMALL:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
  %cst_small = arith.constant dense<[1, 2]> : tensor<2xi32>

  // CHECK-NEXT: %[[CST_LARGE_DEST:.*]] = tensor.empty() : tensor<4xi32>
  // CHECK-NEXT: %[[CST_LARGE:.*]] = preprocessing.load_resource "runtime_dir/constant_{{.*}}.bin" into %[[CST_LARGE_DEST]] : (tensor<4xi32>) -> tensor<4xi32>
  %cst_large = arith.constant dense<[3, 4, 5, 6]> : tensor<4xi32>

  // CHECK-NEXT: %[[CST_I1_DEST:.*]] = tensor.empty() : tensor<4xi1>
  // CHECK-NEXT: %[[CST_I1:.*]] = preprocessing.load_resource "runtime_dir/constant_{{.*}}.bin" into %[[CST_I1_DEST]] : (tensor<4xi1>) -> tensor<4xi1>
  %cst_i1 = arith.constant dense<[true, false, true, false]> : tensor<4xi1>

  // CHECK-NEXT: %[[CST_RES_I32_DEST:.*]] = tensor.empty() : tensor<4xi32>
  // CHECK-NEXT: %[[CST_RES_I32:.*]] = preprocessing.load_resource "runtime_dir/constant_{{.*}}.bin" into %[[CST_RES_I32_DEST]] : (tensor<4xi32>) -> tensor<4xi32>
  %cst_res_i32 = arith.constant dense_resource<resource_i32> : tensor<4xi32>

  // CHECK-NEXT: %[[CST_RES_I1_DEST:.*]] = tensor.empty() : tensor<4xi1>
  // CHECK-NEXT: %[[CST_RES_I1:.*]] = preprocessing.load_resource "runtime_dir/constant_{{.*}}.bin" into %[[CST_RES_I1_DEST]] : (tensor<4xi1>) -> tensor<4xi1>
  %cst_res_i1 = arith.constant dense_resource<resource_i1> : tensor<4xi1>

  // CHECK-NEXT: return %[[CST_SMALL]], %[[CST_LARGE]], %[[CST_I1]], %[[CST_RES_I32]], %[[CST_RES_I1]]
  return %cst_small, %cst_large, %cst_i1, %cst_res_i32, %cst_res_i1 : tensor<2xi32>, tensor<4xi32>, tensor<4xi1>, tensor<4xi32>, tensor<4xi1>
}

// CHECK-ERROR: Failed to open file for writing

{-#
  dialect_resources: {
    builtin: {
      resource_i32: "0x0400000001000000020000000300000004000000",
      resource_i1: "0x0100000001000100"
    }
  }
#-}
