// RUN: heir-opt --secret-insert-mgmt-ckks --optimize-relinearization %s | FileCheck %s

// Ensure that optimize-relinearization handles operations with multiple results.
module {
  // CHECK-LABEL: func @loop
  // CHECK: affine.for
  // CHECK-NOT: mgmt.relinearize
  // CHECK: affine.yield
  // CHECK: return
  func.func @loop(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1 iter_args(%arg2 = %cst, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %3 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %4 = arith.addf %arg2, %3 : tensor<1x1024xf32>
        %5 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %4, %5 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      secret.yield %1#0 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
